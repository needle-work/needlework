"""Simulator worker processes (spawned), one per slot, driven over pipes."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from needlework.constants import REPLY_TIMEOUT_S, SHUTDOWN_TIMEOUT_S
from needlework.geometry.actions import ActionLayout
from needlework.sim.factory import make_env
from needlework.sim.wrappers import (
    ChunkRunner,
    EpisodeEnd,
    PolicyActions,
    SeededReset,
    StoreObservations,
)


@dataclass(frozen=True)
class WorkerSpec:
    hdf5: Path
    cameras: tuple[str, ...]
    proprio: tuple[str, ...]
    arms: tuple[str, ...]
    max_steps: int
    n_obs: int
    construction_seed: int
    egl_index: int


def build(spec: WorkerSpec) -> ChunkRunner:
    env = make_env(
        hdf5=spec.hdf5,
        cameras=spec.cameras,
        proprio=spec.proprio,
        construction_seed=spec.construction_seed,
        egl_index=spec.egl_index,
    )
    env = StoreObservations(SeededReset(env), spec.cameras, spec.proprio)
    env = EpisodeEnd(PolicyActions(env, ActionLayout(spec.arms)), spec.max_steps)
    return ChunkRunner(env, spec.n_obs)


def _serve(conn: Connection, spec: WorkerSpec) -> None:
    try:
        env = build(spec)
        conn.send(("ok", None))
        while True:
            command, argument = conn.recv()
            if command == "close":
                return
            if command == "reset":
                conn.send(("ok", env.reset(argument)))
            elif command == "step":
                obs, done = env.step(argument)
                conn.send(("ok", (obs, done, env.result())))
            else:
                raise ValueError(f"unknown command {command}")
    except Exception:
        conn.send(("error", traceback.format_exc()))


class SimPool:
    def __init__(self, specs: list[WorkerSpec]) -> None:
        context = mp.get_context("spawn")
        self.conns: list[Connection] = []
        self.procs: list[Any] = []
        for spec in specs:
            parent, child = context.Pipe()
            proc = context.Process(target=_serve, args=(child, spec), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)
        try:
            for slot in range(len(specs)):
                self.recv(slot)
        except BaseException:
            for proc in self.procs:
                if proc.is_alive():
                    proc.terminate()
                proc.join(timeout=SHUTDOWN_TIMEOUT_S)
            for conn in self.conns:
                conn.close()
            raise

    def send(self, slot: int, command: str, argument: Any = None) -> None:
        self.conns[slot].send((command, argument))

    def recv(self, slot: int) -> Any:
        """The slot's reply. A dead worker closes its end of the pipe, so it is noticed
        at once, not after the timeout, which only catches a live worker that hangs."""
        if not self.conns[slot].poll(REPLY_TIMEOUT_S):
            raise TimeoutError(
                f"simulator slot {slot} gave no reply in {REPLY_TIMEOUT_S} s"
            )
        try:
            status, payload = self.conns[slot].recv()
        except EOFError:
            proc = self.procs[slot]
            proc.join(timeout=SHUTDOWN_TIMEOUT_S)
            raise RuntimeError(
                f"simulator slot {slot} exited with code {proc.exitcode} "
                "without replying"
            ) from None
        if status == "error":
            raise RuntimeError(f"simulator slot {slot} failed:\n{payload}")
        return payload

    def close(self) -> None:
        for conn, proc in zip(self.conns, self.procs, strict=True):
            if proc.is_alive():
                conn.send(("close", None))
            proc.join(timeout=SHUTDOWN_TIMEOUT_S)
            if proc.is_alive():
                proc.kill()
