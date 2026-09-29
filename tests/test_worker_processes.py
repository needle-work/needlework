"""Simulator and stitch worker processes: death, hangs and slow exits are reported
promptly, by slot or GPU, and never confused with each other."""

import multiprocessing as mp
import os
import select
import sys
import time

import pytest

from needlework.sim import vector_env
from needlework.stitching import workers


def _pool(target, args_of_child) -> vector_env.SimPool:
    """A one-slot SimPool whose worker runs ``target`` (not a simulator)."""
    ctx = mp.get_context("spawn")
    pool = object.__new__(vector_env.SimPool)
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=target, args=args_of_child(child), daemon=True)
    proc.start()
    child.close()
    pool.conns, pool.procs = [parent], [proc]
    return pool


def test_dead_simulator_worker_is_named_with_its_exit_code() -> None:
    pool = _pool(sys.exit, lambda child: (child,))  # holds the pipe, exits, no reply
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match=r"slot 0 exited with code 1"):
        pool.recv(0)
    assert time.perf_counter() - started < 30


def test_silent_live_simulator_worker_times_out(monkeypatch) -> None:
    monkeypatch.setattr(vector_env, "REPLY_TIMEOUT_S", 2)
    pool = _pool(select.select, lambda child: ([child], [], []))  # blocks forever
    with pytest.raises(TimeoutError, match="slot 0"):
        pool.recv(0)
    pool.procs[0].kill()


def _start(target, args):
    proc = mp.get_context("spawn").Process(target=target, args=args, daemon=True)
    proc.start()
    return proc


def test_stitch_worker_shutdown_distinguishes_hung_from_failed() -> None:
    clean = [_start(os._exit, (0,)), _start(os._exit, (0,))]
    workers.stop_workers(clean, [0, 1], grace_s=30)  # a clean exit is never an error
    hung = _start(time.sleep, (600,))
    with pytest.raises(RuntimeError, match=r"GPU 3 still running after 2 s"):
        workers.stop_workers([hung], [3], grace_s=2)
    assert not hung.is_alive()  # killed, not left behind
    failed = _start(os._exit, (7,))
    with pytest.raises(RuntimeError, match=r"GPU 5 exited with code 7"):
        workers.stop_workers([failed], [5], grace_s=30)
