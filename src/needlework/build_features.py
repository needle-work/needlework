"""Build the DINOv3 feature caches of one task.

    python -m needlework.build_features --domain robomimic --task square \
        --poolings spatial_softmax_14 --device cuda:0

Policy and IDM read ``constants.SPATIAL``; the verifier reads ``constants.PATCHES``.
Batch size and decode threads only change speed, never the features.
"""

from __future__ import annotations

import argparse

import torch

from needlework.data import features
from needlework.data.store import EpisodeStore
from needlework.models.dinov3 import POOLINGS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", required=True, choices=["robomimic", "umi"])
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--poolings", required=True, nargs="+", choices=sorted(POOLINGS)
    )
    parser.add_argument("--device", required=True, help="a CUDA device, e.g. cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--decode-threads", type=int, default=8)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"--device must be a CUDA device, got {args.device}")
    features.build(
        EpisodeStore.open(args.domain, args.task),
        tuple(args.poolings),
        device=device,
        batch_size=args.batch_size,
        decode_threads=args.decode_threads,
    )


if __name__ == "__main__":
    main()
