"""Training entry point. Stage 1 (grounded) and stage 2 (binary-only transfer).

Only the CLI and dry-run path exist so far (milestone M0); the training loop lands in M4/M5.
"""

from __future__ import annotations

import argparse
import sys
import time

from safedti.utils.config import load_config
from safedti.utils.seeding import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--stage", type=int, choices=[1, 2], default=1)
    p.add_argument("--data", default="ib_id")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--set", nargs="*", default=[], help="overrides, e.g. loss.lambda.rank=0.5")
    p.add_argument("--dry-run", action="store_true", help="resolve config and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)
    cfg = load_config(args.config, args.set)
    start = time.time()
    if args.dry_run:
        print(
            f"[dry-run] stage={args.stage} data={args.data} seed={args.seed} "
            f"model={cfg['model']['name']} ({time.time() - start:.2f}s)"
        )
        return 0
    raise NotImplementedError("Training loop is implemented in milestones M4/M5.")


if __name__ == "__main__":
    sys.exit(main())
