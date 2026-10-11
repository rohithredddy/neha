import json

import torch

from safedti import train
from safedti.utils.config import load_config
from safedti.utils.io import write_result
from safedti.utils.seeding import seed_everything


def test_base_config_loads_and_overrides():
    cfg = load_config("configs/base.yaml", ["loss.lambda.rank=0.25"])
    assert cfg["loss"]["lambda"]["rank"] == 0.25
    assert cfg["model"]["hidden"] == 128
    assert "con" not in cfg["loss"]["lambda"]  # spec §4b dropped (DECISIONS 2026-10-11)


def test_dry_run_exits_zero():
    assert train.main(["--config", "configs/base.yaml", "--seed", "0", "--dry-run"]) == 0


def test_seeding_is_deterministic():
    seed_everything(3)
    a = torch.rand(5)
    seed_everything(3)
    assert torch.equal(a, torch.rand(5))


def test_write_result(tmp_path):
    p = write_result("x", {"a": 1}, 0, {"auroc": 0.5}, 1.0, out_dir=tmp_path)
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["seed"] == 0 and rec["metrics"]["auroc"] == 0.5 and "git_commit" in rec
