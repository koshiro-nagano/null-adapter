import copy
from pathlib import Path

import pytest
import yaml

CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture(scope="session")
def cfg():
    from src.config import load_config
    return load_config(CFG_PATH)


@pytest.fixture(scope="session")
def world(cfg):
    from src.data import build_world
    return build_world(cfg)


@pytest.fixture(scope="session")
def noisy_world(cfg):
    """Larger observation noise so the posterior-vs-MC check is non-trivial."""
    from src.data import build_world
    c = copy.deepcopy(cfg)
    c["noise_sigma"] = 0.02
    return build_world(c)


@pytest.fixture(scope="session")
def mog_cfg():
    from src.config import load_config
    return load_config(CFG_PATH.parent / "mog.yaml")


@pytest.fixture(scope="session")
def mog_world(mog_cfg):
    from src.data import build_world
    return build_world(mog_cfg)
