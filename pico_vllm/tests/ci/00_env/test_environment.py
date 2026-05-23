import sys

import pytest
import torch

from ci_utils import detect_environment


pytestmark = pytest.mark.env


def test_environment_detection():
    env = detect_environment()

    assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {env.python}"
    assert env.pytest_available, "pytest is required for local CI"
    assert env.transformers_available, "transformers is required"
    assert env.safetensors_available, "safetensors is required"

    if env.cuda_available:
        assert env.gpu_count > 0
        assert env.triton_available, "triton is required for CUDA operator tests"

    print(
        "\n"
        f"python={env.python}\n"
        f"torch={env.torch_version}\n"
        f"cuda_available={env.cuda_available}\n"
        f"gpu_count={env.gpu_count}\n"
        f"triton_available={env.triton_available}\n"
        f"pytest_available={env.pytest_available}\n"
        f"weights_available={env.weights_available}\n"
        f"torchrun_available={env.torchrun_available}"
    )


def test_project_imports():
    from blockmanager import BlockManager
    from cache import PagedKVCache
    from model import ModelConfig, Qwen25_15B
    from ops import get_ops_backend

    assert BlockManager is not None
    assert PagedKVCache is not None
    assert ModelConfig is not None
    assert Qwen25_15B is not None
    assert get_ops_backend(torch.device("cpu")).device_type == "cpu"
