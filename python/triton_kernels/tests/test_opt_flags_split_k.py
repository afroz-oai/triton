# isort: off
# fmt: off
import types

import pytest

torch = pytest.importorskip("torch")

import triton_kernels.matmul_ogs_details.opt_flags as opt_flags


class _DummyPrecisionConfig:
    def __init__(self):
        self.weight_scale = None
        self.max_num_imprecise_acc = None
        self.act_scale = None
        self.out_scale = None
        self.enforce_bitwise_invariance = False


def _stub_cuda_props(*_args, **_kwargs):
    return types.SimpleNamespace(multi_processor_count=16)


def test_make_default_opt_flags_amd_split_k_callable(monkeypatch):
    monkeypatch.setattr(opt_flags, "get_cdna_version", lambda: 3)
    monkeypatch.setattr(opt_flags.torch.cuda, "get_device_properties", _stub_cuda_props)
    monkeypatch.setattr(
        opt_flags.opt_flags_amd,
        "compute_block_nk",
        lambda *args, **kwargs: (64, 32),
    )

    captured_args = {}

    def split_k_callable(batch_size, m, n, k, out_dtype):
        captured_args["value"] = (batch_size, m, n, k, out_dtype)
        return 5

    precision_config = _DummyPrecisionConfig()
    flags = opt_flags.make_default_opt_flags_amd(
        torch.float16,
        torch.float16,
        torch.float16,
        precision_config,
        2,
        128,
        64,
        32,
        None,
        False,
        False,
        False,
        0,
        False,
        False,
        {"split_k": split_k_callable},
    )

    assert flags.split_k == 5
    assert captured_args["value"] == (2, 128, 64, 32, torch.float16)


def test_make_default_opt_flags_nvidia_split_k_callable(monkeypatch):
    monkeypatch.setattr(opt_flags.torch.cuda, "get_device_properties", _stub_cuda_props)
    monkeypatch.setattr(opt_flags.torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(
        opt_flags.opt_flags_nvidia,
        "compute_block_n",
        lambda n, arch, precision_config: (64, 32),
    )
    monkeypatch.setattr(
        opt_flags.opt_flags_nvidia,
        "compute_grid_size",
        lambda routing_data, batch_size, m, n, block_m, block_n: 4,
    )
    monkeypatch.setattr(
        opt_flags.opt_flags_nvidia,
        "compute_block_k",
        lambda m, k, is_persistent, lhs_dtype, rhs_dtype, precision_config, has_y_acc_in: 32,
    )
    monkeypatch.setattr(
        opt_flags.opt_flags_nvidia,
        "compute_split_k",
        lambda block_k, k, estimated_actual_grid_size: 1,
    )
    monkeypatch.setattr(
        opt_flags.opt_flags_nvidia,
        "compute_num_stages",
        lambda *args, **kwargs: 2,
    )
    monkeypatch.setattr(
        opt_flags.opt_flags_nvidia,
        "compute_num_warps",
        lambda block_m, block_n, is_persistent, precision_config: 4,
    )

    captured_args = {}

    def split_k_callable(batch_size, m, n, k, out_dtype):
        captured_args["value"] = (batch_size, m, n, k, out_dtype)
        return 3

    precision_config = _DummyPrecisionConfig()
    flags = opt_flags.make_default_opt_flags_nvidia(
        torch.float16,
        torch.float16,
        torch.float16,
        precision_config,
        4,
        256,
        128,
        64,
        None,
        False,
        False,
        False,
        0,
        False,
        False,
        {"split_k": split_k_callable},
    )

    assert flags.split_k == 3
    assert captured_args["value"] == (4, 256, 128, 64, torch.float16)
