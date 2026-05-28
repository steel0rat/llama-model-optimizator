from pathlib import Path

from moe_optimizator.optimizer.config import OptimizationConfig, SearchSpace
from moe_optimizator.optimizer.phases import (
    PHASE1_DEPTH,
    config_key_to_bench_args,
    default_config_key,
    inference_tuning_matrix,
    tuning_depth_for,
)


def test_config_key_to_bench_args() -> None:
    key = (99, 8, 2048, 512, "f16", "q8_0", True, False, True, "layer", 0, 50)
    args = config_key_to_bench_args(key)
    assert "-ngl" in args and "99" in args
    assert "-fa" in args and "1" in args


def test_default_config_key_uses_first_values() -> None:
    space = SearchSpace(n_threads=[4, 8], flash_attn=[0, 1])
    key = default_config_key(space)
    assert key[1] == 4
    assert key[6] is False


def test_tuning_depth_is_zero() -> None:
    cfg = OptimizationConfig(model=Path("m.gguf"), llama_bench=Path("bench"), ctx_max=65536)
    assert tuning_depth_for(cfg) == PHASE1_DEPTH == 0


def test_inference_tuning_matrix_uses_d_zero() -> None:
    cfg = OptimizationConfig(model=Path("m.gguf"), llama_bench=Path("bench"))
    matrix = inference_tuning_matrix(cfg)
    assert len(matrix) == 3
    assert all(args[-1] == "0" for args in matrix)
