from pathlib import Path

from moe_optimizator.bench.parser import parse_bench_json
from moe_optimizator.optimizer.phases import rank_configurations

FIXTURE = Path(__file__).parent / "fixtures" / "bench_sample.json"


def test_parse_bench_json():
    records = parse_bench_json(FIXTURE.read_text())
    assert len(records) == 2
    assert records[0].is_tg
    assert records[0].metric_id == "tg@ctx:n128"
    assert records[0].avg_ts == 128.0


def test_rank_prefers_higher_tg_at_ctx():
    records = parse_bench_json(FIXTURE.read_text())
    ranked = rank_configurations(records, ctx_max=65536)
    assert ranked[0][1]["tg@ctx"] == 128.0
    assert ranked[0][1]["tg@ctx"] > ranked[1][1]["tg@ctx"]
