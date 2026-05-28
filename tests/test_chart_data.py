from moe_optimizator.optimizer.chart_data import config_short_label, ranking_chart_series


def test_ranking_chart_series_empty() -> None:
    assert ranking_chart_series([]) is None


def test_ranking_chart_series_top() -> None:
    key = (99, 8, 2048, 512, "f16", "f16", True, False, True, "layer", 0, 50)
    ranked = [
        (key, {"tg@ctx": 120.5, "pp@ctx": 80.0, "pg@ctx": 90.0}, 1.0),
    ]
    series = ranking_chart_series(ranked, limit=5)
    assert series is not None
    assert len(series.labels) == 1
    assert series.tg_ctx[0] == 120.5
    assert "ngl=99" in series.labels[0]


def test_config_short_label() -> None:
    key = (32, 4, 512, 256, "f16", "f16", False, False, True, "layer", 0, 50)
    label = config_short_label(key, 3)
    assert label.startswith("#3")
    assert "ngl=32" in label
