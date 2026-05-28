"""Series for optimization result charts (no GUI dependencies)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankingChartSeries:
    """Top configurations for bar charts."""

    labels: list[str]
    tg_ctx: list[float]
    pp_ctx: list[float]
    pg_ctx: list[float]


def config_short_label(key: tuple, rank: int) -> str:
    """Compact label for a configuration key from ``rank_configurations``."""
    ngl, threads, batch, ubatch = key[0], key[1], key[2], key[3]
    fa = key[6]
    return f"#{rank} ngl={ngl} t={threads} b={batch}/{ubatch} fa={fa}"


def ranking_chart_series(
    ranked: list[tuple],
    *,
    limit: int = 15,
) -> RankingChartSeries | None:
    """Build bar-chart data from ranked configurations (newest rank first in list)."""
    if not ranked:
        return None

    labels: list[str] = []
    tg: list[float] = []
    pp: list[float] = []
    pg: list[float] = []

    for i, (key, metrics, _) in enumerate(ranked[:limit]):
        labels.append(config_short_label(key, i + 1))
        tg.append(float(metrics.get("tg@ctx") or metrics.get("tg@cold") or 0.0))
        pp.append(float(metrics.get("pp@ctx") or metrics.get("pp@cold") or 0.0))
        pg.append(float(metrics.get("pg@ctx") or metrics.get("pg@cold") or 0.0))

    return RankingChartSeries(labels=labels, tg_ctx=tg, pp_ctx=pp, pg_ctx=pg)
