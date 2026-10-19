"""Cost per question vs execution accuracy, from committed run summaries.

Writes two static SVGs (light and dark) for the README. Each point is one (model, system)
pair: x = mean cost per question in USD (log scale), y = EX with its 95% bootstrap CI.
Colour encodes the model, marker shape the system, and every point is labelled directly.

    uv run python -m evals.text2sql.plot_cost_ex --out docs/img/cost-vs-ex
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

REPORTS = Path(__file__).resolve().parent.parent / "reports"

# (model label, report directory, systems to plot)
SOURCES = [
    ("Haiku 5.5", "2026-10-19-haiku-test-350", ["baseline", "agent_v2"]),
    ("Sonnet 5.5", "2026-10-19-sonnet-test-350", ["baseline", "agent_v2"]),
]
SYSTEM_LABELS = {"baseline": "single shot", "agent_v2": "agent v2"}

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e6e5e1",
        "series": ["#2a78d6", "#eb6834"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#33332f",
        "series": ["#3987e5", "#d95926"],
    },
}

W, H = 760, 440
LEFT, RIGHT, TOP, BOTTOM = 72, 150, 84, 64


@dataclass
class Point:
    model: str
    model_index: int
    system: str
    cost: float
    ex: float
    lo: float
    hi: float
    n: int


def load_points() -> list[Point]:
    points = []
    for i, (model, report, systems) in enumerate(SOURCES):
        summary = json.loads((REPORTS / report / "summary.json").read_text(encoding="utf-8"))
        for system in systems:
            s = summary["systems"][system]
            lo, hi = s["ex_ci95"]
            points.append(
                Point(
                    model=model,
                    model_index=i,
                    system=system,
                    cost=s["cost_usd"]["mean_per_question"],
                    ex=s["ex"],
                    lo=lo,
                    hi=hi,
                    n=s["n"],
                )
            )
    return points


def _nice_log_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10**e
            if lo <= v <= hi:
                ticks.append(v)
    return ticks


def _fmt_usd(v: float) -> str:
    return f"${v:.4f}".rstrip("0").rstrip(".") if v < 0.01 else f"${v:.2f}"


def render(points: list[Point], theme: dict[str, object]) -> str:
    series = theme["series"]
    assert isinstance(series, list)
    x_lo = min(p.cost for p in points) / 1.6
    x_hi = max(p.cost for p in points) * 1.6
    y_lo = math.floor(min(p.lo for p in points) * 10) / 10
    y_hi = math.ceil(max(p.hi for p in points) * 10) / 10
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM

    def sx(v: float) -> float:
        return LEFT + (math.log10(v) - math.log10(x_lo)) / (math.log10(x_hi / x_lo)) * pw

    def sy(v: float) -> float:
        return TOP + (y_hi - v) / (y_hi - y_lo) * ph

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" '
        f'role="img" aria-labelledby="t d">',
        "<title id='t'>Cost per question vs execution accuracy, BIRD Mini-Dev test split</title>",
        "<desc id='d'>"
        + "; ".join(
            f"{p.model} {SYSTEM_LABELS[p.system]}: EX {p.ex:.1%} "
            f"(95% CI {p.lo:.1%} to {p.hi:.1%}), {_fmt_usd(p.cost)} per question"
            for p in points
        )
        + "</desc>",
        f'<rect width="{W}" height="{H}" fill="{theme["surface"]}"/>',
        f'<text x="{LEFT}" y="26" font-size="15" font-weight="600" fill="{theme["text"]}">'
        "Cost per question vs execution accuracy</text>",
        f'<text x="{LEFT}" y="44" font-size="12" fill="{theme["muted"]}">'
        f"BIRD Mini-Dev test split, n={points[0].n} questions, 95% bootstrap CI</text>",
    ]
    # grid and axes
    y = y_lo
    while y <= y_hi + 1e-9:
        yy = sy(y)
        out.append(
            f'<line x1="{LEFT}" x2="{LEFT + pw}" y1="{yy:.1f}" y2="{yy:.1f}" '
            f'stroke="{theme["grid"]}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{LEFT - 8}" y="{yy + 4:.1f}" font-size="11" text-anchor="end" '
            f'fill="{theme["muted"]}">{y:.0%}</text>'
        )
        y += 0.05
    for t in _nice_log_ticks(x_lo, x_hi):
        xx = sx(t)
        out.append(
            f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{TOP}" y2="{TOP + ph}" '
            f'stroke="{theme["grid"]}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{xx:.1f}" y="{TOP + ph + 18}" font-size="11" text-anchor="middle" '
            f'fill="{theme["muted"]}">{_fmt_usd(t)}</text>'
        )
    out.append(
        f'<text x="{LEFT + pw / 2:.1f}" y="{H - 18}" font-size="12" text-anchor="middle" '
        f'fill="{theme["muted"]}">Mean cost per question (USD, log scale)</text>'
    )
    out.append(
        f'<text x="18" y="{TOP + ph / 2:.1f}" font-size="12" text-anchor="middle" '
        f'fill="{theme["muted"]}" transform="rotate(-90 18 {TOP + ph / 2:.1f})">'
        "Execution accuracy (EX)</text>"
    )
    # points: CI whisker, then marker with a surface ring, then a direct label
    for p in points:
        color = series[p.model_index]
        cx, cy = sx(p.cost), sy(p.ex)
        out.append(
            f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{sy(p.hi):.1f}" y2="{sy(p.lo):.1f}" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
        )
        for yv in (p.lo, p.hi):
            out.append(
                f'<line x1="{cx - 5:.1f}" x2="{cx + 5:.1f}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}" '
                f'stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
            )
        if p.system == "baseline":
            out.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{color}" '
                f'stroke="{theme["surface"]}" stroke-width="2"/>'
            )
        else:
            out.append(
                f'<rect x="{cx - 6:.1f}" y="{cy - 6:.1f}" width="12" height="12" rx="2" '
                f'fill="{color}" stroke="{theme["surface"]}" stroke-width="2"/>'
            )
        out.append(
            f'<text x="{cx + 12:.1f}" y="{cy + 4:.1f}" font-size="12" fill="{theme["text"]}" '
            f'stroke="{theme["surface"]}" stroke-width="4" stroke-linejoin="round" '
            f'paint-order="stroke">'
            f"{p.model.split()[0]} · {SYSTEM_LABELS[p.system]} {p.ex:.1%}</text>"
        )
    # legend, one row under the subtitle: colour = model, shape = system
    lx, ly = LEFT, 68
    for i, (model, _, _) in enumerate(SOURCES):
        out.append(f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" fill="{series[i]}"/>')
        out.append(
            f'<text x="{lx + 15}" y="{ly}" font-size="12" fill="{theme["text"]}">{model}</text>'
        )
        lx += 110
    lx += 20
    out.append(f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" fill="{theme["muted"]}"/>')
    out.append(
        f'<text x="{lx + 15}" y="{ly}" font-size="12" fill="{theme["text"]}">single shot</text>'
    )
    lx += 100
    out.append(
        f'<rect x="{lx}" y="{ly - 9}" width="10" height="10" rx="2" fill="{theme["muted"]}"/>'
    )
    out.append(
        f'<text x="{lx + 15}" y="{ly}" font-size="12" fill="{theme["text"]}">agent v2</text>'
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="docs/img/cost-vs-ex", help="path prefix for the SVGs")
    args = parser.parse_args()
    points = load_points()
    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        path = prefix.with_name(f"{prefix.name}-{name}.svg")
        path.write_text(render(points, theme), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
