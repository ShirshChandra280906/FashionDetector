"""Power and sample size for the Shop-the-Look A/B test.

All inputs are PLANNING ASSUMPTIONS, not measured data. They are stated in
docs/AB_TEST.md and can be overridden on the command line.

Primary metric: user-level add-to-cart (ATC) conversion = share of randomized
users with >= 1 add-to-cart during the test window. It is a two-proportion test.

Dilution: only users who *adopt* the new entry point can be affected. If a
share `adoption` of treatment users use it, and it lifts their ATC rate by
`adopter_lift` (relative), the whole-arm lift is about adoption * adopter_lift.
This is why we also plan a triggered analysis (exposed users only).

Usage:
    python analysis/power_analysis.py
    python analysis/power_analysis.py --baseline 0.08 --mde 0.03 --daily-users 100000
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

OUT = Path(__file__).resolve().parents[1] / "results" / "ab_planning"


def n_per_arm(p0: float, rel_mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Closed-form sample size per arm for a two-sided two-proportion z-test
    (unpooled variance). Detects p1 = p0 * (1 + rel_mde)."""
    p1 = p0 * (1 + rel_mde)
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return math.ceil((z_a + z_b) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2)


def n_per_arm_statsmodels(p0: float, rel_mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Cross-check with statsmodels (Cohen's h, arcsine transform)."""
    h = proportion_effectsize(p0 * (1 + rel_mde), p0)
    return math.ceil(NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power, ratio=1.0))


def mde_for_n(p0: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest relative lift detectable with n users per arm (bisection)."""
    lo, hi = 1e-4, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if n_per_arm(p0, mid, alpha, power) <= n else (mid, hi)
    return hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=float, default=0.08, help="control user-level ATC rate (assumption)")
    ap.add_argument("--mde", type=float, default=0.03, help="relative minimum detectable effect on the whole arm")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--daily-users", type=int, default=100_000, help="NEW eligible users entering the test per day (both arms)")
    ap.add_argument("--min-days", type=int, default=14, help="floor: two full weekly cycles")
    ap.add_argument("--adoption", type=float, default=0.12, help="share of treatment users who use the entry point")
    args = ap.parse_args()

    n = n_per_arm(args.baseline, args.mde, args.alpha, args.power)
    n_sm = n_per_arm_statsmodels(args.baseline, args.mde, args.alpha, args.power)
    days_needed = math.ceil(2 * n / args.daily_users)
    duration = max(days_needed, args.min_days)
    plan = {
        "assumptions": vars(args),
        "n_per_arm": n,
        "n_per_arm_statsmodels_check": n_sm,
        "total_users": 2 * n,
        "days_to_reach_sample": days_needed,
        "planned_duration_days": duration,
        "absolute_mde_pp": round(args.baseline * args.mde * 100, 4),
        "implied_adopter_lift_needed": args.mde / args.adoption,
        "mde_achieved_at_planned_duration": mde_for_n(args.baseline, duration * args.daily_users // 2, args.alpha, args.power),
    }

    # Sensitivity table: users per arm for a grid of baselines x MDEs
    grid = []
    for p0 in (0.04, 0.06, 0.08, 0.10, 0.12):
        for mde in (0.02, 0.03, 0.05, 0.10):
            m = n_per_arm(p0, mde, args.alpha, args.power)
            grid.append({"baseline": p0, "relative_mde": mde, "n_per_arm": m,
                         "days_at_daily_users": math.ceil(2 * m / args.daily_users)})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "power_plan.json").write_text(json.dumps(plan, indent=2))
    import pandas as pd

    pd.DataFrame(grid).to_csv(OUT / "sample_size_grid.csv", index=False)

    print("PLANNING ASSUMPTIONS (not measured data)")
    print(json.dumps(plan, indent=2))
    print("\nUsers per arm (alpha=%.2f, power=%.2f):" % (args.alpha, args.power))
    print(pd.DataFrame(grid).pivot(index="baseline", columns="relative_mde", values="n_per_arm").to_string())


if __name__ == "__main__":
    main()
