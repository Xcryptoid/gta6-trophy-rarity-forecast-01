"""Forecast each GTA VI PS5 trophy's Exophase earn rate at a case's target time.

Model, in brief: Exophase only refreshes its counts hours apart, so the
house's frequent reads collapse into a few distinct snapshots. A trophy's
earn rate moves for two reasons -- new players joining the tracked
population (dilution, rates fall) and existing players earning the trophy
(rates rise) -- so the forecast is built from implied earner counts, not
from the rates directly. Both the tracked-players count and each trophy's
earner count are fitted in log-log space against time since release, read
at the target, and divided to give the median rate. The spread around the
median is honest about refresh jumps the history has not seen yet: residual
scatter, a random-walk term in the horizon, and a jump allowance that grows
with the predicted influx of new players, skewed downward because dilution
pushes rates down.

Standard library only. No network, no randomness, no wall clock: two runs
on the same case write byte-identical output.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The game's release, the origin of the log-log time axis.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# The five quantile levels every case asks for, in row order.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Spread constants, in logit units. Tuned so a flat launch-night history
# still gets a real band, and a multi-day horizon with a big predicted
# influx gets a wide one.
RESID_FLOOR = 0.20       # floor on the per-trophy fit scatter
WALK_SIGMA = 0.45        # random-walk sigma per square-root day
JUMP_FLOOR = 0.15        # refresh-jump allowance even with no growth
JUMP_PER_LOG_GROWTH = 0.35  # extra jump allowance per log unit of influx
JUMP_UP_FRACTION = 0.55  # upward jumps stretch less than downward ones
FALLBACK_DRIFT = 0.5     # no-fit trophies drift half-way with population

# Log-log slope clamps: counts cannot plausibly fall or explode faster.
SLOPE_MIN = -2.0
SLOPE_MAX = 4.0
PLAYERS_SLOPE_MIN = -1.0
PLAYERS_SLOPE_MAX = 3.0


def parse_time(text):
    """ISO-8601 timestamp (with a trailing Z) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    """Hours from release to a snapshot, never negative."""
    delta = (moment - RELEASE).total_seconds() / 3600.0
    return delta if delta > 0.0 else 0.0


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def logistic(x):
    if x >= 0.0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4_half_up(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def normal_quantile(level):
    """Acklam's rational approximation of the standard normal quantile."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if level < p_low:
        q = math.sqrt(-2.0 * math.log(level))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if level <= p_high:
        q = level - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - level))
    num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
    den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    return -num / den


def least_squares(xs, ys):
    """Ordinary least squares of ys on xs; None when the fit is undefined."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        sxx += dx * dx
        sxy += dx * (y - mean_y)
    if sxx <= 0.0:
        return None
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    return slope, intercept


def residual_spread(xs, ys, fit):
    """Standard deviation of the fit's residuals; 0 when it cannot be measured."""
    n = len(xs)
    if fit is None or n < 3:
        return 0.0
    slope, intercept = fit
    total = 0.0
    for x, y in zip(xs, ys):
        diff = y - (intercept + slope * x)
        total += diff * diff
    return math.sqrt(total / (n - 2))


def distinct_snapshots(history):
    """Collapse consecutive identical reads into one state each.

    Exophase refreshes hours apart, so the house's frequent reads repeat
    the same page many times. A snapshot is keyed by players-tracked and
    the full rate vector; its time is the earliest read of that state.
    """
    states = []
    previous_key = None
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates")
        if players is None or rates is None or players <= 0:
            continue
        key = (players, tuple(rates))
        if key == previous_key:
            continue
        previous_key = key
        states.append((parse_time(read["slot"]), players, list(rates)))
    return states


def forecast_count(xs, ys, x_target, last_value, slope_lo, slope_hi):
    """Read a log-log count fit at the target; None when no fit exists."""
    fit = least_squares(xs, ys)
    if fit is None:
        return None, 0.0
    slope, intercept = fit
    slope = min(slope_hi, max(slope_lo, slope))
    value = math.exp(intercept + slope * x_target)
    if not (value > 0.0) or math.isinf(value):
        return None, residual_spread(xs, ys, fit)
    return value, residual_spread(xs, ys, fit)


def forecast_case(case):
    levels = list(case.get("levels") or LEVELS)
    z_values = [normal_quantile(level) for level in levels]
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    target = parse_time(case["target"])
    x_target = math.log1p(hours_since_release(target))
    trophies = list(case.get("trophies") or [])
    snapshots = distinct_snapshots(case.get("history") or [])

    # --- tracked-players forecast -------------------------------------
    if snapshots:
        px = [math.log1p(hours_since_release(t)) for t, _, _ in snapshots]
        py = [math.log(players) for _, players, _ in snapshots]
        players_forecast, _ = forecast_count(
            px, py, x_target, snapshots[-1][1],
            PLAYERS_SLOPE_MIN, PLAYERS_SLOPE_MAX)
        if players_forecast is None:
            players_forecast = float(snapshots[-1][1])
        last_players = float(snapshots[-1][1])
        # A population forecast is bounded: it cannot collapse or explode
        # without limit over one horizon.
        players_forecast = min(max(players_forecast, last_players * 0.2),
                               last_players * 100.0)
        growth = players_forecast / last_players
    else:
        players_forecast = 0.0
        last_players = 0.0
        growth = 1.0

    # --- spread shared by every trophy --------------------------------
    walk = WALK_SIGMA * math.sqrt(horizon_days) if horizon_days > 0.0 else 0.0
    if growth >= 1.0:
        jump = JUMP_FLOOR + JUMP_PER_LOG_GROWTH * math.log(growth)
    else:
        jump = JUMP_FLOOR + 0.5 * JUMP_PER_LOG_GROWTH * abs(math.log(growth))

    rows = []
    for index, trophy in enumerate(trophies):
        # Earner counts at each distinct snapshot, skipping reads that
        # lacked this trophy and snapshots where nobody held it.
        exs = []
        eys = []
        last_earners = None
        last_rate = None
        for moment, players, rates in snapshots:
            rate = rates[index] if index < len(rates) else None
            if rate is None:
                continue
            last_rate = rate
            earners = rate / 100.0 * players
            if earners > 0.0:
                exs.append(math.log1p(hours_since_release(moment)))
                eys.append(math.log(earners))
                last_earners = earners

        if snapshots and last_rate is not None and last_earners is not None:
            earners_forecast, resid = forecast_count(
                exs, eys, x_target, last_earners, SLOPE_MIN, SLOPE_MAX)
            if earners_forecast is None:
                # No usable trend: hold the last earner count, drifting
                # half-way with the tracked population's move.
                earners_forecast = last_earners * (growth ** FALLBACK_DRIFT)
                resid = 0.0
            # Earners can never exceed the tracked population.
            earners_forecast = min(max(earners_forecast, 0.0), players_forecast)
            median = 100.0 * earners_forecast / players_forecast
        elif snapshots and last_rate is not None:
            # Zero earners in every read: the trophy stays near zero.
            median = 0.01
            resid = 0.0
        else:
            # No usable history at all for this trophy.
            median = 1.0
            resid = 1.0

        spread = max(resid, RESID_FLOOR)
        sigma_down = math.sqrt(spread * spread + walk * walk + jump * jump)
        sigma_up = math.sqrt(spread * spread + walk * walk
                             + (JUMP_UP_FRACTION * jump) ** 2)
        centre = logit(median)
        quantiles = []
        for z in z_values:
            sigma = sigma_down if z < 0.0 else sigma_up
            value = logistic(centre + z * sigma)
            quantiles.append(round4_half_up(min(100.0, max(0.0, value))))
        # Rounding can only move a value by half a unit in the last place,
        # so enforce the non-decreasing order explicitly.
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
