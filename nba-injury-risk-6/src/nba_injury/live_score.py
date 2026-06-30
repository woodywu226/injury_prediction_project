"""Stage 10 — live monitoring job (scheduled scoring of current games).

Runs weekly (via GitHub Actions) to:
  1. pull the current week's player game logs (live data),
  2. build the same weekly features the training pipeline uses,
  3. score them with the frozen trained model,
  4. append the scores + a drift snapshot to reports/live/ so the monitoring
     layer has a rolling record.

DESIGN: training uses the historical pipeline; this job is INFERENCE +
MONITORING only. It never retrains. It writes a small JSON per run so the drift
history accumulates over time in the repo (free, no database).

Because live data sources (stats.nba.com) aren't reachable from every
environment, the job degrades gracefully: if the pull fails, it records the
failure in the run log rather than crashing the workflow.

Run:
    python -m nba_injury.live_score                 # real (needs network)
    python -m nba_injury.live_score --demo          # synthetic current-week demo
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nba_injury.cache import processed_path, _PKG_ROOT
from nba_injury.model_hazard import (
    load_table, temporal_split, TIER1_FEATURES, TIER2_FEATURES,
)
from nba_injury.prescriptive import _fit_model

LIVE_DIR = _PKG_ROOT / "reports" / "live"


def _trained_model():
    df = load_table()
    features = TIER1_FEATURES + TIER2_FEATURES
    train, _, _, _ = temporal_split(df)
    clf = _fit_model(train, features)
    return clf, features, df


def _fetch_current_week_features() -> pd.DataFrame:
    """Pull the current week's player game logs and aggregate to weekly features.

    Uses nba_api LeagueGameLog for the current season; aggregates per player over
    the last 7 days. Returns a frame with the TIER1 feature columns (Tier-2 left
    NaN — the model handles it). Raises on network failure (caught by caller).
    """
    from nba_api.stats.endpoints import leaguegamelog

    season = _current_season_str()
    ep = leaguegamelog.LeagueGameLog(season=season, timeout=45)
    rows = ep.get_normalized_dict().get("LeagueGameLog", [])
    if not rows:
        raise RuntimeError("no current-season game rows returned")
    g = pd.DataFrame(rows)
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"]).dt.date
    cutoff = date.today()
    last7 = g[g["GAME_DATE"] >= (cutoff - pd.Timedelta(days=7).to_pytimedelta())]
    feats = []
    for pid, pg in last7.groupby("PLAYER_ID"):
        feats.append({
            "player_id": int(pid),
            "games_this_week": len(pg),
            "minutes_this_week": float(pg["MIN"].astype(float).sum()),
            "back_to_backs_this_week": _count_b2b(sorted(pg["GAME_DATE"])),
            "games_in_7days": len(pg),
            "cum_season_minutes": float(g[g.PLAYER_ID == pid]["MIN"].astype(float).sum()),
            "cum_season_games": int((g.PLAYER_ID == pid).sum()),
            "usg_pct": np.nan, "pace": np.nan, "prior_injury_count": 0,
        })
    return pd.DataFrame(feats)


def _count_b2b(days) -> int:
    return sum(1 for i in range(1, len(days)) if (days[i] - days[i - 1]).days == 1)


def _current_season_str() -> str:
    t = date.today()
    start = t.year if t.month >= 10 else t.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def _demo_current_week(df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic 'current week' = sample recent at-risk rows, so the job runs
    end-to-end without network. Clearly not real."""
    at_risk = df[df["at_risk"] == 1]
    sample = at_risk.sample(min(40, len(at_risk)), random_state=int(datetime.now().timestamp()) % 1000)
    return sample.copy()


def run(demo: bool = False) -> dict:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    clf, features, df = _trained_model()
    stamp = datetime.now().strftime("%Y-%m-%d")

    status = "ok"
    detail = ""
    try:
        if demo:
            cur = _demo_current_week(df)
            source = "synthetic-demo"
        else:
            cur = _fetch_current_week_features()
            source = "nba_api-live"
            # align columns; Tier-2 absent live -> NaN (model handles it)
            for f in features:
                if f not in cur:
                    cur[f] = np.nan
    except Exception as e:  # noqa: BLE001 — never crash the scheduled workflow
        status = "fetch_failed"
        detail = f"{type(e).__name__}: {e}"
        record = {"date": stamp, "status": status, "detail": detail,
                  "note": "live source unreachable; no scores this run"}
        _append_log(record)
        print(f"[live] {status}: {detail}", file=sys.stderr)
        return record

    X = cur[features].to_numpy(float)
    cur["pred_hazard"] = clf.predict_proba(X)[:, 1]
    top = cur.sort_values("pred_hazard", ascending=False).head(15)

    record = {
        "date": stamp,
        "status": status,
        "source": source,
        "n_scored": int(len(cur)),
        "mean_hazard": float(cur["pred_hazard"].mean()),
        "p90_hazard": float(cur["pred_hazard"].quantile(0.9)),
        "top_players": [
            {"player_id": int(r.player_id), "hazard": round(float(r.pred_hazard), 4)}
            for r in top.itertuples(index=False)
        ],
    }
    _append_log(record)
    # also write the latest snapshot for the dashboard / README badge
    (LIVE_DIR / "latest.json").write_text(json.dumps(record, indent=2))
    print(f"[live] {source}: scored {record['n_scored']} players, "
          f"mean hazard {record['mean_hazard']:.4f}")
    return record


def _append_log(record: dict) -> None:
    log = LIVE_DIR / "history.jsonl"
    with log.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 10 live scoring job.")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic current-week run (no network)")
    args = ap.parse_args()
    run(demo=args.demo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
