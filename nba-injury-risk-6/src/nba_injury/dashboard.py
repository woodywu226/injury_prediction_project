"""Stage 10 — lightweight Streamlit dashboard (Phase-2 frontend).

Player -> risk curve -> prescriptive levers -> monitoring, reading the trained
model and the frozen person-period table. This is a THIN presentation layer over
the Stage 4-8 logic — no modeling lives here; it calls the same functions the
gates do, so the dashboard can never disagree with the analysis.

Run (on your machine, after Stage 3 has produced person_period.parquet):
    pip install streamlit
    streamlit run src/nba_injury/dashboard.py

If the parquet is missing, the app offers to build a synthetic world so it's
demonstrable without network access (clearly flagged NOT real data).
"""
from __future__ import annotations

import sys
from pathlib import Path

# make the package importable when run via `streamlit run src/nba_injury/dashboard.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from nba_injury.cache import processed_path
from nba_injury.model_hazard import (
    load_table, temporal_split, TIER1_FEATURES, TIER2_FEATURES,
)
from nba_injury.prescriptive import (
    _fit_model, player_levers, shap_attributions, STANDING_CAVEAT,
    MODIFIABLE_FEATURES,
)
from nba_injury.monitoring import run as run_monitoring


st.set_page_config(page_title="NBA Injury-Risk Decision Support", layout="wide")


@st.cache_data(show_spinner=False)
def _load():
    df = load_table()
    return df


@st.cache_resource(show_spinner=False)
def _model(_df):
    features = TIER1_FEATURES + TIER2_FEATURES
    train, _, _, _ = temporal_split(_df)
    clf = _fit_model(train, features)
    return clf, features


def _ensure_data() -> bool:
    if processed_path("person_period.parquet").exists():
        return True
    st.warning("No person-period table found. This dashboard needs Stage 3 output.")
    if st.button("Build a SYNTHETIC world (NOT real data) so I can explore the UI"):
        from nba_injury.make_synthetic_world import main as mk
        from nba_injury.build_person_period import main as build
        with st.spinner("Building synthetic world + person-period table..."):
            mk()
            sys.argv = ["build_person_period"]
            build()
        st.success("Synthetic table built. Rerun.")
        st.rerun()
    return False


def main():
    st.title("🏀 NBA Injury-Risk & Career-Longevity Decision Support")
    st.caption("Decision support for performance/sports-science staff — "
               "NOT autonomous medical advice. Outputs are model-implied "
               "hypotheses for expert review.")

    if not _ensure_data():
        st.stop()

    df = _load()
    clf, features = _model(df)
    at_risk = df[df["at_risk"] == 1].copy()
    at_risk["_pred"] = clf.predict_proba(at_risk[features].to_numpy(float))[:, 1]

    tab_player, tab_levers, tab_monitor, tab_about = st.tabs(
        ["Player risk", "Prescriptive levers", "Monitoring", "About / limits"])

    # ---- Player risk curve ----
    with tab_player:
        players = sorted(at_risk["player_id"].unique())
        pid = st.selectbox("Player", players,
                           format_func=lambda p: f"player {p}")
        g = at_risk[at_risk["player_id"] == pid].sort_values("week_start")
        st.subheader(f"Weekly modeled hazard — player {pid}")
        chart_df = g[["week_start", "_pred"]].rename(
            columns={"week_start": "week", "_pred": "modeled hazard"}
        ).set_index("week")
        st.line_chart(chart_df)
        c1, c2, c3 = st.columns(3)
        c1.metric("peak modeled hazard", f"{g['_pred'].max():.3f}")
        c2.metric("at-risk weeks", len(g))
        c3.metric("injury events on record", int(g["event"].sum()))
        st.info("The hazard is the model's estimate of a time-loss injury "
                "beginning that week. It ranks weeks by risk; it is not a "
                "diagnosis.")

    # ---- Prescriptive levers ----
    with tab_levers:
        st.subheader("What the MODEL attributes risk to")
        attr = shap_attributions(clf, df[df.at_risk == 1][features].to_numpy(float),
                                 features)
        attr_df = pd.DataFrame(attr["ranking"], columns=["feature", "mean |impact|"])
        attr_df["type"] = attr_df["feature"].map(
            lambda f: "actionable" if f in MODIFIABLE_FEATURES else "fixed/context")
        st.dataframe(attr_df, use_container_width=True, hide_index=True)
        st.caption(f"Attribution method: {attr['method']}. This explains the "
                   "MODEL, not real-world causation.")

        st.subheader("Model-implied modifiable levers (a hypothesis, not advice)")
        pl = player_levers(df, clf, features, pid)
        if pl.get("top_modifiable_levers"):
            lev_df = pd.DataFrame(pl["top_modifiable_levers"])[
                ["lever", "from_value", "to_value", "hazard_delta"]]
            st.dataframe(lev_df, use_container_width=True, hide_index=True)
        else:
            st.write("No modifiable levers surfaced for this player-week.")
        st.warning(STANDING_CAVEAT)

    # ---- Monitoring ----
    with tab_monitor:
        st.subheader("Model-assurance / monitoring")
        st.caption("Distinguishes harmless input drift from real degradation.")
        if st.button("Run monitoring pass"):
            with st.spinner("Computing drift, calibration, late-label, demo..."):
                mon = run_monitoring(use_evidently=False)
            demo = mon.get("verifiable_demo", {})
            st.markdown("**Verifiable-event demo (COVID seasons):**")
            st.write(demo.get("verdict", "n/a"))
            cal = pd.DataFrame(mon.get("calibration_drift", []))
            if not cal.empty:
                st.markdown("**Calibration drift over time:**")
                st.line_chart(cal.set_index("window")[["brier", "trivial_brier"]])
            ll = pd.DataFrame(mon.get("late_label", []))
            if not ll.empty:
                st.markdown("**Late-label: estimated (pre) vs actual (post) Brier:**")
                st.dataframe(ll, use_container_width=True, hide_index=True)

    # ---- About ----
    with tab_about:
        st.markdown(
            "### What this is\n"
            "A public-data NBA injury-risk model (discrete-time hazard on a "
            "person-period dataset) with a prescriptive lever layer and a "
            "monitoring layer.\n\n"
            "### What it can and cannot say\n"
            "- **Can:** rank at-risk player-weeks by modeled hazard; surface "
            "which in-game load features the model leans on; flag drift vs "
            "degradation over time.\n"
            "- **Cannot:** claim causation, advise on training/recovery (not in "
            "public data), or replace clinical judgment.\n\n"
            "### Honest limitation\n"
            "Public data captures workload and availability, not the imaging, "
            "biomechanics, treatment history, or practice exposure a team holds. "
            "The label quality caps everything.\n")
        st.warning(STANDING_CAVEAT)


if __name__ == "__main__":
    main()
else:
    main()
