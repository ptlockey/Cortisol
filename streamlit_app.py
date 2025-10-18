"""Streamlit application for exploring cortisol time-course data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

TIME_POINTS: Tuple[int, ...] = (0, 15, 30, 45)
PHASE_PREFIXES: Dict[str, str] = {"Baseline": "B", "Post": "P"}
GROUP_LABELS: Dict[int, str] = {1: "MBLC", 2: "Control"}


@dataclass(frozen=True)
class SummaryRow:
    """Container that makes it easier to build a tidy summary table."""

    group: str
    phase: str
    time: int
    mean: float
    sd: float


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the Excel file shipped with the repository."""

    df = pd.read_excel("Cortisol.xlsx")
    df.columns = df.columns.str.strip()
    df["GroupLabel"] = df["Group"].map(GROUP_LABELS).fillna(df["Group"].astype(str))
    return df


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with cortisol concentrations log-transformed."""

    transformed = df.copy()
    value_columns = [
        f"{prefix}{time} nmol/l" for prefix in PHASE_PREFIXES.values() for time in TIME_POINTS
    ]
    for column in value_columns:
        if column in transformed:
            transformed[column] = np.log(transformed[column].clip(lower=0.01))
    return transformed


def to_numeric(values: Iterable) -> np.ndarray:
    """Convert an iterable of values to a float numpy array."""

    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def summarise_time_points(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and SD for each group/phase/time combination."""

    rows: List[SummaryRow] = []
    for group, gdf in df.groupby("GroupLabel"):
        for phase, prefix in PHASE_PREFIXES.items():
            for time_point in TIME_POINTS:
                column = f"{prefix}{time_point} nmol/l"
                if column not in gdf:
                    continue
                values = pd.to_numeric(gdf[column], errors="coerce")
                rows.append(
                    SummaryRow(
                        group=group,
                        phase=phase,
                        time=int(time_point),
                        mean=values.mean(),
                        sd=values.std(ddof=1),
                    )
                )
    return pd.DataFrame(rows)


def compute_auc(df: pd.DataFrame) -> pd.DataFrame:
    """Compute AUCg and AUCi for each participant and phase."""

    records: List[Dict[str, object]] = []
    total_duration = TIME_POINTS[-1] - TIME_POINTS[0]
    for _, row in df.iterrows():
        participant = row.get("ID", row.get("Participant", row.name))
        group = row["GroupLabel"]
        for phase, prefix in PHASE_PREFIXES.items():
            value_columns = [f"{prefix}{t} nmol/l" for t in TIME_POINTS]
            if not set(value_columns).issubset(row.index):
                continue
            values = to_numeric(row[value_columns])
            if np.isnan(values).any():
                continue
            auc_ground = np.trapz(values, dx=TIME_POINTS[1] - TIME_POINTS[0])
            auc_incremental = auc_ground - values[0] * total_duration
            records.append(
                {
                    "Participant": participant,
                    "Group": group,
                    "Phase": phase,
                    "Metric": "AUCg",
                    "Value": auc_ground,
                }
            )
            records.append(
                {
                    "Participant": participant,
                    "Group": group,
                    "Phase": phase,
                    "Metric": "AUCi",
                    "Value": auc_incremental,
                }
            )
    return pd.DataFrame.from_records(records)


def summarise_auc(auc_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return a summary table and a table of p-values for AUC metrics."""

    summary = (
        auc_df.groupby(["Metric", "Phase", "Group"], dropna=False)["Value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    p_values: List[Dict[str, object]] = []
    for (metric, phase), phase_df in auc_df.groupby(["Metric", "Phase"]):
        group_values = [gdf["Value"].dropna() for _, gdf in phase_df.groupby("Group")]
        if len(group_values) == 2 and all(len(vals) > 1 for vals in group_values):
            statistic, p_value = stats.ttest_ind(
                group_values[0], group_values[1], equal_var=False, nan_policy="omit"
            )
        else:
            statistic, p_value = np.nan, np.nan
        p_values.append(
            {
                "Metric": metric,
                "Phase": phase,
                "t-statistic": statistic,
                "p-value": p_value,
            }
        )
    return summary, pd.DataFrame(p_values)


def reshape_auc_summary(summary: pd.DataFrame, p_values: pd.DataFrame) -> pd.DataFrame:
    """Pivot the summary table to display each group alongside the p-value."""

    pivot = summary.pivot_table(
        index=["Metric", "Phase"],
        columns="Group",
        values=["mean", "std", "count"],
    )
    pivot.columns = [f"{stat} ({group})" for stat, group in pivot.columns]
    pivot = pivot.reset_index()
    table = pivot.merge(p_values, on=["Metric", "Phase"], how="left")
    return table.sort_values(["Metric", "Phase"]).reset_index(drop=True)


def make_line_plot(summary_df: pd.DataFrame, groups: List[str], phases: List[str]) -> go.Figure:
    """Create a line plot with mean ± SD whiskers for each selection."""

    fig = go.Figure()
    for group in groups:
        for phase in phases:
            subset = summary_df[(summary_df.group == group) & (summary_df.phase == phase)]
            if subset.empty:
                continue
            line_dash = "solid" if phase == "Post" else "dash"
            fig.add_trace(
                go.Scatter(
                    x=subset["time"],
                    y=subset["mean"],
                    error_y=dict(
                        type="data",
                        array=subset["sd"],
                        visible=True,
                    ),
                    mode="lines+markers",
                    name=f"{group} {phase}",
                    line=dict(dash=line_dash),
                )
            )
    fig.update_layout(
        xaxis_title="Time (minutes)",
        yaxis_title="Cortisol (nmol/L)",
        template="plotly_white",
        legend_title="Condition",
        hovermode="x unified",
    )
    return fig


def make_auc_timecourse_plot(
    summary_df: pd.DataFrame, groups: List[str], phases: List[str], metric: str
) -> go.Figure | None:
    """Create an area plot that visualises the AUC curves for the selections."""

    metric_key = metric.strip().upper()
    if metric_key not in {"AUCG", "AUCI"}:
        raise ValueError("metric must be either 'AUCg' or 'AUCi'")

    fig = go.Figure()
    has_trace = False
    for group in groups:
        for phase in phases:
            subset = summary_df[(summary_df.group == group) & (summary_df.phase == phase)]
            if subset.empty:
                continue
            y_values = subset["mean"].to_numpy()
            baseline = y_values[0] if len(y_values) else np.nan
            if metric_key == "AUCI":
                y_values = y_values - baseline
            fig.add_trace(
                go.Scatter(
                    x=subset["time"],
                    y=y_values,
                    mode="lines+markers",
                    fill="tozeroy",
                    name=f"{group} {phase}",
                )
            )
            has_trace = True

    if not has_trace:
        return None

    y_axis_title = (
        "Cortisol (nmol/L)"
        if metric_key == "AUCG"
        else "Cortisol above baseline (nmol/L)"
    )
    fig.update_layout(
        xaxis=dict(
            title="Time (minutes)",
            tickmode="array",
            tickvals=list(TIME_POINTS),
        ),
        yaxis=dict(title=y_axis_title, dtick=5),
        template="plotly_white",
        legend_title="Condition",
        hovermode="x unified",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Cortisol analysis dashboard", layout="wide")
    st.title("Cortisol time-course analysis")
    st.markdown(
        """
        Explore the cortisol measurements collected before (Baseline) and after (Post)
        the intervention. Use the controls in the sidebar to switch between the raw
        concentrations and a log-transformed view, and to focus on specific groups or phases.
        """
    )

    with st.sidebar:
        st.header("Controls")
        use_log = st.checkbox("Use log-transformed cortisol", value=False)

    raw_df = load_data()
    df = apply_log_transform(raw_df) if use_log else raw_df

    available_groups = sorted(df["GroupLabel"].dropna().unique().tolist())
    available_phases = list(PHASE_PREFIXES.keys())
    with st.sidebar:
        selected_groups = st.multiselect(
            "Groups", options=available_groups, default=available_groups
        )
        selected_phases = st.multiselect(
            "Phases", options=available_phases, default=available_phases
        )

    st.subheader("Raw data preview")
    st.dataframe(df.head())

    st.subheader("Means and standard deviations by time point")
    summary_df = summarise_time_points(df)
    st.dataframe(summary_df)

    st.subheader("Cortisol trajectory")
    if selected_groups and selected_phases:
        figure = make_line_plot(summary_df, selected_groups, selected_phases)
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("Select at least one group and phase to display the line plot.")

    st.subheader("Area under the curve metrics")
    auc_df = compute_auc(df)
    if auc_df.empty:
        st.warning("Not enough complete cases to compute AUC metrics.")
        return

    st.markdown("### AUC visualisations")
    if selected_groups and selected_phases:
        tabs = st.tabs(["AUCg", "AUCi"])
        for metric, tab in zip(["AUCg", "AUCi"], tabs):
            with tab:
                auc_figure = make_auc_timecourse_plot(summary_df, selected_groups, selected_phases, metric)
                if auc_figure is None:
                    st.info("No data available for the selected combination.")
                else:
                    st.plotly_chart(auc_figure, use_container_width=True)
    else:
        st.info("Select at least one group and phase to display the AUC plots.")

    summary_auc, p_values = summarise_auc(auc_df)
    table = reshape_auc_summary(summary_auc, p_values)
    st.markdown("### Group summaries")
    st.dataframe(table)

    st.markdown("### Individual participant AUCs")
    st.dataframe(auc_df)

    st.caption(
        "AUCg: area under the curve with respect to ground. "
        "AUCi: area under the curve with respect to increase."
    )


if __name__ == "__main__":
    main()
