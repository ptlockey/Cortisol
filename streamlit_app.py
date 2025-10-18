"""Streamlit application for exploring cortisol time-course data."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import colors as plotly_colors
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
    if "Ppt ID" in df.columns:
        df["Cohort"] = df["Ppt ID"].astype(str).str.extract(r"^(C\d)")
    return df


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with cortisol concentrations log-transformed."""

    transformed = df.copy()
    value_columns = [
        f"{prefix}{time} nmol/l" for prefix in PHASE_PREFIXES.values() for time in TIME_POINTS
    ]
    for column in value_columns:
        if column in transformed:
            numeric = pd.to_numeric(transformed[column], errors="coerce")
            with np.errstate(divide="ignore"):
                log_values = np.where(numeric > 0, np.log(numeric), np.nan)
            transformed[column] = pd.Series(log_values, index=transformed.index)
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


def make_line_plot(
    summary_df: pd.DataFrame, groups: List[str], phases: List[str], use_log_scale: bool
) -> go.Figure:
    """Create a line plot with mean ± SD whiskers for each selection."""

    fig = go.Figure()
    max_whisker = None
    for group in groups:
        for phase in phases:
            subset = summary_df[(summary_df.group == group) & (summary_df.phase == phase)]
            if subset.empty:
                continue
            if not use_log_scale and "mean" in subset and "sd" in subset:
                whisker = (subset["mean"] + subset["sd"]).max()
                if pd.notna(whisker):
                    max_whisker = max(whisker, max_whisker or whisker)
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

    y_axis_title = "Log cortisol (ln(nmol/L))" if use_log_scale else "Cortisol (nmol/L)"
    yaxis_config = _build_cortisol_axis_config(
        max_whisker, use_log_scale=use_log_scale, axis_title=y_axis_title
    )

    fig.update_layout(
        xaxis=dict(
            title="Time (minutes)",
            tickmode="array",
            tickvals=list(TIME_POINTS),
        ),
        yaxis=yaxis_config,
        template="plotly_white",
        legend_title="Condition",
        hovermode="x unified",
    )
    return fig


def _build_cortisol_axis_config(
    max_value: float | None,
    *,
    use_log_scale: bool = False,
    axis_title: str | None = None,
) -> Dict[str, object]:
    """Return a consistent axis configuration for cortisol visualisations."""

    title = axis_title or ("Log cortisol (ln(nmol/L))" if use_log_scale else "Cortisol (nmol/L)")
    config: Dict[str, object] = {"title": title}
    if use_log_scale:
        return config

    config.update({"rangemode": "tozero", "tick0": 0, "dtick": 5})
    if max_value is not None and np.isfinite(max_value):
        padding = max(0.5, max_value * 0.05)
        config["range"] = [0, max_value + padding]
    return config


def make_baseline_distribution_plot(
    df: pd.DataFrame, groups: List[str], phases: List[str]
) -> go.Figure | None:
    """Display individual baseline measurements for the selected groups/phases."""

    phase_columns = {"Baseline": "B0 nmol/l", "Post": "P0 nmol/l"}
    available_phases = [phase for phase in phases if phase in phase_columns]
    fig = go.Figure()
    has_trace = False
    max_value = None

    for phase in available_phases:
        column = phase_columns[phase]
        if column not in df:
            continue
        for group in groups:
            subset = df[df["GroupLabel"] == group]
            if subset.empty:
                continue
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            if values.empty:
                continue
            candidate = values.max()
            if pd.notna(candidate):
                max_value = max(candidate, max_value or candidate)
            fig.add_trace(
                go.Box(
                    y=values,
                    name=f"{group} {phase}",
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=0,
                    marker=dict(size=6),
                )
            )
            has_trace = True

    if not has_trace:
        return None

    fig.update_layout(
        xaxis_title="Condition",
        yaxis=_build_cortisol_axis_config(max_value),
        template="plotly_white",
        hovermode="closest",
        showlegend=False,
    )
    return fig


def make_timepoint_distribution_plot(
    df: pd.DataFrame,
    groups: List[str],
    phase: str,
    *,
    use_log_scale: bool,
) -> go.Figure | None:
    """Show individual measurements for every time point within a phase."""

    prefix = PHASE_PREFIXES.get(phase)
    if prefix is None:
        return None

    fig = go.Figure()
    has_trace = False
    max_value = None

    for group in groups:
        subset = df[df["GroupLabel"] == group]
        if subset.empty:
            continue
        for time_point in TIME_POINTS:
            column = f"{prefix}{time_point} nmol/l"
            if column not in subset:
                continue
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            if values.empty:
                continue
            candidate = values.max()
            if pd.notna(candidate):
                max_value = max(candidate, max_value or candidate)
            fig.add_trace(
                go.Box(
                    y=values,
                    name=f"{group} {time_point} min",
                    boxpoints="all",
                    boxmean=True,
                    jitter=0.4,
                    pointpos=0,
                    marker=dict(size=6),
                )
            )
            has_trace = True

    if not has_trace:
        return None

    axis_title = "Log cortisol (ln(nmol/L))" if use_log_scale else "Cortisol (nmol/L)"
    fig.update_layout(
        title=f"{phase} Individual Values in Relation to Mean of Group",
        xaxis_title="Group and time point",
        yaxis=_build_cortisol_axis_config(
            max_value, use_log_scale=use_log_scale, axis_title=axis_title
        ),
        template="plotly_white",
        showlegend=False,
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
    max_value = None
    for group in groups:
        for phase in phases:
            subset = summary_df[(summary_df.group == group) & (summary_df.phase == phase)]
            if subset.empty:
                continue
            y_values = subset["mean"].to_numpy()
            if len(y_values):
                candidate = np.nanmax(y_values)
                if np.isfinite(candidate):
                    max_value = candidate if max_value is None else max(max_value, candidate)
            fig.add_trace(
                go.Scatter(
                    x=subset["time"],
                    y=y_values,
                    mode="lines+markers",
                    fill="tozeroy" if metric_key == "AUCG" else None,
                    name=f"{group} {phase}",
                )
            )
            has_trace = True

    if not has_trace:
        return None

    y_axis_title = (
        "Cortisol (nmol/L)"
        if metric_key == "AUCG"
        else "Cortisol change from baseline (nmol/L)"
    )
    if metric_key == "AUCG":
        yaxis_config = _build_cortisol_axis_config(max_value, axis_title=y_axis_title)
    else:
        yaxis_config = dict(
            title=y_axis_title,
            zeroline=True,
            zerolinecolor="#999999",
            zerolinewidth=1,
        )

    fig.update_layout(
        xaxis=dict(
            title="Time (minutes)",
            tickmode="array",
            tickvals=list(TIME_POINTS),
        ),
        yaxis=yaxis_config,
        template="plotly_white",
        legend_title="Condition",
        hovermode="x unified",
    )
    return fig


def _extract_phase_timecourse(row: pd.Series, phase: str) -> Tuple[List[int], List[float]]:
    """Return the time points and cortisol values for a phase from a wide row."""

    prefix = PHASE_PREFIXES.get(phase)
    if prefix is None:
        return [], []

    times: List[int] = []
    values: List[float] = []
    for time_point in TIME_POINTS:
        column = f"{prefix}{time_point} nmol/l"
        if column not in row:
            continue
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        times.append(int(time_point))
        values.append(float(value))
    return times, values


def make_participant_timecourse_plot(
    row: pd.Series, phases: Iterable[str], y_axis_title: str, force_zero_start: bool
) -> go.Figure | None:
    """Return a line plot of cortisol values for a single participant."""

    fig = go.Figure()
    has_trace = False

    for phase in phases:
        times, values = _extract_phase_timecourse(row, phase)
        if not times:
            continue
        fig.add_trace(
            go.Scatter(x=times, y=values, mode="lines+markers", name=phase)
        )
        has_trace = True

    if not has_trace:
        return None

    max_value = None
    for trace in fig.data:
        y_data = trace["y"]
        if y_data is None:
            continue
        candidate = np.nanmax(y_data)
        if np.isfinite(candidate):
            max_value = candidate if max_value is None else max(max_value, candidate)

    yaxis_config = _build_cortisol_axis_config(
        max_value, use_log_scale=not force_zero_start, axis_title=y_axis_title
    )

    fig.update_layout(
        xaxis=dict(
            title="Time (minutes)",
            tickmode="array",
            tickvals=list(TIME_POINTS),
        ),
        yaxis=yaxis_config,
        template="plotly_white",
        legend_title="Phase",
        hovermode="x unified",
    )
    return fig


def make_group_spaghetti_plot(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    participant_column: str,
    groups: List[str],
    phases: List[str],
) -> go.Figure | None:
    """Show every participant trajectory for the selected groups and phases."""

    if participant_column not in df:
        return None

    filtered = df[df["GroupLabel"].isin(groups)] if groups else df
    if filtered.empty:
        return None

    colour_cycle = cycle(plotly_colors.qualitative.Plotly)
    colour_map: Dict[Tuple[str, str], str] = {}
    for group in groups:
        for phase in phases:
            colour_map[(group, phase)] = next(colour_cycle)

    fig = go.Figure()
    has_trace = False
    max_value = None

    for (group, phase), colour in colour_map.items():
        phase_subset = filtered[filtered["GroupLabel"] == group]
        if phase_subset.empty:
            continue

        for _, row in phase_subset.iterrows():
            times, values = _extract_phase_timecourse(row, phase)
            if not times:
                continue
            participant_label = row.get(participant_column, "Unknown")
            if values:
                candidate = max(values)
                if max_value is None or candidate > max_value:
                    max_value = candidate
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=values,
                    mode="lines",
                    line=dict(color=colour, width=1),
                    opacity=0.35,
                    legendgroup=f"{group}-{phase}",
                    showlegend=False,
                    hovertemplate=(
                        "Group: "
                        + group
                        + "<br>Phase: "
                        + phase
                        + "<br>Participant: %{customdata[0]}"
                        + "<br>Time: %{x} min<br>Cortisol: %{y:.2f} nmol/L"
                        + "<extra></extra>"
                    ),
                    customdata=np.array([[participant_label]] * len(times)),
                )
            )
            has_trace = True

        summary_subset = summary_df[
            (summary_df.group == group) & (summary_df.phase == phase)
        ]
        if not summary_subset.empty:
            if {"mean", "sd"}.issubset(summary_subset.columns):
                whisker = (summary_subset["mean"] + summary_subset["sd"]).max()
                if pd.notna(whisker):
                    max_value = max(whisker, max_value or whisker)
            fig.add_trace(
                go.Scatter(
                    x=summary_subset["time"],
                    y=summary_subset["mean"],
                    mode="lines+markers",
                    name=f"{group} {phase} mean",
                    line=dict(color=colour, width=3),
                    legendgroup=f"{group}-{phase}",
                    showlegend=True,
                    hovertemplate=(
                        "Group: "
                        + group
                        + "<br>Phase: "
                        + phase
                        + "<br>Time: %{x} min"
                        + "<br>Mean cortisol: %{y:.2f} nmol/L"
                        + "<extra></extra>"
                    ),
                )
            )
            has_trace = True

    if not has_trace:
        return None

    fig.update_layout(
        xaxis=dict(
            title="Time (minutes)",
            tickmode="array",
            tickvals=list(TIME_POINTS),
        ),
        yaxis=_build_cortisol_axis_config(max_value),
        template="plotly_white",
        legend_title="Condition",
        hovermode="x unified",
    )
    return fig


def make_individual_auc_plot(
    auc_df: pd.DataFrame, metric: str, groups: List[str], phases: List[str]
) -> go.Figure | None:
    """Display participant-level AUC values for the requested metric."""

    metric_key = metric.strip().upper()
    subset = auc_df[auc_df["Metric"].str.upper() == metric_key]
    if groups:
        subset = subset[subset["Group"].isin(groups)]
    if phases:
        subset = subset[subset["Phase"].isin(phases)]

    if subset.empty:
        return None

    fig = go.Figure()
    max_value = None
    min_value = None
    colour_cycle = cycle(plotly_colors.qualitative.Set2)
    group_colours = {group: next(colour_cycle) for group in groups} if groups else {}

    phase_order = {phase: index for index, phase in enumerate(sorted(phases))}
    category_values = [
        f"{group} {phase}"
        for phase in phases
        for group in groups
        if ((subset["Group"] == group) & (subset["Phase"] == phase)).any()
    ]

    for participant, participant_df in subset.groupby("Participant"):
        participant_df = participant_df.sort_values(
            "Phase", key=lambda col: col.map(phase_order).fillna(len(phase_order))
        )
        x_values = [f"{row['Group']} {row['Phase']}" for _, row in participant_df.iterrows()]
        y_values = participant_df["Value"].to_numpy()
        if y_values.size:
            candidate_max = np.nanmax(y_values)
            candidate_min = np.nanmin(y_values)
            if np.isfinite(candidate_max):
                max_value = candidate_max if max_value is None else max(max_value, candidate_max)
            if np.isfinite(candidate_min):
                min_value = candidate_min if min_value is None else min(min_value, candidate_min)
        group_label = participant_df["Group"].iloc[0] if not participant_df.empty else ""
        colour = group_colours.get(group_label, next(colour_cycle))
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=f"{participant} ({group_label})",
                marker=dict(color=colour),
                line=dict(color=colour),
                hovertemplate=(
                    "Participant: "
                    + str(participant)
                    + "<br>Group: "
                    + str(group_label)
                    + "<br>Phase: %{x}"
                    + "<br>AUC: %{y:.2f}"
                    + "<extra></extra>"
                ),
            )
        )

    yaxis_title = "AUC (nmol/L × min)" if metric_key == "AUCG" else "AUC increase (nmol/L × min)"
    yaxis_config = dict(title=yaxis_title)
    if metric_key == "AUCG":
        yaxis_config["rangemode"] = "tozero"
        if max_value is not None and np.isfinite(max_value):
            padding = max(5.0, max_value * 0.05)
            yaxis_config["range"] = [0, max_value + padding]
    else:
        if min_value is not None and max_value is not None and np.isfinite(min_value) and np.isfinite(max_value):
            padding = max(5.0, (max_value - min_value) * 0.05)
            yaxis_config["range"] = [min_value - padding, max_value + padding]
        yaxis_config.update(
            zeroline=True,
            zerolinecolor="#999999",
            zerolinewidth=1,
        )

    fig.update_layout(
        template="plotly_white",
        legend_title="Participant",
        xaxis=dict(
            title="Group and phase",
            categoryorder="array",
            categoryarray=category_values,
        ),
        yaxis=yaxis_config,
    )
    return fig


def make_phase_overlay_plot(
    df: pd.DataFrame,
    participant_column: str,
    groups: List[str],
    phase: str,
    use_log_scale: bool,
) -> go.Figure | None:
    """Plot each participant's cortisol trajectory for a single phase."""

    if participant_column not in df:
        return None
    prefix = PHASE_PREFIXES.get(phase)
    if prefix is None:
        return None

    filtered = df.dropna(subset=[participant_column])
    if groups:
        filtered = filtered[filtered["GroupLabel"].isin(groups)]
    if filtered.empty:
        return None

    colour_cycle = cycle(plotly_colors.qualitative.Dark24)
    fig = go.Figure()
    has_trace = False
    max_value = None

    for _, row in filtered.iterrows():
        times, values = _extract_phase_timecourse(row, phase)
        if not times:
            continue
        participant = row.get(participant_column, "Unknown")
        group = row.get("GroupLabel", "Unknown")
        colour = next(colour_cycle)
        if values:
            candidate = max(values)
            if max_value is None or candidate > max_value:
                max_value = candidate
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines+markers",
                line=dict(color=colour, width=2),
                name=str(participant),
                hovertemplate=(
                    "Group: "
                    + str(group)
                    + "<br>Participant: "
                    + str(participant)
                    + "<br>Time: %{x} min"
                    + "<br>Cortisol: %{y:.2f}"
                    + (" ln(nmol/L)" if use_log_scale else " nmol/L")
                    + "<extra></extra>"
                ),
            )
        )
        has_trace = True

    if not has_trace:
        return None

    y_axis_title = "Log cortisol (ln(nmol/L))" if use_log_scale else "Cortisol (nmol/L)"
    yaxis_config = _build_cortisol_axis_config(max_value, use_log_scale=use_log_scale, axis_title=y_axis_title)

    fig.update_layout(
        title=f"{phase} cortisol trajectories",
        xaxis=dict(
            title="Time (minutes)",
            tickmode="array",
            tickvals=list(TIME_POINTS),
        ),
        yaxis=yaxis_config,
        template="plotly_white",
        hovermode="x unified",
        legend_title="Participant",
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
    participant_column = next(
        (column for column in ("Participant", "ID", "Ppt ID") if column in df.columns),
        None,
    )
    participant_labels: Dict[object, str] = {}
    with st.sidebar:
        selected_groups = st.multiselect(
            "Groups", options=available_groups, default=available_groups
        )
        selected_phases = st.multiselect(
            "Phases", options=available_phases, default=available_phases
        )
        available_cohorts = (
            sorted(df["Cohort"].dropna().unique().tolist()) if "Cohort" in df.columns else []
        )
        selected_cohorts = (
            st.multiselect("Cohorts", options=available_cohorts, default=available_cohorts)
            if available_cohorts
            else []
        )
        if participant_column:
            participant_df = df[[participant_column, "GroupLabel", "Cohort"]].dropna(
                subset=[participant_column]
            )
            if selected_groups:
                participant_df = participant_df[
                    participant_df["GroupLabel"].isin(selected_groups)
                ]
            if selected_cohorts:
                participant_df = participant_df[
                    participant_df["Cohort"].isin(selected_cohorts)
                ]
            participant_df = participant_df.drop_duplicates().sort_values(
                ["GroupLabel", participant_column]
            )
            participant_options = participant_df[participant_column].tolist()
            participant_labels = {
                row[participant_column]: f"{row[participant_column]} ({row['GroupLabel']})"
                for _, row in participant_df.iterrows()
            }
            selected_participants = st.multiselect(
                "Participants",
                options=participant_options,
                format_func=lambda value: participant_labels.get(value, str(value)),
            )
        else:
            st.caption("No participant identifier found in the dataset.")
            selected_participants = []

    filtered_df = df.copy()
    filtered_raw_df = raw_df.copy()
    if selected_groups:
        filtered_df = filtered_df[filtered_df["GroupLabel"].isin(selected_groups)]
        filtered_raw_df = filtered_raw_df[filtered_raw_df["GroupLabel"].isin(selected_groups)]
    if selected_cohorts:
        filtered_df = filtered_df[filtered_df["Cohort"].isin(selected_cohorts)]
        filtered_raw_df = filtered_raw_df[filtered_raw_df["Cohort"].isin(selected_cohorts)]

    st.subheader("Raw data preview")
    st.dataframe(filtered_df.head())

    st.subheader("Means and standard deviations by time point")
    summary_df = summarise_time_points(filtered_df)
    raw_summary_df = summarise_time_points(filtered_raw_df)
    st.dataframe(summary_df)

    st.subheader("Cortisol trajectory")
    if selected_groups and selected_phases:
        figure = make_line_plot(summary_df, selected_groups, selected_phases, use_log)
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("Select at least one group and phase to display the line plot.")

    st.subheader("Group trajectories (individual participants)")
    if (
        participant_column
        and selected_groups
        and selected_phases
        and not raw_df.empty
    ):
        spaghetti_fig = make_group_spaghetti_plot(
            filtered_raw_df,
            raw_summary_df,
            participant_column,
            selected_groups,
            selected_phases,
        )
        if spaghetti_fig is None:
            st.info(
                "Unable to display participant trajectories for the selected combination."
            )
        else:
            st.caption(
                "Raw cortisol for every participant is plotted with semi-transparent"
                " lines; bold traces show group means."
            )
            st.plotly_chart(spaghetti_fig, use_container_width=True)
    else:
        st.info(
            "Participant identifiers together with at least one group and phase are required"
            " to display the group trajectory plot."
        )

    st.subheader("Individual Values in Relation to Mean of Group")
    if selected_groups and selected_phases:
        for phase in selected_phases:
            distribution_fig = make_timepoint_distribution_plot(
                filtered_df if use_log else filtered_raw_df,
                selected_groups,
                phase,
                use_log_scale=use_log,
            )
            if distribution_fig is None:
                st.info(
                    f"No cortisol measurements available for the {phase.lower()} phase with the current selections."
                )
            else:
                st.plotly_chart(distribution_fig, use_container_width=True)
    else:
        st.info("Select at least one group and phase to display the distribution plots.")

    st.subheader("Participant overlays by phase")
    if participant_column and selected_groups and selected_phases:
        for phase in selected_phases:
            overlay_fig = make_phase_overlay_plot(
                filtered_df, participant_column, selected_groups, phase, use_log
            )
            if overlay_fig is None:
                st.info(
                    f"No cortisol measurements available for the {phase.lower()} phase with the current selections."
                )
            else:
                st.plotly_chart(overlay_fig, use_container_width=True)
    elif not participant_column:
        st.info("Participant identifiers are required to display the overlay plots.")
    else:
        st.info("Select at least one group and phase to display the overlay plots.")

    if participant_column:
        y_axis_title = (
            "Log cortisol (ln(nmol/L))" if use_log else "Cortisol (nmol/L)"
        )
        force_zero_start = not use_log
        st.subheader("Individual participant cortisol trajectories")
        if not selected_phases:
            st.info("Select at least one phase to display participant trajectories.")
        elif not selected_participants:
            st.info("Select one or more participants from the sidebar to display their trajectories.")
        else:
            for participant in selected_participants:
                display_label = participant_labels.get(participant, str(participant))
                st.markdown(f"#### {display_label}")
                participant_rows = filtered_df[filtered_df[participant_column] == participant]
                if participant_rows.empty:
                    st.warning("No data available for this participant.")
                    continue
                figure = make_participant_timecourse_plot(
                    participant_rows.iloc[0], selected_phases, y_axis_title, force_zero_start
                )
                if figure is None:
                    st.info(
                        "No cortisol measurements available for the selected phases and participant."
                    )
                else:
                    st.plotly_chart(figure, use_container_width=True)

    st.subheader("Area under the curve metrics")
    auc_df = compute_auc(filtered_df)
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

    st.markdown("### Individual AUC distributions")
    if selected_groups and selected_phases:
        auc_tabs = st.tabs(["AUCg", "AUCi"])
        for metric, tab in zip(["AUCg", "AUCi"], auc_tabs):
            with tab:
                individual_auc_fig = make_individual_auc_plot(
                    auc_df, metric, selected_groups, selected_phases
                )
                if individual_auc_fig is None:
                    st.info("No AUC data available for the selected combination.")
                else:
                    st.plotly_chart(individual_auc_fig, use_container_width=True)
    else:
        st.info("Select at least one group and phase to display the individual AUC plots.")

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
