import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ---- Page config ----
st.set_page_config(page_title="FS Sweep Visualizer (Spline)", layout="wide")


# ---- CSS ----
def _inject_bold_tick_css():
    st.markdown(
        """
        <style>
        .plotly .xtick text, .plotly .ytick text,
        .plotly .scene .xtick text, .plotly .scene .ytick text {
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---- Data loading ----
@st.cache_data(show_spinner=False)
def load_fs_sweep_xlsx(path_or_buf) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    xls = pd.ExcelFile(path_or_buf)
    for name in ["R1", "X1", "R0", "X0"]:
        if name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=name)
            # Frequency column normalization
            freq_col = None
            for c in df.columns:
                c_norm = str(c).strip().lower().replace(" ", "")
                if c_norm in ["frequency(hz)", "frequencyhz", "frequency_"]:
                    freq_col = c
                    break
                if str(c).strip().lower() in ["frequency (hz)", "frequency"]:
                    freq_col = c
                    break
            if freq_col is None:
                if "Frequency (Hz)" in df.columns:
                    freq_col = "Frequency (Hz)"
                else:
                    raise ValueError(f"Sheet '{name}' missing 'Frequency (Hz)' column")
            df = df.rename(columns={freq_col: "Frequency (Hz)"})
            df["Frequency (Hz)"] = pd.to_numeric(df["Frequency (Hz)"], errors="coerce")
            df = df.dropna(subset=["Frequency (Hz)"])
            dfs[name] = df
    return dfs


def list_case_columns(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None:
        return []
    return [c for c in df.columns if c != "Frequency (Hz)"]


def split_case_parts(cases: List[str]) -> Tuple[List[List[str]], int]:
    parts_list: List[List[str]] = []
    max_parts = 0
    for name in cases:
        parts = str(name).split("_")
        parts_list.append(parts)
        max_parts = max(max_parts, len(parts))
    for parts in parts_list:
        if len(parts) < max_parts:
            parts.extend([""] * (max_parts - len(parts)))
    return parts_list, max_parts


def build_filters_for_case_parts(all_cases: List[str]) -> List[str]:
    st.sidebar.header("Case Filters")
    if not all_cases:
        return []
    parts_list, max_parts = split_case_parts(all_cases)
    keep = np.ones(len(all_cases), dtype=bool)
    reset_clicked = st.sidebar.button("Reset filters", key="reset_case_filters", help="Select all values in all Case parts")
    for i in range(max_parts):
        col_key = f"case_part_{i+1}_ms"
        options = sorted({parts_list[j][i] for j in range(len(all_cases))})
        options_disp = [o if o != "" else "<empty>" for o in options]
        # init/sanitize
        if reset_clicked:
            st.session_state[col_key] = list(options_disp)
        elif col_key not in st.session_state:
            st.session_state[col_key] = list(options_disp)
        else:
            st.session_state[col_key] = [v for v in st.session_state[col_key] if v in options_disp] or list(options_disp)
        st.sidebar.markdown(f"Case part {i+1}")
        c1, c2 = st.sidebar.columns([1, 1])
        if c1.button("Select all", key=f"{col_key}_all"):
            st.session_state[col_key] = list(options_disp)
        if c2.button("Clear all", key=f"{col_key}_none"):
            st.session_state[col_key] = []
        selected_disp = st.sidebar.multiselect(
            label=" ", options=options_disp, default=st.session_state[col_key], key=col_key, label_visibility="collapsed"
        )
        selected_raw = ["" if s == "<empty>" else s for s in selected_disp]
        if 0 < len(selected_raw) < len(options):
            mask_i = np.array([parts_list[j][i] in selected_raw for j in range(len(all_cases))])
            keep &= mask_i
        if len(selected_raw) == 0:
            keep &= False
    return [c for c, k in zip(all_cases, keep) if k]


def compute_common_n_range(f_series: List[pd.Series], f_base: float) -> Tuple[float, float]:
    vals: List[float] = []
    for s in f_series:
        if s is None:
            continue
        v = pd.to_numeric(s, errors="coerce").dropna()
        if not v.empty:
            vals.extend([v.min() / f_base, v.max() / f_base])
    if not vals:
        return 0.0, 1.0
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    return (0.0, 1.0) if (not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi) else (lo, hi)


def add_harmonic_lines(fig: go.Figure, n_min: float, n_max: float, f_base: float, show_markers: bool, bin_width_hz: float):
    if not show_markers and (bin_width_hz is None or bin_width_hz <= 0):
        return
    shapes = []
    k_start = max(1, int(np.floor(n_min)))
    k_end = int(np.ceil(n_max))
    for k in range(k_start, k_end + 1):
        if show_markers:
            shapes.append(dict(type="line", xref="x", yref="paper", x0=k, x1=k, y0=0, y1=1, line=dict(color="rgba(0,0,0,0.3)", width=1.5)))
        if bin_width_hz and bin_width_hz > 0:
            dn = (bin_width_hz / (2.0 * f_base))
            for edge in (k - dn, k + dn):
                shapes.append(dict(type="line", xref="x", yref="paper", x0=edge, x1=edge, y0=0, y1=1, line=dict(color="rgba(0,0,0,0.2)", width=1, dash="dot")))
    fig.update_layout(shapes=fig.layout.shapes + tuple(shapes) if fig.layout.shapes else tuple(shapes))


def make_spline_traces(df: pd.DataFrame, cases: List[str], f_base: float, y_title: str, smooth: float) -> Tuple[List[go.Scatter], Optional[pd.Series]]:
    if df is None:
        return [], None
    f = df["Frequency (Hz)"]
    n = f / f_base
    traces: List[go.Scatter] = []
    for case in cases:
        if case not in df.columns:
            continue
        y = pd.to_numeric(df[case], errors="coerce")
        cd = np.column_stack([f.values])
        traces.append(
            go.Scatter(
                x=n,
                y=y,
                customdata=cd,
                mode="lines",
                name=str(case),
                line=dict(shape="spline", smoothing=float(smooth)),
                hovertemplate=(
                    "Case=%{fullData.name}<br>n=%{x:.3f}<br>f=%{customdata[0]:.1f} Hz" + f"<br>{y_title}=%{{y}}<extra></extra>"
                ),
            )
        )
    return traces, f


def apply_common_layout(fig: go.Figure, plot_height: int, y_title: str, legend_offset: float, legend_entrywidth: int):
    fig.update_layout(
        height=plot_height,
        margin=dict(l=60, r=20, t=40, b=160),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=float(legend_offset),
            xanchor="center",
            x=0.5,
            entrywidth=int(legend_entrywidth),
            entrywidthmode="pixels",
        ),
    )
    fig.update_xaxes(title_text="Harmonic number n = f / f_base", tick0=1, dtick=1)
    fig.update_yaxes(title_text=y_title)


def build_plot_spline(df: Optional[pd.DataFrame], cases: List[str], f_base: float, plot_height: int, y_title: str,
                      smooth: float, legend_offset: float, legend_entrywidth: int) -> Tuple[go.Figure, Optional[pd.Series]]:
    fig = go.Figure()
    traces, f_series = make_spline_traces(df, cases, f_base, y_title, smooth)
    for tr in traces:
        fig.add_trace(tr)
    apply_common_layout(fig, plot_height, y_title, legend_offset, legend_entrywidth)
    return fig, f_series


def build_x_over_r_spline(df_r: Optional[pd.DataFrame], df_x: Optional[pd.DataFrame], cases: List[str], f_base: float,
                          plot_height: int, seq_label: str, smooth: float, legend_offset: float, legend_entrywidth: int
                          ) -> Tuple[go.Figure, Optional[pd.Series], int, int]:
    fig = go.Figure()
    xr_dropped = 0
    xr_total = 0
    f_series = None
    eps = 1e-9
    if df_r is not None and df_x is not None:
        both = [c for c in cases if c in df_r.columns and c in df_x.columns]
        f_series = df_r["Frequency (Hz)"]
        n = f_series / f_base
        for case in both:
            r = pd.to_numeric(df_r[case], errors="coerce")
            x = pd.to_numeric(df_x[case], errors="coerce")
            denom_ok = r.abs() >= eps
            y = pd.Series(np.where(denom_ok, x / r, np.nan))
            xr_dropped += int((~denom_ok | r.isna() | x.isna()).sum())
            xr_total += int(len(r))
            cd = np.column_stack([f_series.values])
            fig.add_trace(
                go.Scatter(
                    x=n,
                    y=y,
                    customdata=cd,
                    mode="lines",
                    name=str(case),
                    line=dict(shape="spline", smoothing=float(smooth)),
                    hovertemplate=(
                        "Case=%{fullData.name}<br>n=%{x:.3f}<br>f=%{customdata[0]:.1f} Hz<br>X/R=%{y}<extra></extra>"
                    ),
                )
            )
    y_title = "X1/R1 (unitless)" if seq_label == "Positive" else "X0/R0 (unitless)"
    apply_common_layout(fig, plot_height, y_title, legend_offset, legend_entrywidth)
    return fig, f_series, xr_dropped, xr_total


def main():
    st.title("FS Sweep Visualizer (Spline)")
    _inject_bold_tick_css()

    # Data source
    default_path = "FS_sweep.xlsx"
    st.sidebar.header("Data Source")
    up = st.sidebar.file_uploader("Upload Excel", type=["xlsx"], help="If empty, loads 'FS_sweep.xlsx' from this folder.")
    try:
        if up is not None:
            data = load_fs_sweep_xlsx(up)
        elif os.path.exists(default_path):
            data = load_fs_sweep_xlsx(default_path)
            st.sidebar.info(f"Loaded local file: {default_path}")
        else:
            st.warning("Upload an Excel file or place 'FS_sweep.xlsx' here.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to load Excel: {e}")
        st.stop()

    # Controls
    st.sidebar.header("Controls")
    seq_label = st.sidebar.radio("Sequence", ["Positive", "Zero"], index=0)
    seq = ("R1", "X1") if seq_label == "Positive" else ("R0", "X0")
    base_label = st.sidebar.radio("Base frequency", ["50 Hz", "60 Hz"], index=0)
    f_base = 50.0 if base_label.startswith("50") else 60.0
    plot_height = st.sidebar.slider("Plot height (px)", min_value=250, max_value=1000, value=400, step=50)
    smooth = st.sidebar.slider("Spline smoothing", min_value=0.0, max_value=1.3, value=0.6, step=0.05)

    # Legend/Export controls
    st.sidebar.header("Legend & Export")
    legend_offset = st.sidebar.slider("Legend vertical offset", min_value=-0.60, max_value=-0.05, value=-0.25, step=0.01)
    legend_entrywidth = st.sidebar.slider("Legend column width (px)", min_value=120, max_value=320, value=180, step=10)
    download_config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": "plot",
            "scale": 2,
        }
    }

    # Cases / filters
    df_r = data.get(seq[0])
    df_x = data.get(seq[1])
    if df_r is None and df_x is None:
        st.error(f"Missing sheets for sequence '{seq_label}' ({seq[0]}/{seq[1]}).")
        st.stop()
    all_cases = sorted(list({*list_case_columns(df_r), *list_case_columns(df_x)}))
    filtered_cases = build_filters_for_case_parts(all_cases)
    if not filtered_cases:
        st.warning("No cases after filtering. Adjust filters.")
        st.stop()

    # Harmonic decorations
    show_harmonics = st.sidebar.checkbox("Show harmonic lines", value=True)
    bin_width_hz = st.sidebar.number_input("Bin width (Hz)", min_value=0.0, value=0.0, step=1.0, help="0 disables tolerance bands")

    # Build plots
    r_title = "R1 (Ω)" if seq_label == "Positive" else "R0 (Ω)"
    x_title = "X1 (Ω)" if seq_label == "Positive" else "X0 (Ω)"
    fig_r, f_r = build_plot_spline(df_r, filtered_cases, f_base, plot_height, r_title, smooth, legend_offset, legend_entrywidth)
    fig_x, f_x = build_plot_spline(df_x, filtered_cases, f_base, plot_height, x_title, smooth, legend_offset, legend_entrywidth)
    fig_xr, f_xr, xr_dropped, xr_total = build_x_over_r_spline(df_r, df_x, filtered_cases, f_base, plot_height, seq_label, smooth, legend_offset, legend_entrywidth)

    f_refs = [s for s in [f_r, f_x, f_xr] if s is not None]
    n_lo, n_hi = compute_common_n_range(f_refs, f_base)
    for fig in (fig_r, fig_x, fig_xr):
        fig.update_xaxes(range=[n_lo, n_hi])
        add_harmonic_lines(fig, n_lo, n_hi, f_base, show_harmonics, bin_width_hz)

    # Render
    st.subheader(f"Sequence: {seq_label} | Base: {int(f_base)} Hz")
    if xr_total > 0 and xr_dropped > 0:
        st.caption(f"X/R: dropped {xr_dropped} of {xr_total} points where |R| < 1e-9 or data missing.")

    st.plotly_chart(fig_r, use_container_width=True, config=download_config)
    st.plotly_chart(fig_x, use_container_width=True, config=download_config)
    st.plotly_chart(fig_xr, use_container_width=True, config=download_config)


if __name__ == "__main__":
    main()
