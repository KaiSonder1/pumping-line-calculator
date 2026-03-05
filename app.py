import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from typing import Tuple

import nunes_pumping as n

st.set_page_config(page_title="Pumping Line Calculator", layout="wide")
st.title("Pumping Line Calculator")

DEFAULT_SECTIONS = pd.DataFrame([
    {"name":"L1", "length_cm":10.0,  "diameter_cm":1.27, "T_in_K":0.3, "T_out_K":1.2,  "substeps":20},
    {"name":"L2", "length_cm":15.0,  "diameter_cm":1.90, "T_in_K":1.2, "T_out_K":4.2,  "substeps":30},
    {"name":"L3", "length_cm":100.0, "diameter_cm":6.35, "T_in_K":4.2, "T_out_K":300.0,"substeps":60},
])

# ----------------------------
# Sidebar inputs
# ----------------------------
with st.sidebar:
    st.header("Inputs")

    gas_choice = st.selectbox("Helium isotope", ["He3", "He4"], index=0)
    gas = n.HE3 if gas_choice == "He3" else n.HE4

    Qm = st.number_input(
        "Mass flow (g/s)",
        value=1.8e-5,
        format="%.6g",
        help="Mass flow = grams of helium per second moving through the line (steady state)."
    )
    P0 = st.number_input("Cold-end pressure (torr)", value=1.9e-3, format="%.6g")
    T0 = st.number_input("Cold-end temperature (K)", value=0.3, format="%.6g")

    st.header("Model settings")
    eta_model = st.selectbox("Viscosity model", ["accurate", "simple"], index=0)
    end_corr = st.checkbox("Short-tube correction (molecular flow)", value=True)

    st.header("Pump (optional)")
    show_pump = st.checkbox("Compute pump inlet pressure from pump speed", value=True)
    Tp = st.number_input("Pump inlet temperature (K)", value=300.0, format="%.3f")
    S = st.number_input("Pump speed S (L/s)", value=500.0, format="%.6g")
    Pback = st.number_input("Pump outlet/backing pressure (torr) (display only)", value=760.0, format="%.6g")

    st.header("Solve for unknown")
    solve_mode = st.selectbox(
        "Mode",
        ["Forward: compute pressures", "Solve: diameter of one section"],
        index=0
    )

# ----------------------------
# Session state
# ----------------------------
if "segments_df" not in st.session_state:
    st.session_state["segments_df"] = DEFAULT_SECTIONS.copy()

if "last_pressures" not in st.session_state:
    st.session_state["last_pressures"] = None

if "last_diag" not in st.session_state:
    st.session_state["last_diag"] = None

if "last_error" not in st.session_state:
    st.session_state["last_error"] = None


# ----------------------------
# Helpers
# ----------------------------
def build_segments(df: pd.DataFrame):
    segments = []
    for _, r in df.iterrows():
        segments.append(n.Segment(
            name=str(r["name"]),
            length_cm=float(r["length_cm"]),
            diameter_cm=float(r["diameter_cm"]),
            T_in_K=float(r["T_in_K"]),
            T_out_K=float(r["T_out_K"]),
            substeps=int(r["substeps"]),
        ))
    return segments


def _spread_label_positions(xs_true, x_min, x_max, min_dx):
    """Greedy spacing so labels never overlap horizontally."""
    idx_sorted = sorted(range(len(xs_true)), key=lambda i: xs_true[i])
    xs_sorted = [xs_true[i] for i in idx_sorted]

    xs_lab = []
    last = -1e99
    for x in xs_sorted:
        x_new = x if (x - last) >= min_dx else (last + min_dx)
        xs_lab.append(x_new)
        last = x_new

    overflow = xs_lab[-1] - x_max
    if overflow > 0:
        xs_lab = [x - overflow for x in xs_lab]

    under = x_min - xs_lab[0]
    if under > 0:
        xs_lab = [x + under for x in xs_lab]

    out = [0.0] * len(xs_true)
    for j, i in enumerate(idx_sorted):
        out[i] = xs_lab[j]
    return out


def make_diagram(segments, pressures=None, pump_info=None):
    """
    Scaled schematic:
    - segments: rectangles (width = length_cm, height = diameter_cm)
    - labels: spaced out + leader lines
    - pump: scaled body + rotor symbol, sized relative to line length and max diameter
    """
    if not segments:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No sections defined", ha="center", va="center")
        ax.axis("off")
        return fig

    total_L = sum(s.length_cm for s in segments)
    max_D = max(s.diameter_cm for s in segments)

    # Pump sizing (scales with line length + diameter)
    pump_w = max(0.18 * total_L, 3.5 * max_D)   # width in "cm-length units"
    pump_h = 1.6 * max_D                        # height in "cm-diameter units"
    gap = max(0.04 * total_L, 2.0 * max_D)      # spacing from end of line to pump

    pump_x0 = total_L + gap
    pump_x1 = pump_x0 + pump_w
    x_max = pump_x1 + 0.06 * total_L

    fig_w = max(11, min(22, (x_max) / 10))
    fig_h = 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw segments
    x0 = 0.0
    mids, top_y_at_mid, boundaries = [], [], [0.0]
    for seg in segments:
        L, d = seg.length_cm, seg.diameter_cm
        ax.add_patch(Rectangle((x0, -d/2), L, d, fill=False, linewidth=2))
        mids.append(x0 + L/2)
        top_y_at_mid.append(d/2)
        x0 += L
        boundaries.append(x0)

    # Labels: spread out and connect with leader lines
    min_dx = max(18.0, 0.22 * (total_L / max(1, len(segments))))
    label_xs = _spread_label_positions(mids, x_min=0.0, x_max=total_L, min_dx=min_dx)
    y_label = max_D/2 + 0.60*max_D

    for i, seg in enumerate(segments):
        ax.annotate(
            "",
            xy=(mids[i], top_y_at_mid[i]),
            xytext=(label_xs[i], y_label - 0.12*max_D),
            arrowprops=dict(arrowstyle="-", linewidth=1)
        )
        ax.text(
            label_xs[i], y_label,
            f"{seg.name}  L={seg.length_cm:g} cm, D={seg.diameter_cm:g} cm\nT: {seg.T_in_K:g}→{seg.T_out_K:g} K",
            ha="center", va="bottom", fontsize=8
        )

    # Node pressures (staggered)
    if pressures is not None and len(pressures) == len(boundaries):
        for i, xb in enumerate(boundaries):
            yoff = (-max_D/2 - 0.30*max_D) if (i % 2 == 0) else (-max_D/2 - 0.46*max_D)
            ax.text(xb, yoff, f"P{i}={pressures[i]:.3g} torr", ha="center", va="top", fontsize=8)

    # Connection line to pump
    ax.plot([total_L, pump_x0], [0, 0], linewidth=2)

    # Pump body (scaled)
    ax.add_patch(Rectangle((pump_x0, -pump_h/2), pump_w, pump_h, fill=False, linewidth=2))
    ax.text((pump_x0 + pump_x1)/2, pump_h/2 + 0.20*max_D, "PUMP", ha="center", va="bottom", fontsize=10)

    # Rotor inside pump (scaled)
    cx = pump_x0 + 0.38 * pump_w
    cy = 0.0
    r_outer = 0.32 * pump_h
    r_inner = 0.16 * pump_h

    ax.add_patch(Circle((cx, cy), r_outer, fill=False, linewidth=2))
    ax.add_patch(Circle((cx + 0.10*r_outer, cy), r_inner, fill=False, linewidth=1.5))

    # vane/rotation arrow
    ax.add_patch(FancyArrowPatch(
        (cx - 0.20*r_outer, cy + 0.15*r_outer),
        (cx + 0.35*r_outer, cy + 0.15*r_outer),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.5
    ))

    # Pump label (specs)
    if pump_info is None:
        pump_info = {}

    lines = []
    if "S_L_s" in pump_info:
        lines.append(f"S = {pump_info['S_L_s']:.3g} L/s")
    if "Tp_K" in pump_info:
        lines.append(f"T = {pump_info['Tp_K']:.3g} K")
    if "Pin_torr" in pump_info and pump_info["Pin_torr"] is not None:
        lines.append(f"Pin = {pump_info['Pin_torr']:.3g} torr")
    if "q_torrL_s" in pump_info and pump_info["q_torrL_s"] is not None:
        lines.append(f"q = {pump_info['q_torrL_s']:.3g} torr·L/s")

    if lines:
        ax.text(pump_x0 + 0.72*pump_w, -0.05*pump_h, "\n".join(lines), ha="left", va="top", fontsize=9)

    ax.set_xlim(0, x_max)
    ax.set_ylim(-max_D - 0.85*max_D, y_label + 0.60*max_D)
    ax.axis("off")
    ax.set_title("Pumping line schematic", fontsize=12)
    return fig


def run_forward_compute(segments):
    st.session_state["last_error"] = None
    try:
        pressures, diags = n.propagate_line(
            segments=segments,
            gas=gas,
            P0_torr=P0,
            Qm_g_s=Qm,
            eta_model=eta_model,
            use_end_correction=end_corr,
        )
        st.session_state["last_pressures"] = pressures
        st.session_state["last_diag"] = diags
    except n.InfeasibleFlowError as e:
        st.session_state["last_pressures"] = None
        st.session_state["last_diag"] = None
        st.session_state["last_error"] = str(e)


def solve_diameter_for_target_end_pressure(
    df: pd.DataFrame,
    target_end_pressure_torr: float,
    section_name: str,
    dmin_cm: float,
    dmax_cm: float,
    max_iter: int = 60
) -> Tuple[float | None, str]:

    def compute_endP_for_d(d_cm: float) -> float:
        df2 = df.copy()
        df2.loc[df2["name"] == section_name, "diameter_cm"] = float(d_cm)
        segs = build_segments(df2)
        try:
            pressures, _ = n.propagate_line(
                segments=segs,
                gas=gas,
                P0_torr=P0,
                Qm_g_s=Qm,
                eta_model=eta_model,
                use_end_correction=end_corr,
            )
            return pressures[-1]
        except n.InfeasibleFlowError:
            return 0.0

    if dmin_cm <= 0 or dmax_cm <= dmin_cm:
        return None, "Diameter bounds are invalid."

    lo, hi = dmin_cm, dmax_cm
    f_lo = compute_endP_for_d(lo) - target_end_pressure_torr
    f_hi = compute_endP_for_d(hi) - target_end_pressure_torr

    if f_lo < 0 and f_hi < 0:
        return None, "Even at the maximum diameter, warm-end pressure is below target. Increase max diameter or reduce mass flow."
    if f_lo > 0 and f_hi > 0:
        return None, "Even at the minimum diameter, warm-end pressure is above target. Lower target or allow smaller minimum diameter."

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = compute_endP_for_d(mid) - target_end_pressure_torr
        if abs(f_mid) < 1e-6:
            return mid, "Solved."
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi), "Solved (max iterations reached)."


# ----------------------------
# Table UI (FIXED: non-glitchy editing)
# ----------------------------
st.subheader("Pumping line sections")

col_reset, col_hint = st.columns([1, 6])
with col_reset:
    if st.button("Reset to example"):
        st.session_state["segments_df"] = DEFAULT_SECTIONS.copy()
        st.session_state["last_pressures"] = None
        st.session_state["last_diag"] = None
        st.session_state["last_error"] = None
        st.rerun()

with col_hint:
    st.caption("Editing tip: change values in the table, then click **Apply table changes**. This prevents edits from being lost while the app reruns.")

with st.form("table_form", clear_on_submit=False):
    edited_df = st.data_editor(
        st.session_state["segments_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="segments_editor"
    )
    apply_clicked = st.form_submit_button("Apply table changes")

if apply_clicked:
    st.session_state["segments_df"] = edited_df
    # Clear results so the user knows they need to recompute
    st.session_state["last_pressures"] = None
    st.session_state["last_diag"] = None
    st.session_state["last_error"] = None
    st.rerun()

df = st.session_state["segments_df"]
segments_preview = build_segments(df)

# Pump info for diagram label
pump_info = {"S_L_s": float(S), "Tp_K": float(Tp), "Pin_torr": None, "q_torrL_s": None}
if show_pump:
    pump_info["q_torrL_s"] = float(n.throughput_torr_L_s_from_massflow(Qm_g_s=Qm, gas=gas, T_K=Tp))
    pump_info["Pin_torr"] = float(n.pump_inlet_pressure_from_speed(Qm_g_s=Qm, gas=gas, T_K=Tp, S_L_s=S))

st.subheader("Schematic preview")
st.pyplot(make_diagram(segments_preview, pressures=st.session_state["last_pressures"], pump_info=pump_info), use_container_width=True)

# ----------------------------
# Compute / Solve buttons
# ----------------------------
left, right = st.columns([1.2, 1])

with left:
    if solve_mode == "Forward: compute pressures":
        if st.button("Compute", type="primary"):
            run_forward_compute(segments_preview)
        st.caption("Tip: Press **Compute** once for the calculation. If the diagram/results didn't update, press **Compute** again.")
    else:
        st.markdown("### Solve: diameter of one section")

        section_names = list(df["name"].astype(str).values) if len(df) else []
        if not section_names:
            st.warning("Add at least one section first.")
        else:
            section_to_solve = st.selectbox("Section to solve", section_names, index=0)
            use_pump_target = st.checkbox("Use pump inlet pressure as target warm-end pressure", value=True)

            if use_pump_target and show_pump:
                target = n.pump_inlet_pressure_from_speed(Qm_g_s=Qm, gas=gas, T_K=Tp, S_L_s=S)
                st.write(f"Target warm-end pressure: **{target:.6g} torr**")
            else:
                target = st.number_input("Target warm-end pressure (torr)", value=2e-3, format="%.6g")

            dmin = st.number_input("Min diameter to try (cm)", value=0.2, format="%.6g")
            dmax = st.number_input("Max diameter to try (cm)", value=10.0, format="%.6g")

            if st.button("Solve diameter", type="primary"):
                d_sol, msg = solve_diameter_for_target_end_pressure(
                    df=df,
                    target_end_pressure_torr=float(target),
                    section_name=str(section_to_solve),
                    dmin_cm=float(dmin),
                    dmax_cm=float(dmax),
                )
                if d_sol is None:
                    st.error(msg)
                else:
                    st.success(f"{msg}  Diameter for {section_to_solve}: **{d_sol:.4g} cm**")
                    df2 = df.copy()
                    df2.loc[df2["name"] == section_to_solve, "diameter_cm"] = float(d_sol)
                    st.session_state["segments_df"] = df2
                    run_forward_compute(build_segments(df2))
                    st.rerun()

with right:
    st.subheader("Design check")
    try:
        ssum = n.sum_li_over_ai3(segments_preview)
        rhs = n.nunes_constraint_rhs(P0_torr=P0, T0_K=T0, Qm_g_s=Qm, gas=gas)
        st.write(f"Restriction score Σ(l/a³): **{ssum:.6g} cm⁻²**")
        st.write(f"Maximum allowed: **{rhs:.6g} cm⁻²**")
        if ssum < rhs:
            st.success("OK")
        else:
            st.error("Too restrictive")
    except Exception as e:
        st.warning(f"Design check unavailable: {e}")

    st.subheader("Pump")
    if show_pump:
        q = n.throughput_torr_L_s_from_massflow(Qm_g_s=Qm, gas=gas, T_K=Tp)
        Pin = n.pump_inlet_pressure_from_speed(Qm_g_s=Qm, gas=gas, T_K=Tp, S_L_s=S)
        st.write(f"Throughput q: **{q:.6g} torr·L/s**")
        st.write(f"Pump inlet pressure: **{Pin:.6g} torr**")
        st.write(f"Pump outlet/backing pressure (display): **{Pback:.6g} torr**")

# ----------------------------
# Results
# ----------------------------
if st.session_state["last_error"]:
    st.error("Computation failed (inputs are too restrictive for the chosen mass flow).")
    st.warning(st.session_state["last_error"])

if st.session_state["last_pressures"] is not None:
    st.subheader("Computed node pressures (torr)")
    pressures = st.session_state["last_pressures"]
    node_df = pd.DataFrame({"node": [f"P{i}" for i in range(len(pressures))], "P_torr": pressures})
    st.dataframe(node_df, use_container_width=True)
