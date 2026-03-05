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

    gas_choice = st.selectbox(
        "Helium isotope",
        ["He3", "He4"],
        index=0,
        help="Choose the helium isotope used in the pumping line."
    )
    gas = n.HE3 if gas_choice == "He3" else n.HE4

    Qm = st.number_input(
        "Mass flow (g/s)",
        value=1.8e-5,
        format="%.6g",
        help="How many grams of helium per second move through the pumping line (steady-state)."
    )
    P0 = st.number_input(
        "Cold-end pressure (torr)",
        value=1.9e-3,
        format="%.6g",
        help="Pressure at the cold end of the pumping line."
    )
    T0 = st.number_input(
        "Cold-end temperature (K)",
        value=0.3,
        format="%.6g",
        help="Temperature at the cold end (used in the design check)."
    )

    st.header("Model settings")
    eta_model = st.selectbox(
        "Viscosity model",
        ["accurate", "simple"],
        index=0
    )
    end_corr = st.checkbox(
        "Short-tube correction (molecular flow)",
        value=True
    )

    st.header("Pump (optional)")
    show_pump = st.checkbox(
        "Compute pump inlet pressure from pump speed",
        value=True,
        help="Uses q = p·S to estimate the pressure at the pump inlet for the chosen mass flow."
    )
    Tp = st.number_input(
        "Pump inlet temperature (K)",
        value=300.0,
        format="%.3f"
    )
    S = st.number_input(
        "Pump speed S (L/s)",
        value=500.0,
        format="%.6g"
    )
    Pback = st.number_input(
        "Pump outlet/backing pressure (torr) (display only)",
        value=760.0,
        format="%.6g"
    )

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
    """
    Greedy spacing to avoid overlap:
    - sort labels by true x
    - enforce min spacing between label x positions
    - shift block back into [x_min, x_max]
    Returns label_x list aligned to original ordering.
    """
    idx_sorted = sorted(range(len(xs_true)), key=lambda i: xs_true[i])
    xs_sorted = [xs_true[i] for i in idx_sorted]

    xs_lab = []
    last = -1e99
    for x in xs_sorted:
        x_new = x if (x - last) >= min_dx else (last + min_dx)
        xs_lab.append(x_new)
        last = x_new

    # Shift back if we ran past x_max
    overflow = xs_lab[-1] - x_max
    if overflow > 0:
        xs_lab = [x - overflow for x in xs_lab]

    # Shift forward if we went below x_min
    under = x_min - xs_lab[0]
    if under > 0:
        xs_lab = [x + under for x in xs_lab]

    # Map back to original order
    out = [0.0] * len(xs_true)
    for j, i in enumerate(idx_sorted):
        out[i] = xs_lab[j]
    return out


def make_diagram(segments, pressures=None, pump_info=None):
    """
    Stepped schematic:
    - segments drawn as rectangles (width=length, height=diameter)
    - segment labels are spread out (no overlap) and connected with leader lines
    - node pressures shown at boundaries, staggered vertically
    - pump symbol at the right with label
    """
    if not segments:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No sections defined", ha="center", va="center")
        ax.axis("off")
        return fig

    total_L = sum(s.length_cm for s in segments)
    max_D = max(s.diameter_cm for s in segments)

    # Pump geometry
    pad = max(10.0, 0.08 * total_L)       # cm
    pump_r = max(0.6, 0.18 * max_D)       # "height units"
    pump_x = total_L + pad + pump_r
    x_max = pump_x + pump_r + pad * 0.5

    fig_w = max(10, min(20, (total_L + 2*pad) / 10))
    fig_h = 4.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw segments
    x0 = 0.0
    mids = []
    top_y_at_mid = []
    boundaries = [0.0]

    for seg in segments:
        L = seg.length_cm
        d = seg.diameter_cm

        ax.add_patch(Rectangle((x0, -d/2), L, d, fill=False, linewidth=2))
        mids.append(x0 + L/2)
        top_y_at_mid.append(d/2)
        x0 += L
        boundaries.append(x0)

    # Spread segment label x-positions to avoid overlap
    # Choose a spacing that works well for short sections:
    min_dx = max(18.0, 0.22 * (total_L / max(1, len(segments))))
    label_xs = _spread_label_positions(mids, x_min=0.0, x_max=total_L, min_dx=min_dx)

    y_label = max_D/2 + 0.55*max_D

    for i, seg in enumerate(segments):
        # Leader line (arrow) from label to segment
        ax.annotate(
            "",
            xy=(mids[i], top_y_at_mid[i]),
            xytext=(label_xs[i], y_label - 0.12*max_D),
            arrowprops=dict(arrowstyle="-", linewidth=1)
        )
        # Label text (no overlap)
        ax.text(
            label_xs[i], y_label,
            f"{seg.name}  L={seg.length_cm:g} cm, D={seg.diameter_cm:g} cm\nT: {seg.T_in_K:g}→{seg.T_out_K:g} K",
            ha="center", va="bottom", fontsize=8
        )

    # Node pressure labels (staggered so they don't collide)
    if pressures is not None and len(pressures) == len(boundaries):
        for i, xb in enumerate(boundaries):
            # alternate vertical offsets
            yoff = (-max_D/2 - 0.30*max_D) if (i % 2 == 0) else (-max_D/2 - 0.44*max_D)
            ax.text(
                xb, yoff,
                f"P{i}={pressures[i]:.3g} torr",
                ha="center", va="top", fontsize=8
            )

    # Draw pump connection line
    ax.plot([total_L, total_L + pad*0.6], [0, 0], linewidth=2)

    # Draw pump symbol
    ax.add_patch(Circle((pump_x, 0), pump_r, fill=False, linewidth=2))
    ax.add_patch(Circle((pump_x, 0), 0.45*pump_r, fill=False, linewidth=1.5))
    # simple "rotor" arrow
    ax.add_patch(FancyArrowPatch(
        (pump_x - 0.15*pump_r, 0.10*pump_r),
        (pump_x + 0.35*pump_r, 0.10*pump_r),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=1.5
    ))
    ax.text(pump_x, pump_r + 0.22*max_D, "PUMP", ha="center", va="bottom", fontsize=9)

    # Pump label
    if pump_info is None:
        pump_info = {}
    pump_lines = []
    if "S_L_s" in pump_info:
        pump_lines.append(f"S = {pump_info['S_L_s']:.3g} L/s")
    if "Tp_K" in pump_info:
        pump_lines.append(f"T = {pump_info['Tp_K']:.3g} K")
    if "Pin_torr" in pump_info and pump_info["Pin_torr"] is not None:
        pump_lines.append(f"Pin = {pump_info['Pin_torr']:.3g} torr")
    if "q_torrL_s" in pump_info and pump_info["q_torrL_s"] is not None:
        pump_lines.append(f"q = {pump_info['q_torrL_s']:.3g} torr·L/s")

    if pump_lines:
        ax.text(
            pump_x, -pump_r - 0.22*max_D,
            "\n".join(pump_lines),
            ha="center", va="top", fontsize=8
        )

    ax.set_xlim(0, x_max)
    ax.set_ylim(-max_D - 0.7*max_D, y_label + 0.5*max_D)
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
        return None, "Even at the maximum diameter, the warm-end pressure is still below the target. Increase max diameter or reduce mass flow."
    if f_lo > 0 and f_hi > 0:
        return None, "Even at the minimum diameter, the warm-end pressure is above the target. Lower the target or allow a smaller minimum diameter."

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
# Table UI
# ----------------------------
st.subheader("Pumping line sections")

col_reset, _ = st.columns([1, 8])
with col_reset:
    if st.button("Reset to example"):
        st.session_state["segments_df"] = DEFAULT_SECTIONS.copy()
        st.session_state["last_pressures"] = None
        st.session_state["last_diag"] = None
        st.session_state["last_error"] = None
        st.rerun()

df = st.data_editor(
    st.session_state["segments_df"],
    num_rows="dynamic",
    use_container_width=True
)
st.session_state["segments_df"] = df

segments_preview = build_segments(df)

# Pump info for diagram label
pump_info = {
    "S_L_s": float(S),
    "Tp_K": float(Tp),
    "Pin_torr": None,
    "q_torrL_s": None,
}
if show_pump:
    pump_info["q_torrL_s"] = float(n.throughput_torr_L_s_from_massflow(Qm_g_s=Qm, gas=gas, T_K=Tp))
    pump_info["Pin_torr"] = float(n.pump_inlet_pressure_from_speed(Qm_g_s=Qm, gas=gas, T_K=Tp, S_L_s=S))

# Always show schematic preview (pressures overlay if computed)
st.subheader("Schematic preview")
st.pyplot(
    make_diagram(segments_preview, pressures=st.session_state["last_pressures"], pump_info=pump_info),
    use_container_width=True
)

# ----------------------------
# Compute / Solve buttons
# ----------------------------
left, right = st.columns([1.2, 1])

with left:
    if solve_mode == "Forward: compute pressures":
        if st.button("Compute", type="primary"):
            run_forward_compute(segments_preview)

        st.caption("Tip: Press **Compute** once to run the numerical calculation. If your browser didn’t refresh the diagram/results, press **Compute** again.")

    else:
        st.markdown("### Solve: diameter of one section")

        section_names = list(df["name"].astype(str).values) if len(df) else []
        if not section_names:
            st.warning("Add at least one section first.")
        else:
            section_to_solve = st.selectbox("Section to solve", section_names, index=0)

            use_pump_target = st.checkbox("Use pump inlet pressure as the target warm-end pressure", value=True)

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

    st.subheader("Diagnostics (advanced)")
    diag_df = pd.DataFrame(st.session_state["last_diag"])
    st.dataframe(diag_df, use_container_width=True, height=360)
