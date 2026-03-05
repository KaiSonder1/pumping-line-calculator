import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
# Sidebar inputs (with clear descriptions)
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
        help="How many grams of helium per second move through the pumping line (steady-state). "
             "If you don't know this, you typically estimate it from cooling power + latent heat."
    )
    P0 = st.number_input(
        "Cold-end pressure (torr)",
        value=1.9e-3,
        format="%.6g",
        help="Pressure at the cold end of the pumping line (at the coldest stage)."
    )
    T0 = st.number_input(
        "Cold-end temperature (K)",
        value=0.3,
        format="%.6g",
        help="Temperature at the cold end. Used for the design check summary."
    )

    st.header("Model settings")
    eta_model = st.selectbox(
        "Viscosity model",
        ["accurate", "simple"],
        index=0,
        help="Controls how helium viscosity depends on temperature for the non-molecular calculation."
    )
    end_corr = st.checkbox(
        "Short-tube correction (molecular flow)",
        value=True,
        help="Improves accuracy when a tube segment is short compared to its diameter."
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
        format="%.3f",
        help="Temperature of the gas at the pump inlet (usually room temperature)."
    )
    S = st.number_input(
        "Pump speed S (L/s)",
        value=500.0,
        format="%.6g",
        help="Pump speed at the pump inlet, in liters per second."
    )
    Pback = st.number_input(
        "Pump outlet/backing pressure (torr) (display only)",
        value=760.0,
        format="%.6g",
        help="This app displays this value but does not currently simulate the backing line."
    )

    st.header("Solve for unknown")
    solve_mode = st.selectbox(
        "Mode",
        ["Forward: compute pressures", "Solve: diameter of one section"],
        index=0,
        help="Forward = compute pressures for your current geometry. Solve = choose a section and solve for its diameter."
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


def make_diagram(segments, pressures=None):
    """Stepped schematic. Uses length as horizontal width and diameter as vertical height."""
    if not segments:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No sections defined", ha="center", va="center")
        ax.axis("off")
        return fig

    total_L = sum(s.length_cm for s in segments)
    max_D = max(s.diameter_cm for s in segments)

    fig_w = max(9, min(18, total_L / 12))
    fig_h = 3.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = 0.0
    for i, seg in enumerate(segments):
        d = seg.diameter_cm
        L = seg.length_cm

        ax.add_patch(Rectangle((x, -d/2), L, d, fill=False, linewidth=2))

        ax.text(
            x + L/2, d/2 + 0.12*max_D,
            f"{seg.name}  L={L:g} cm, D={d:g} cm\nT: {seg.T_in_K:g}→{seg.T_out_K:g} K",
            ha="center", va="bottom", fontsize=8
        )

        if pressures is not None and i < len(pressures):
            ax.text(
                x, -d/2 - 0.18*max_D,
                f"P{i}={pressures[i]:.3g} torr",
                ha="left", va="top", fontsize=8
            )

        x += L

    if pressures is not None:
        ax.text(
            total_L, -max_D/2 - 0.18*max_D,
            f"P{len(pressures)-1}={pressures[-1]:.3g} torr",
            ha="right", va="top", fontsize=8
        )

    ax.set_xlim(0, total_L)
    ax.set_ylim(-max_D, max_D)
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
    """Bisection solve for one section diameter such that final pressure matches target."""

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
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

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

# Always show schematic preview
st.subheader("Schematic preview")
st.pyplot(make_diagram(segments_preview, pressures=st.session_state["last_pressures"]), use_container_width=True)

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
        st.success("OK") if ssum < rhs else st.error("Too restrictive")
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
