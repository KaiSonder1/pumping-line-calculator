import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import nunes_pumping as n

st.set_page_config(page_title="Cryogenic Pumping Line Calculator", layout="wide")
st.title("Cryogenic Pumping Line Calculator (Nunes chapter, section-by-section)")

st.markdown("""
Model the pumping line as a **series of sections** (Fig. 2.15 style).
Each section has: **length (cm), diameter (cm), T_in (K), T_out (K)**.
Then we compute pressures across the line and draw a stepped technical diagram (Fig. 2.16 style).
""")

DEFAULT_SECTIONS = pd.DataFrame([
    {"name":"L1", "length_cm":10.0,  "diameter_cm":1.27, "T_in_K":0.3, "T_out_K":1.2,  "substeps":20},
    {"name":"L2", "length_cm":15.0,  "diameter_cm":1.90, "T_in_K":1.2, "T_out_K":4.2,  "substeps":30},
    {"name":"L3", "length_cm":100.0, "diameter_cm":6.35, "T_in_K":4.2, "T_out_K":300.0,"substeps":60},
])

with st.sidebar:
    st.header("Global inputs")
    gas_choice = st.selectbox("Gas", ["He3", "He4"], index=0)
    gas = n.HE3 if gas_choice == "He3" else n.HE4

    Qm = st.number_input("Mass flow Qm (g/s)", value=1.8e-5, format="%.6g")
    P0 = st.number_input("Cold-end pressure P0 (torr)", value=1.9e-3, format="%.6g")
    T0 = st.number_input("Cold-end temperature T0 (K)", value=0.3, format="%.6g")

    st.header("Model options")
    eta_model = st.selectbox("Viscosity model for Eq (2.14)", ["accurate", "simple"], index=0)
    end_corr = st.checkbox("Apply short-tube correction in molecular flow (Eq 2.11 idea)", value=True)

    st.header("Pump block (optional)")
    show_pump = st.checkbox("Show pump inlet pressure from pump speed (q = pS)", value=True)
    Tp = st.number_input("Pump inlet temperature (K)", value=300.0, format="%.3f")
    S = st.number_input("Pump speed S (L/s)", value=500.0, format="%.6g")
    Pback = st.number_input("Pump outlet/backing pressure (torr) (display only)", value=760.0, format="%.6g")

st.subheader("Pumping line sections")

if "segments_df" not in st.session_state:
    st.session_state["segments_df"] = DEFAULT_SECTIONS.copy()

col_reset, col_spacer = st.columns([1, 6])
with col_reset:
    if st.button("Reset to example"):
        st.session_state["segments_df"] = DEFAULT_SECTIONS.copy()
        st.rerun()

df = st.data_editor(
    st.session_state["segments_df"],
    num_rows="dynamic",
    use_container_width=True
)
st.session_state["segments_df"] = df

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
    # Horizontal stepped diagram: width = length, height = diameter
    total_L = sum(s.length_cm for s in segments)
    max_D = max(s.diameter_cm for s in segments) if segments else 1.0

    fig_w = max(8, min(16, total_L / 15))
    fig_h = 3.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = 0.0
    for i, seg in enumerate(segments):
        d = seg.diameter_cm
        ax.add_patch(Rectangle((x, -d/2), seg.length_cm, d, fill=False, linewidth=2))

        # Labels
        ax.text(x + seg.length_cm/2, d/2 + 0.15*max_D,
                f"{seg.name}\nL={seg.length_cm:g} cm\nD={d:g} cm\nT: {seg.T_in_K:g}→{seg.T_out_K:g} K",
                ha="center", va="bottom", fontsize=8)

        # Node pressure labels
        if pressures is not None and i < len(pressures):
            ax.text(x, -d/2 - 0.2*max_D, f"P{i}={pressures[i]:.3g} torr",
                    ha="left", va="top", fontsize=8)

        x += seg.length_cm

    if pressures is not None:
        ax.text(x, -max_D/2 - 0.2*max_D, f"P{len(pressures)-1}={pressures[-1]:.3g} torr",
                ha="right", va="top", fontsize=8)

    ax.set_xlim(0, total_L)
    ax.set_ylim(-max_D, max_D)
    ax.axis("off")
    ax.set_title("Pumping line schematic (stepped diameters, like Fig. 2.16)", fontsize=12)
    return fig

left, right = st.columns([1.2, 1])

with left:
    if st.button("Compute", type="primary"):
        segments = build_segments(df)

        # Basic sanity checks
        for s in segments:
            if s.length_cm <= 0 or s.diameter_cm <= 0 or s.T_in_K <= 0 or s.T_out_K <= 0:
                st.error("All lengths, diameters, and temperatures must be > 0.")
                st.stop()

        try:
            pressures, diags = n.propagate_line(
                segments=segments,
                gas=gas,
                P0_torr=P0,
                Qm_g_s=Qm,
                eta_model=eta_model,
                use_end_correction=end_corr,
            )

            st.success("Computed successfully.")

            st.subheader("Node pressures (torr)")
            node_df = pd.DataFrame({
                "node": [f"P{i}" for i in range(len(pressures))],
                "P_torr": pressures,
            })
            st.dataframe(node_df, use_container_width=True)

            st.subheader("Technical schematic")
            fig = make_diagram(segments, pressures=pressures)
            st.pyplot(fig, use_container_width=True)

            st.subheader("Per-slice diagnostics (regime + L/a)")
            diag_df = pd.DataFrame(diags)
            st.dataframe(diag_df, use_container_width=True, height=360)

        except n.InfeasibleFlowError as e:
            st.error("This set of inputs is physically infeasible for the model (pressure collapses to ~0).")
            st.warning(str(e))
            st.info("Try: reduce Qm, increase diameter of the narrowest section, shorten that section, or reset to the example.")
        except Exception as e:
            st.error("Unexpected error:")
            st.exception(e)

with right:
    st.subheader("Design check (Eqs 2.18–2.19)")
    segments_preview = build_segments(df)
    ssum = n.sum_li_over_ai3(segments_preview)
    rhs = n.nunes_constraint_rhs(P0_torr=P0, T0_K=T0, Qm_g_s=Qm, gas=gas)

    st.write(f"Σ(lᵢ / aᵢ³) = **{ssum:.6g} cm⁻²**")
    st.write(f"Max allowed Σ = P0 / (c √T0 Qm) = **{rhs:.6g} cm⁻²**")
    if ssum < rhs:
        st.success("Constraint satisfied.")
    else:
        st.error("Constraint fails (increase diameter / shorten narrow sections / reduce Qm).")

    st.subheader("Pump block")
    if show_pump:
        q = n.throughput_torr_L_s_from_massflow(Qm_g_s=Qm, gas=gas, T_K=Tp)
        Pin = n.pump_inlet_pressure_from_speed(Qm_g_s=Qm, gas=gas, T_K=Tp, S_L_s=S)
        st.write(f"Throughput at pump inlet q = **{q:.6g} torr·L/s**")
        st.write(f"Pump inlet pressure Pin = q/S = **{Pin:.6g} torr**")
        st.write(f"Pump outlet/backing pressure (display): **{Pback:.6g} torr**")
