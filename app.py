import streamlit as st
import pandas as pd

import nunes_pumping as n

st.set_page_config(page_title="Cryogenic Pumping Line Calculator", layout="wide")

st.title("Cryogenic Pumping Line Calculator (Nunes chapter, section-by-section)")

st.markdown("""
This calculator follows the **Pumps and Plumbing** chapter you photographed (Geoffrey Nunes, Jr.).
You model a pumping line as a **series of sections** (Fig. 2.15 style). Each section has:

- length (cm)
- diameter (cm)
- temperature at the start and end of that section (K)

You also provide:

- cold-end pressure **P0** (torr)
- mass flow **Qm** (g/s)

The app then computes **P_in → P_out** across each section using:
- molecular-flow gradient formula (Eq. 2.15) when L/a > 1
- transition/viscous approximation (Eq. 2.14) otherwise (applied on small slices)

No inputs are saved by this app.
""")

with st.sidebar:
    st.header("Global inputs")

    gas_choice = st.selectbox("Gas", ["He3", "He4"], index=0)
    gas = n.HE3 if gas_choice == "He3" else n.HE4

    Qm = st.number_input("Mass flow Qm (g/s)", value=1.8e-5, format="%.6g", help="Paper example: ~1.8e-5 g/s")
    P0 = st.number_input("Cold-end pressure P0 (torr)", value=1.9e-3, format="%.6g", help="Paper example: ~1.9e-3 torr at 0.3 K")
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
    st.session_state["segments_df"] = pd.DataFrame([
        {"name":"L1", "length_cm":10.0,  "diameter_cm":1.27, "T_in_K":0.3, "T_out_K":1.2,  "substeps":20},
        {"name":"L2", "length_cm":15.0,  "diameter_cm":1.90, "T_in_K":1.2, "T_out_K":4.2,  "substeps":30},
        {"name":"L3", "length_cm":100.0, "diameter_cm":6.35, "T_in_K":4.2, "T_out_K":300.0,"substeps":60},
    ])

df = st.data_editor(
    st.session_state["segments_df"],
    num_rows="dynamic",
    use_container_width=True
)
st.session_state["segments_df"] = df

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

left, right = st.columns([1,1])

with left:
    if st.button("Compute", type="primary"):
        pressures, diags = n.propagate_line(
            segments=segments,
            gas=gas,
            P0_torr=P0,
            Qm_g_s=Qm,
            eta_model=eta_model,
            use_end_correction=end_corr,
        )

        st.success("Done. Scroll down for results.")

        st.subheader("Node pressures (torr)")
        node_df = pd.DataFrame({
            "node": [f"P{i}" for i in range(len(pressures))],
            "P_torr": pressures,
        })
        st.dataframe(node_df, use_container_width=True)

        st.subheader("Per-slice diagnostics (regime + L/a)")
        st.caption("This is the transparency table: you can see where the model switches regime.")
        diag_df = pd.DataFrame(diags)
        st.dataframe(diag_df, use_container_width=True, height=360)

with right:
    st.subheader("Design check (Eqs 2.18–2.19)")
    s = n.sum_li_over_ai3(segments)
    rhs = n.nunes_constraint_rhs(P0_torr=P0, T0_K=T0, Qm_g_s=Qm, gas=gas)

    st.write(f"Σ(lᵢ / aᵢ³) = **{s:.6g} cm⁻²**")
    st.write(f"Max allowed Σ = P0 / (c √T0 Qm) = **{rhs:.6g} cm⁻²**")
    if s < rhs:
        st.success("Constraint satisfied (easier to reach target).")
    else:
        st.error("Constraint fails (increase diameter of the narrow sections, shorten them, or reduce Qm).")

    st.subheader("Pump block")
    if show_pump:
        q = n.throughput_torr_L_s_from_massflow(Qm_g_s=Qm, gas=gas, T_K=Tp)
        Pin = n.pump_inlet_pressure_from_speed(Qm_g_s=Qm, gas=gas, T_K=Tp, S_L_s=S)
        st.write(f"Throughput at pump inlet q = **{q:.6g} torr·L/s**")
        st.write(f"Pump inlet pressure Pin = q/S = **{Pin:.6g} torr**")
        st.write(f"Pump outlet/backing pressure (display): **{Pback:.6g} torr**")
    else:
        st.info("Enable the pump block in the sidebar to compute pump inlet pressure from speed.")
