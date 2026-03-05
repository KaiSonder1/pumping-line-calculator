from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple, Dict
import math

NA = 6.02214076e23
k_erg = 1.380649e-16
TORR_TO_DYNE_PER_CM2 = 1333.2236842105262

GasName = Literal["He3", "He4"]

class InfeasibleFlowError(ValueError):
    """Raised when the requested flow + geometry make pressure collapse to ~0."""
    pass

@dataclass(frozen=True)
class Gas:
    name: GasName
    molar_mass_g_mol: float
    hard_sphere_d_angstrom: float

    @property
    def molecule_mass_g(self) -> float:
        return self.molar_mass_g_mol / NA

    @property
    def d_cm(self) -> float:
        return self.hard_sphere_d_angstrom * 1e-8

HE3 = Gas("He3", molar_mass_g_mol=3.016,  hard_sphere_d_angstrom=2.2)
HE4 = Gas("He4", molar_mass_g_mol=4.0026, hard_sphere_d_angstrom=2.2)

def viscosity_kennard_he4_g_cm_s(T_K: float, model: Literal["simple", "accurate"]="accurate") -> float:
    if T_K <= 0:
        raise ValueError("T must be > 0 K")
    if model == "simple":
        return 7.5e-6 * math.sqrt(T_K)
    return 5.18e-6 * (T_K ** 0.64)

def viscosity_kennard(gas: Gas, T_K: float, model: Literal["simple", "accurate"]="accurate") -> float:
    eta4 = viscosity_kennard_he4_g_cm_s(T_K, model=model)
    if gas.name == "He4":
        return eta4
    return eta4 * math.sqrt(3.0/4.0)

def mean_free_path_helium_cm(T_K: float, P_torr: float) -> float:
    if T_K <= 0 or P_torr <= 0:
        raise ValueError("T and P must be > 0")
    return 4.8e-5 * T_K / P_torr

def regime_from_L_over_a(L_cm: float, a_cm: float) -> Literal["viscous", "transition", "molecular"]:
    x = L_cm / a_cm
    if x < 0.01:
        return "viscous"
    if x > 1.0:
        return "molecular"
    return "transition"

def transmission_probability_santeler(L_cm: float, a_cm: float) -> float:
    if L_cm <= 0 or a_cm <= 0:
        raise ValueError("L and a must be > 0")
    R = a_cm
    x = L_cm / R
    le_over_l = 1.0 + 1.0 / (3.0 + (3.0/7.0)*x)
    le_over_R = (L_cm * le_over_l) / R
    tau = 1.0 / (1.0 + (3.0/8.0)*le_over_R)
    return max(0.0, min(1.0, tau))

def clausing_K_approx(L_cm: float, a_cm: float) -> float:
    tau = transmission_probability_santeler(L_cm, a_cm)
    return min(1.0, 2.0 * tau)

def molecular_end_correction_factor(L_cm: float, a_cm: float) -> float:
    K = clausing_K_approx(L_cm, a_cm)
    return (3.0 * L_cm / (8.0 * a_cm)) * K

def step_molecular_P2_torr(
    gas: Gas,
    P1_torr: float,
    T1_K: float,
    T2_K: float,
    a_cm: float,
    l_cm: float,
    Qm_g_s: float,
    use_end_correction: bool = True
) -> float:
    if P1_torr <= 0:
        raise InfeasibleFlowError("Upstream pressure reached ~0 torr (design too restrictive for this Qm).")
    if any(x <= 0 for x in (T1_K, T2_K, a_cm, l_cm, Qm_g_s)):
        raise ValueError("All inputs must be > 0")

    P1 = P1_torr * TORR_TO_DYNE_PER_CM2
    m = gas.molecule_mass_g

    coeff = (4.0 * (a_cm**3) / (3.0 * l_cm)) * math.sqrt(2.0 * math.pi * m / k_erg)
    if use_end_correction:
        coeff *= molecular_end_correction_factor(l_cm, a_cm)

    drive = (P1 / math.sqrt(T1_K)) - (Qm_g_s / coeff)
    if drive <= 0:
        raise InfeasibleFlowError(
            "Molecular-flow step would require negative pressure. "
            "Reduce Qm, increase diameter, or shorten the narrow section."
        )

    P2 = math.sqrt(T2_K) * drive
    P2_torr = P2 / TORR_TO_DYNE_PER_CM2
    return max(0.0, P2_torr)

def step_transition_viscous_P2_torr(
    gas: Gas,
    P1_torr: float,
    T_K: float,
    a_cm: float,
    l_cm: float,
    Qm_g_s: float,
    eta_model: Literal["simple", "accurate"]="accurate",
) -> float:
    if P1_torr <= 0:
        raise InfeasibleFlowError("Upstream pressure reached ~0 torr (design too restrictive for this Qm).")
    if any(x <= 0 for x in (T_K, a_cm, l_cm, Qm_g_s)):
        raise ValueError("All inputs must be > 0")

    P1 = P1_torr * TORR_TO_DYNE_PER_CM2
    d = gas.d_cm
    eta = viscosity_kennard(gas, T_K, model=eta_model)
    m = gas.molecule_mass_g
    Ndot = Qm_g_s / m

    Z = 8.0 * l_cm / (math.pi * (a_cm**4))
    Px = (64.0 * math.sqrt(2.0) * k_erg * T_K) / (9.0 * (math.pi**2) * a_cm * (d**2))

    inside = (P1 + Px)**2 - 2.0 * eta * Ndot * k_erg * Z * T_K
    if inside <= 0:
        raise InfeasibleFlowError(
            "Transition/viscous step would make pressure imaginary (inside sqrt <= 0). "
            "Reduce Qm or increase diameter/shorten length."
        )

    P2 = math.sqrt(inside) - Px
    return max(0.0, P2 / TORR_TO_DYNE_PER_CM2)

@dataclass
class Segment:
    name: str
    length_cm: float
    diameter_cm: float
    T_in_K: float
    T_out_K: float
    substeps: int = 20

    def a_cm(self) -> float:
        return 0.5 * self.diameter_cm

def propagate_line(
    segments: List[Segment],
    gas: Gas,
    P0_torr: float,
    Qm_g_s: float,
    eta_model: Literal["simple", "accurate"]="accurate",
    use_end_correction: bool = True,
) -> Tuple[List[float], List[Dict]]:

    if P0_torr <= 0 or Qm_g_s <= 0:
        raise ValueError("P0 and Qm must be > 0")

    P = P0_torr
    pressures = [P]
    diags: List[Dict] = []

    for seg in segments:
        a = seg.a_cm()
        n = max(1, int(seg.substeps))
        dl = seg.length_cm / n

        for j in range(n):
            T1 = seg.T_in_K + (j / n) * (seg.T_out_K - seg.T_in_K)
            T2 = seg.T_in_K + ((j + 1) / n) * (seg.T_out_K - seg.T_in_K)
            Tmid = 0.5 * (T1 + T2)

            Lmfp = mean_free_path_helium_cm(Tmid, max(P, 1e-30))
            reg = regime_from_L_over_a(Lmfp, a)

            Pin = P
            try:
                if reg == "molecular":
                    P = step_molecular_P2_torr(
                        gas=gas, P1_torr=Pin, T1_K=T1, T2_K=T2,
                        a_cm=a, l_cm=dl, Qm_g_s=Qm_g_s,
                        use_end_correction=use_end_correction
                    )
                else:
                    P = step_transition_viscous_P2_torr(
                        gas=gas, P1_torr=Pin, T_K=Tmid,
                        a_cm=a, l_cm=dl, Qm_g_s=Qm_g_s,
                        eta_model=eta_model
                    )
            except InfeasibleFlowError as e:
                raise InfeasibleFlowError(
                    f"{e}\n\nWhere it happened: segment={seg.name}, slice={j+1}/{n}, "
                    f"Pin={Pin:.3g} torr, T~{Tmid:.3g} K, diameter={seg.diameter_cm:.3g} cm, length_slice={dl:.3g} cm"
                ) from e

            diags.append({
                "segment": seg.name,
                "slice": j + 1,
                "slices_total": n,
                "P_in_torr": Pin,
                "P_out_torr": P,
                "T_mid_K": Tmid,
                "mean_free_path_cm": Lmfp,
                "a_cm": a,
                "L_over_a": Lmfp / a,
                "regime": reg,
            })

        pressures.append(P)

    return pressures, diags

def sum_li_over_ai3(segments: List[Segment]) -> float:
    s = 0.0
    for seg in segments:
        a = seg.a_cm()
        s += seg.length_cm / (a**3)
    return s

def nunes_constraint_rhs(P0_torr: float, T0_K: float, Qm_g_s: float, gas: Gas) -> float:
    if P0_torr <= 0 or T0_K <= 0 or Qm_g_s <= 0:
        raise ValueError("P0, T0, Qm must be > 0")
    c = 1.2 if gas.name == "He3" else 1.2 * math.sqrt(4.0/3.0)
    return P0_torr / (c * math.sqrt(T0_K) * Qm_g_s)

def throughput_torr_L_s_from_massflow(Qm_g_s: float, gas: Gas, T_K: float) -> float:
    R_L_torr = 62.3637
    n_dot = Qm_g_s / gas.molar_mass_g_mol
    return n_dot * R_L_torr * T_K

def pump_inlet_pressure_from_speed(Qm_g_s: float, gas: Gas, T_K: float, S_L_s: float) -> float:
    q = throughput_torr_L_s_from_massflow(Qm_g_s, gas, T_K)
    return q / max(S_L_s, 1e-30)
