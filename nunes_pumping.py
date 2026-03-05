from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple, Dict
import math

# -----------------------------
# Units / constants (cgs-based)
# -----------------------------
# The chapter (Nunes, "Pumps and Plumbing") uses a mixture of torr, cm, g, and K.
# This module implements the key equations in a consistent CGS core and converts
# torr <-> dyn/cm^2 where required.

NA = 6.02214076e23                 # 1/mol
k_erg = 1.380649e-16               # erg/K (1 erg = 1 dyn*cm)
TORR_TO_DYNE_PER_CM2 = 1333.2236842105262  # 1 torr = 133.322 Pa = 1333.22 dyn/cm^2

GasName = Literal["He3", "He4"]

@dataclass(frozen=True)
class Gas:
    name: GasName
    molar_mass_g_mol: float
    hard_sphere_d_angstrom: float  # paper uses d ~ 2.2 Å for helium

    @property
    def molecule_mass_g(self) -> float:
        return self.molar_mass_g_mol / NA

    @property
    def d_cm(self) -> float:
        return self.hard_sphere_d_angstrom * 1e-8  # Å -> cm

HE3 = Gas("He3", molar_mass_g_mol=3.016,  hard_sphere_d_angstrom=2.2)
HE4 = Gas("He4", molar_mass_g_mol=4.0026, hard_sphere_d_angstrom=2.2)

def viscosity_kennard_he4_g_cm_s(T_K: float, model: Literal["simple", "accurate"]="accurate") -> float:
    """
    Nunes Eq (2.4) (simple):   eta ≈ 7.5e-6 * sqrt(T)    [g/(cm s)] for 4He
    Nunes Eq (2.5) (accurate): eta = 5.18e-6 * T^0.64    [g/(cm s)] for 4He
    """
    if T_K <= 0:
        raise ValueError("T must be > 0 K")
    if model == "simple":
        return 7.5e-6 * math.sqrt(T_K)
    return 5.18e-6 * (T_K ** 0.64)

def viscosity_kennard(gas: Gas, T_K: float, model: Literal["simple", "accurate"]="accurate") -> float:
    """
    Nunes note: to get theoretical viscosity of 3He, multiply 4He by sqrt(3/4).
    """
    eta4 = viscosity_kennard_he4_g_cm_s(T_K, model=model)
    if gas.name == "He4":
        return eta4
    return eta4 * math.sqrt(3.0/4.0)

def mean_free_path_helium_cm(T_K: float, P_torr: float) -> float:
    """
    Nunes Eq (2.3): L = 4.8e-5 * T / P (helium), where L in cm, T in K, P in torr.
    """
    if T_K <= 0 or P_torr <= 0:
        raise ValueError("T and P must be > 0")
    return 4.8e-5 * T_K / P_torr

def regime_from_L_over_a(L_cm: float, a_cm: float) -> Literal["viscous", "transition", "molecular"]:
    """
    Nunes: viscous when L/a < 0.01, molecular when L/a > 1, in between = transition.
    """
    x = L_cm / a_cm
    if x < 0.01:
        return "viscous"
    if x > 1.0:
        return "molecular"
    return "transition"

# -----------------------------
# Clausing / short-tube correction (Fig 2.13 + Eq 2.11 idea)
# -----------------------------

def transmission_probability_santeler(L_cm: float, a_cm: float) -> float:
    """
    Practical approximation for tube transmission probability (diffuse reflections).
    Used as a stand-in for digitizing Fig 2.13 (Clausing factor curves).
    """
    if L_cm <= 0 or a_cm <= 0:
        raise ValueError("L and a must be > 0")
    R = a_cm
    x = L_cm / R
    le_over_l = 1.0 + 1.0 / (3.0 + (3.0/7.0)*x)
    le_over_R = (L_cm * le_over_l) / R
    tau = 1.0 / (1.0 + (3.0/8.0)*le_over_R)
    return max(0.0, min(1.0, tau))

def clausing_K_approx(L_cm: float, a_cm: float) -> float:
    """
    Nunes uses K as Clausing's factor in Eq (2.11). For long tubes K ~ 8a/(3l).
    For diffuse transport, tau has long-tube limit ~ 4a/(3l), so K ~ 2*tau is consistent.
    """
    tau = transmission_probability_santeler(L_cm, a_cm)
    return min(1.0, 2.0 * tau)

def molecular_end_correction_factor(L_cm: float, a_cm: float) -> float:
    """
    Nunes Eq (2.11): F_actual = (3l/(8a)) * K * F_long.
    Here we convert that into a multiplicative factor applied to the long-tube coefficient.
    """
    K = clausing_K_approx(L_cm, a_cm)
    return (3.0 * L_cm / (8.0 * a_cm)) * K

# -----------------------------
# Core per-substep propagation
# -----------------------------

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
    """
    Nunes Eq (2.15): Qm = (4 a^3 / 3 l) sqrt(2π m / k) * ( P1/sqrt(T1) - P2/sqrt(T2) )
    Solve for P2.
    Pressures are converted to dyn/cm^2 internally.
    """
    if P1_torr <= 0:
        raise ValueError("P1 must be > 0")
    if any(x <= 0 for x in (T1_K, T2_K, a_cm, l_cm)):
        raise ValueError("T, a, l must be > 0")

    P1 = P1_torr * TORR_TO_DYNE_PER_CM2
    m = gas.molecule_mass_g

    coeff = (4.0 * (a_cm**3) / (3.0 * l_cm)) * math.sqrt(2.0 * math.pi * m / k_erg)
    if use_end_correction:
        coeff *= molecular_end_correction_factor(l_cm, a_cm)

    # Qm = coeff*(P1/sqrt(T1) - P2/sqrt(T2))
    P2 = math.sqrt(T2_K) * (P1 / math.sqrt(T1_K) - (Qm_g_s / coeff))
    return max(0.0, P2 / TORR_TO_DYNE_PER_CM2)

def step_transition_viscous_P2_torr(
    gas: Gas,
    P1_torr: float,
    T_K: float,
    a_cm: float,
    l_cm: float,
    Qm_g_s: float,
    eta_model: Literal["simple", "accurate"]="accurate",
) -> float:
    """
    Nunes Eq (2.14) (tractable approximation; apply on small substeps where T is ~const):
        P2 = sqrt((P1+Px)^2 - 2 η Ndot k Z T) - Px
    with:
        Z = 8 l / (π a^4)
        Px ≈ kT / (a d^2)   (Nunes Eq 2.13 gives a more specific prefactor)
    """
    if P1_torr <= 0:
        raise ValueError("P1 must be > 0")
    if any(x <= 0 for x in (T_K, a_cm, l_cm)):
        raise ValueError("T, a, l must be > 0")

    P1 = P1_torr * TORR_TO_DYNE_PER_CM2
    d = gas.d_cm
    eta = viscosity_kennard(gas, T_K, model=eta_model)  # g/(cm s)
    m = gas.molecule_mass_g
    Ndot = Qm_g_s / m  # molecules/s

    Z = 8.0 * l_cm / (math.pi * (a_cm**4))
    Px = (64.0 * math.sqrt(2.0) * k_erg * T_K) / (9.0 * (math.pi**2) * a_cm * (d**2))

    inside = (P1 + Px)**2 - 2.0 * eta * Ndot * k_erg * Z * T_K
    if inside <= 0:
        return 0.0
    P2 = math.sqrt(inside) - Px
    return max(0.0, P2 / TORR_TO_DYNE_PER_CM2)

# -----------------------------
# Segment network (Fig 2.15 style)
# -----------------------------

@dataclass
class Segment:
    name: str
    length_cm: float
    diameter_cm: float
    T_in_K: float
    T_out_K: float
    substeps: int = 20  # split for temperature gradient handling

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
    """
    Forward propagation from the cold end.
    Returns:
      pressures_torr: node pressures [P0, P1, ..., P_end]
      diagnostics: per-substep info (regime, L/a, etc.) for transparency.
    """
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
            # Temperatures for this small slice (linear profile):
            T1 = seg.T_in_K + (j / n) * (seg.T_out_K - seg.T_in_K)
            T2 = seg.T_in_K + ((j + 1) / n) * (seg.T_out_K - seg.T_in_K)
            Tmid = 0.5 * (T1 + T2)

            Lmfp = mean_free_path_helium_cm(Tmid, max(P, 1e-30))
            reg = regime_from_L_over_a(Lmfp, a)

            Pin = P
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

# -----------------------------
# Design constraint (Eqs 2.18–2.19)
# -----------------------------

def sum_li_over_ai3(segments: List[Segment]) -> float:
    """
    Σ l_i / a_i^3 in cm^-2, with l, a in cm.
    """
    s = 0.0
    for seg in segments:
        a = seg.a_cm()
        s += seg.length_cm / (a**3)
    return s

def nunes_constraint_rhs(P0_torr: float, T0_K: float, Qm_g_s: float, gas: Gas) -> float:
    """
    Nunes Eq (2.18) for 3He: P0/sqrt(T0) = 1.2 Qm Σ(l_i/a_i^3)
    => RHS (max allowed Σ) = P0 / (1.2 sqrt(T0) Qm)
    For 4He: multiply by sqrt(4/3) as the text notes.
    """
    if P0_torr <= 0 or T0_K <= 0 or Qm_g_s <= 0:
        raise ValueError("P0, T0, Qm must be > 0")

    c = 1.2 if gas.name == "He3" else 1.2 * math.sqrt(4.0/3.0)
    return P0_torr / (c * math.sqrt(T0_K) * Qm_g_s)

# -----------------------------
# Pump block (Eq 2.27 idea: q = p*S)
# -----------------------------

def throughput_torr_L_s_from_massflow(Qm_g_s: float, gas: Gas, T_K: float) -> float:
    """
    q = n_dot * R * T
    Using R = 62.3637 L·torr/(mol·K); n_dot (mol/s) = Qm/M.
    """
    R_L_torr = 62.3637
    n_dot = Qm_g_s / gas.molar_mass_g_mol
    return n_dot * R_L_torr * T_K

def pump_inlet_pressure_from_speed(Qm_g_s: float, gas: Gas, T_K: float, S_L_s: float) -> float:
    """
    q = p*S -> p = q/S
    """
    q = throughput_torr_L_s_from_massflow(Qm_g_s, gas, T_K)
    return q / max(S_L_s, 1e-30)
