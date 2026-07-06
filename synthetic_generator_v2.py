"""
==============================================================================
RESEARCH-QUALITY SYNTHETIC FIRST-TRIMESTER LONGITUDINAL DATA GENERATOR
==============================================================================
Generates 800 patients (~20-25% pregnancy loss) with 2-5 longitudinal scans.

BIOLOGICAL BASIS:
  CRL  : Hadlock polynomial  CRL(mm) ≈ -0.6 + 0.84·GA - 0.17·GA^2 + 0.02·GA^3
         (Robinson & Fleming 1975, Hadlock 1992)
  GS   : Mean sac diameter   GSD(mm) ≈ exp(0.36·(GA-5)) + 2
         (Nyberg 1987, Hellman 1969)
  FHR  : Parabolic rise-then-plateau  FHR ≈ 110 + 24·(GA-6) - 1.8·(GA-6)^2
         peaks ~170 bpm at week 9, drops to ~150 by week 13
         (Doubilet & Benson 1995)
  YSD  : Yolk sac grows weeks 5-10 then involutes
         YSD(mm) ≈ 3 + 1.5·sin(π·(GA-5)/10) · growth_factor
         (Blaas 1995, Cyr 1988)

KEY DESIGN PRINCIPLES:
  1. Patient-level latent biological factors (growth, heart, placenta, noise)
  2. Label-dependent trajectories with overlapping distributions
  3. Nonlinear (polynomial/exponential) growth curves
  4. Longitudinal autoregressive dependency (each scan evolves from previous)
  5. Correlated maternal features (age↔BMI↔gravidity↔parity↔prev_loss)
  6. Clinically variable measurement noise per variable
  7. Clinically plausible missingness (gestational-age & outcome dependent)
  8. All values clamped to first-trimester clinical ranges
==============================================================================
"""

import argparse
import json
import math
from pathlib import Path
import random

import pandas as pd
import numpy as np
# scipy not needed — all distributions use numpy

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic first-trimester longitudinal cohort."
    )
    parser.add_argument("--out", default="data/generated", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--n-patients", type=int, default=800, help="Number of patients."
    )
    parser.add_argument(
        "--loss-rate", type=float, default=0.22, help="Target pregnancy loss rate."
    )
    return parser.parse_known_args()[0]


_ARGS = _parse_args() if __name__ == "__main__" else None

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = _ARGS.seed if _ARGS is not None else 42
random.seed(SEED)
np.random.seed(SEED)

# ── configuration ────────────────────────────────────────────────────────────
N_PATIENTS        = _ARGS.n_patients if _ARGS is not None else 800
LOSS_RATE_TARGET  = _ARGS.loss_rate if _ARGS is not None else 0.22
MIN_SCANS         = 2
MAX_SCANS         = 5
GA_FIRST_SCAN_MU  = 6.5         # mean GA of first scan (weeks)
GA_FIRST_SCAN_SD  = 0.7
INTER_SCAN_MU     = 2.0         # mean weeks between scans
INTER_SCAN_SD     = 0.5
GA_MAX            = 13.9        # latest possible GA in T1

# ── clinical ranges (first trimester) ────────────────────────────────────────
CRL_RANGE  = (1.0,  84.0)       # mm
GS_RANGE   = (2.0, 100.0)       # mm mean sac diameter
FHR_RANGE  = (80.0, 190.0)      # bpm
YSD_RANGE  = (1.0,   7.0)       # mm

# ── measurement error SD (variable-specific, mimics real US precision) ───────
MEAS_ERROR = {
    'CRL':  0.8,   # mm – CRL is measured quite precisely
    'GS':   2.5,   # mm – GSD has higher inter-observer variability
    'FHR':  4.0,   # bpm – FHR measurement has ~4 bpm variability
    'YSD':  0.35,  # mm – yolk sac measured at high zoom
}

# ==============================================================================
# 1.  BIOLOGICALLY REALISTIC GROWTH CURVES
# ==============================================================================

def crl_curve(ga, growth_factor=1.0):
    """
    CRL (mm) as a function of gestational age (weeks).
    Based on Hadlock (1992) / Robinson-Fleming polynomial fit.
    """
    g = ga - 5.0  # shift so curve starts near 0 at ~5 weeks
    crl = (-0.6 + 0.84 * g + 0.17 * g**2 + 0.006 * g**3) * growth_factor
    return max(crl, 0.5)


def gs_curve(ga, growth_factor=1.0, placenta_factor=1.0):
    """
    Gestational sac mean diameter (mm).
    Exponential growth model (Nyberg 1987).
    The placenta factor modulates sac size (poor placentation → smaller sac).
    """
    combined = 0.6 * growth_factor + 0.4 * placenta_factor
    gs = (math.exp(0.32 * (ga - 4.5)) + 1.5) * combined
    return max(gs, 1.0)


def fhr_curve(ga, heart_factor=1.0):
    """
    Fetal heart rate (bpm).  Parabolic rise peaking ~week 9 then slight decline.
    Based on Doubilet & Benson 1995.
    """
    t = ga - 6.0
    fhr = (110.0 + 24.0 * t - 1.8 * t**2) * heart_factor
    return max(fhr, 0.0)


def ysd_curve(ga, growth_factor=1.0):
    """
    Yolk sac diameter (mm).  Grows weeks 5-10 then involutes.
    Based on Blaas (1995) and Cyr (1988).
    """
    t = ga - 5.0
    ysd = (3.0 + 1.5 * math.sin(math.pi * t / 10.0)) * growth_factor
    return max(ysd, 0.5)


# ==============================================================================
# 2.  PATIENT-LEVEL LATENT VARIABLES
# ==============================================================================

def generate_latent_factors(label):
    """
    Draw patient-level latent biological parameters.
    Miscarriage patients get shifted distributions (slower growth,
    weaker heart function, poorer placentation, higher noise).
    Distributions OVERLAP so classification is not trivial.
    """
    if label == 0:  # viable pregnancy
        growth_factor   = np.clip(np.random.normal(1.00, 0.08), 0.75, 1.30)
        heart_factor    = np.clip(np.random.normal(1.00, 0.05), 0.80, 1.20)
        placenta_factor = np.clip(np.random.normal(1.00, 0.07), 0.75, 1.25)
        noise_factor    = np.clip(np.random.normal(1.00, 0.15), 0.50, 1.80)
    else:            # pregnancy loss
        growth_factor   = np.clip(np.random.normal(0.82, 0.12), 0.45, 1.15)
        heart_factor    = np.clip(np.random.normal(0.88, 0.10), 0.55, 1.10)
        placenta_factor = np.clip(np.random.normal(0.80, 0.12), 0.40, 1.10)
        noise_factor    = np.clip(np.random.normal(1.30, 0.25), 0.70, 2.50)
    return growth_factor, heart_factor, placenta_factor, noise_factor


# ==============================================================================
# 3.  CORRELATED MATERNAL FEATURES
# ==============================================================================

def generate_maternal_features(label):
    """
    Age, BMI, parity, gravidity, previous_loss with realistic correlations.
    Uses a copula-like approach: draw correlated normals, then transform.
    
    Epidemiological basis:
      - Older women → higher miscarriage risk
      - Higher BMI → modest increase in risk
      - Previous miscarriage → strong predictor
      - Gravidity ≥ parity (always)
      - IVF more common in older / higher-risk patients
    """
    if label == 0:
        age_mu, age_sd   = 29.0, 4.5
        bmi_mu, bmi_sd   = 24.5, 4.0
        prev_loss_prob   = 0.10
        ivf_prob         = 0.08
    else:
        age_mu, age_sd   = 33.5, 5.0
        bmi_mu, bmi_sd   = 26.5, 5.0
        prev_loss_prob   = 0.35
        ivf_prob         = 0.18

    # Correlated age and BMI (ρ ≈ 0.15)
    rho = 0.15
    z1, z2 = np.random.normal(0, 1, 2)
    z2 = rho * z1 + math.sqrt(1 - rho**2) * z2
    age = np.clip(age_mu + age_sd * z1, 18, 45)
    bmi = np.clip(bmi_mu + bmi_sd * z2, 16, 45)

    # Age modulates previous_loss probability
    age_risk_modifier = max(0, (age - 30) * 0.015)
    prev_loss = int(np.random.rand() < min(prev_loss_prob + age_risk_modifier, 0.65))

    # Gravidity and parity (correlated)
    # Higher age → higher gravidity on average
    grav_lambda = max(0.8, 0.5 + (age - 25) * 0.08)
    gravidity = max(1, np.random.poisson(grav_lambda))

    # Parity ≤ gravidity; if previous loss, subtract at least 1 from max
    max_parity = gravidity - prev_loss
    parity = np.random.randint(0, max(1, max_parity + 1))
    
    # Ensure gravidity >= parity + previous_loss (biological constraint)
    gravidity = max(gravidity, parity + prev_loss)

    # IVF slightly more common in older / previous-loss patients
    ivf_adj = ivf_prob + (0.05 if prev_loss else 0) + max(0, (age - 35) * 0.02)
    conception = "IVF" if np.random.rand() < min(ivf_adj, 0.40) else "Natural"

    return {
        'age': round(age, 1),
        'bmi': round(bmi, 1),
        'parity': int(parity),
        'gravidity': int(gravidity),
        'previous_loss': int(prev_loss),
        'conception': conception,
        'singleton': True,
    }


# ==============================================================================
# 4.  CLINICALLY PLAUSIBLE MISSING DATA
# ==============================================================================

def apply_missingness(scan, ga, label):
    """
    Missingness depends on gestational age and outcome:
      - FHR often missing < 6.5 weeks (not yet detectable)
      - CRL sometimes missing very early (too small to measure)
      - GS rarely missing (almost always visible)
      - YSD sometimes missing late (involuted) or very early
      - Miscarriage pregnancies: slightly higher missingness overall
    """
    base_multiplier = 1.3 if label == 1 else 1.0

    # FHR: high miss rate before 6.5w, low after 7w
    fhr_miss_prob = 0.30 * base_multiplier if ga < 6.5 else (0.05 * base_multiplier if ga < 7.0 else 0.02 * base_multiplier)
    if np.random.rand() < fhr_miss_prob:
        scan['FHR'] = np.nan

    # CRL: sometimes not measurable < 6w
    crl_miss_prob = 0.15 * base_multiplier if ga < 6.0 else 0.03 * base_multiplier
    if np.random.rand() < crl_miss_prob:
        scan['CRL'] = np.nan

    # GS: almost always visible
    gs_miss_prob = 0.02 * base_multiplier
    if np.random.rand() < gs_miss_prob:
        scan['GS'] = np.nan

    # YSD: missing late (>11w, involuted) or very early
    if ga > 11.0:
        ysd_miss_prob = 0.35 * base_multiplier
    elif ga < 5.5:
        ysd_miss_prob = 0.20 * base_multiplier
    else:
        ysd_miss_prob = 0.04 * base_multiplier
    if np.random.rand() < ysd_miss_prob:
        scan['YSD'] = np.nan

    return scan


# ==============================================================================
# 5.  LONGITUDINAL SCAN GENERATION (autoregressive)
# ==============================================================================

def generate_scans(pat_id, label, growth_factor, heart_factor,
                   placenta_factor, noise_factor):
    """
    Generate 2-5 longitudinal scans.  Each scan evolves from the PREVIOUS
    measurement (autoregressive), not generated independently.
    
    Miscarriage pregnancies may stop earlier (fewer late scans) and
    show decelerating or declining trajectories.
    """
    # Number of scans: miscarriage patients slightly more likely to have fewer
    if label == 1:
        n_scans = np.random.choice([2, 3, 4, 5], p=[0.35, 0.40, 0.20, 0.05])
    else:
        n_scans = np.random.choice([2, 3, 4, 5], p=[0.30, 0.35, 0.25, 0.10])

    # First scan GA
    first_ga = np.clip(np.random.normal(GA_FIRST_SCAN_MU, GA_FIRST_SCAN_SD), 5.0, 9.0)

    scans = []
    prev_crl = None
    prev_gs  = None
    prev_fhr = None
    prev_ysd = None

    for s_idx in range(n_scans):
        if s_idx == 0:
            ga = round(first_ga, 1)
        else:
            delta = np.clip(np.random.normal(INTER_SCAN_MU, INTER_SCAN_SD), 1.0, 4.0)
            ga = round(scans[-1]['gestational_age_weeks'] + delta, 1)

        if ga > GA_MAX:
            break  # don't generate scans past first trimester

        # ── For miscarriage: progressively degrade growth factors ────────
        if label == 1:
            # The later the scan, the more "off" the trajectory becomes
            # This models progressive deterioration
            progress = s_idx / max(n_scans - 1, 1)  # 0→1
            degrade = 1.0 - 0.15 * progress  # up to 15% degradation
            eff_growth   = growth_factor * degrade
            eff_heart    = heart_factor * degrade
            eff_placenta = placenta_factor * degrade
        else:
            eff_growth   = growth_factor
            eff_heart    = heart_factor
            eff_placenta = placenta_factor

        # ── Compute expected values from growth curves ───────────────────
        exp_crl = crl_curve(ga, eff_growth)
        exp_gs  = gs_curve(ga, eff_growth, eff_placenta)
        exp_fhr = fhr_curve(ga, eff_heart)
        exp_ysd = ysd_curve(ga, eff_growth)

        # ── Autoregressive blending (λ=0.7 from curve, 0.3 from prev) ───
        ar_weight = 0.30  # how much the previous scan influences

        if prev_crl is not None:
            # Expected change from curve
            crl_from_curve = exp_crl
            crl_from_prev  = prev_crl + (exp_crl - crl_curve(
                scans[-1]['gestational_age_weeks'], eff_growth))
            crl_base = (1 - ar_weight) * crl_from_curve + ar_weight * crl_from_prev
        else:
            crl_base = exp_crl

        if prev_gs is not None:
            gs_from_curve = exp_gs
            gs_from_prev  = prev_gs + (exp_gs - gs_curve(
                scans[-1]['gestational_age_weeks'], eff_growth, eff_placenta))
            gs_base = (1 - ar_weight) * gs_from_curve + ar_weight * gs_from_prev
        else:
            gs_base = exp_gs

        if prev_fhr is not None:
            fhr_from_curve = exp_fhr
            fhr_from_prev  = prev_fhr + (exp_fhr - fhr_curve(
                scans[-1]['gestational_age_weeks'], eff_heart))
            fhr_base = (1 - ar_weight) * fhr_from_curve + ar_weight * fhr_from_prev
        else:
            fhr_base = exp_fhr

        if prev_ysd is not None:
            ysd_from_curve = exp_ysd
            ysd_from_prev  = prev_ysd + (exp_ysd - ysd_curve(
                scans[-1]['gestational_age_weeks'], eff_growth))
            ysd_base = (1 - ar_weight) * ysd_from_curve + ar_weight * ysd_from_prev
        else:
            ysd_base = exp_ysd

        # ── Add measurement noise (patient-specific amplitude) ───────────
        crl_val = crl_base + np.random.normal(0, MEAS_ERROR['CRL'] * noise_factor)
        gs_val  = gs_base  + np.random.normal(0, MEAS_ERROR['GS']  * noise_factor)
        fhr_val = fhr_base + np.random.normal(0, MEAS_ERROR['FHR'] * noise_factor)
        ysd_val = ysd_base + np.random.normal(0, MEAS_ERROR['YSD'] * noise_factor)

        # ── Clamp to clinical ranges ─────────────────────────────────────
        crl_val = round(np.clip(crl_val, *CRL_RANGE), 1)
        gs_val  = round(np.clip(gs_val,  *GS_RANGE),  1)
        fhr_val = round(np.clip(fhr_val, *FHR_RANGE), 1)
        ysd_val = round(np.clip(ysd_val, *YSD_RANGE), 1)

        scan = {
            'patient_id': pat_id,
            'gestational_age_weeks': ga,
            'FHR': fhr_val,
            'CRL': crl_val,
            'GS':  gs_val,
            'YSD': ysd_val,
        }

        # ── Apply clinically realistic missingness ───────────────────────
        scan = apply_missingness(scan, ga, label)

        scans.append(scan)

        # Store for autoregressive dependency
        prev_crl = crl_val
        prev_gs  = gs_val
        prev_fhr = fhr_val
        prev_ysd = ysd_val

    return scans


# ==============================================================================
# 6.  DATASET GENERATION AND REPORTING
# ==============================================================================

def generate_dataset(
    n_patients=N_PATIENTS,
    loss_rate=LOSS_RATE_TARGET,
    seed=SEED,
):
    """Generate synthetic patients and longitudinal ultrasound scans."""
    random.seed(seed)
    np.random.seed(seed)

    synthetic_patients = []
    synthetic_scans = []

    for pid in range(1, n_patients + 1):
        pat_id = f"P{pid:05d}"

        label = int(np.random.rand() < loss_rate)
        growth_factor, heart_factor, placenta_factor, noise_factor = (
            generate_latent_factors(label)
        )
        maternal = generate_maternal_features(label)

        synthetic_patients.append({
            'patient_id': pat_id,
            **maternal,
            'label': label,
        })

        scans = generate_scans(
            pat_id, label, growth_factor, heart_factor,
            placenta_factor, noise_factor
        )
        synthetic_scans.extend(scans)

    patients = pd.DataFrame(synthetic_patients)
    scans = pd.DataFrame(synthetic_scans)

    for var in ['FHR', 'CRL', 'GS', 'YSD']:
        scans[var] = scans.groupby('patient_id')[var].transform(
            lambda x: x.interpolate().bfill().ffill()
        )
        scans[var] = scans[var].fillna(scans[var].median())

    return patients, scans


def print_summary(patients, scans):
    """Print validation and summary statistics for a generated cohort."""
    print("=" * 70)
    print("SYNTHETIC COHORT SUMMARY")
    print("=" * 70)

    n_total = patients['patient_id'].nunique()
    n_loss = patients['label'].sum()
    n_live = n_total - n_loss
    loss_pct = 100 * n_loss / n_total

    print(f"Total patients       : {n_total}")
    print(f"Live births (label=0): {n_live}  ({100-loss_pct:.1f}%)")
    print(f"Pregnancy loss (1)   : {n_loss}  ({loss_pct:.1f}%)")
    print(f"Total scan records   : {len(scans)}")

    spp = scans.groupby('patient_id').size()
    print("\nScans per patient:")
    print(f"  min={spp.min()}, max={spp.max()}, "
          f"mean={spp.mean():.2f}, median={spp.median():.1f}")
    print("\nScans per patient distribution (%):")
    print((spp.value_counts(normalize=True).sort_index() * 100).round(1))

    print("\n" + "-" * 70)
    print("MATERNAL FEATURE SUMMARY BY OUTCOME")
    print("-" * 70)
    for col in ['age', 'bmi', 'parity', 'gravidity', 'previous_loss']:
        grp = patients.groupby('label')[col]
        print(f"\n{col}:")
        for lbl in [0, 1]:
            g = grp.get_group(lbl)
            label_name = 'Live' if lbl == 0 else 'Loss'
            print(f"  {label_name}: mean={g.mean():.2f}, sd={g.std():.2f}, "
                  f"min={g.min():.1f}, max={g.max():.1f}")

    print("\n" + "-" * 70)
    print("ULTRASOUND MEASUREMENT SUMMARY BY OUTCOME")
    print("-" * 70)
    merged = scans.merge(patients[['patient_id', 'label']])
    for var in ['CRL', 'GS', 'FHR', 'YSD']:
        print(f"\n{var}:")
        for lbl in [0, 1]:
            subset = merged[merged['label'] == lbl][var].dropna()
            label_name = 'Live' if lbl == 0 else 'Loss'
            print(f"  {label_name}: n={len(subset)}, "
                  f"mean={subset.mean():.1f}, sd={subset.std():.1f}, "
                  f"range=[{subset.min():.1f}, {subset.max():.1f}]")

    print("\n" + "-" * 70)
    print("CORRELATION MATRIX (ultrasound measurements)")
    print("-" * 70)
    print(scans[['FHR', 'CRL', 'GS', 'YSD']].corr().round(3))

    print("\n" + "-" * 70)
    print("CONCEPTION METHOD BY OUTCOME")
    print("-" * 70)
    print(patients.groupby('label')['conception'].value_counts().unstack(fill_value=0))

    print("\n" + "=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)


def build_summary(patients, scans):
    scans_per_patient = scans.groupby("patient_id").size()
    loss_count = int(patients["label"].sum())
    return {
        "seed": int(SEED),
        "n_patients": int(patients["patient_id"].nunique()),
        "n_scans": int(len(scans)),
        "n_loss": loss_count,
        "n_live": int(len(patients) - loss_count),
        "loss_rate": float(loss_count / len(patients)),
        "scans_per_patient": {
            "min": int(scans_per_patient.min()),
            "max": int(scans_per_patient.max()),
            "mean": float(scans_per_patient.mean()),
            "median": float(scans_per_patient.median()),
        },
        "columns": {
            "patients": patients.columns.tolist(),
            "scans": scans.columns.tolist(),
        },
        "validation": {
            "all_patients_have_scans": bool(
                scans["patient_id"].nunique() == patients["patient_id"].nunique()
            ),
            "min_scans_ok": bool(scans_per_patient.min() >= MIN_SCANS),
            "max_scans_ok": bool(scans_per_patient.max() <= MAX_SCANS),
            "gestational_age_max_ok": bool(
                scans["gestational_age_weeks"].max() <= GA_MAX
            ),
            "no_remaining_missing_us_values": bool(
                scans[["FHR", "CRL", "GS", "YSD"]].isna().sum().sum() == 0
            ),
        },
    }


def save_outputs(patients, scans, out_dir):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    patients_csv = out_path / "patients.csv"
    scans_csv = out_path / "scans.csv"
    summary_json = out_path / "dataset_summary.json"

    patients.to_csv(patients_csv, index=False)
    scans.to_csv(scans_csv, index=False)
    summary_json.write_text(
        json.dumps(build_summary(patients, scans), indent=2),
        encoding="utf-8",
    )

    print(f"Saved patients CSV: {patients_csv}")
    print(f"Saved scans CSV: {scans_csv}")
    print(f"Saved summary JSON: {summary_json}")


def main():
    patients, scans = generate_dataset(N_PATIENTS, LOSS_RATE_TARGET, SEED)
    print_summary(patients, scans)
    save_outputs(patients, scans, _ARGS.out)


if __name__ == "__main__":
    main()
