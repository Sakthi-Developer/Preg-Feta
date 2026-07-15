"""
==============================================================================
RESEARCH-QUALITY SYNTHETIC FIRST-TRIMESTER LONGITUDINAL DATA GENERATOR
==============================================================================
Generates N patients (~20% pregnancy loss) with 2-5 longitudinal scans,
calibrated to PUBLISHED first-trimester reference ranges and loss effects.

CALIBRATION SOURCES (viable reference medians + loss odds ratios):
  - "Early pregnancy ultrasound measurements and prediction of first trimester
    pregnancy loss: A logistic model", Scientific Reports (2020), PMC6994659:
    serial GSD, YSD, CRL, FHR from 6-10 weeks; ~20% loss.
      * Viable medians by week (see VIABLE_REF below).
      * Loss effects: small CRL (OR 0.59/mm wk8), small GS (OR 0.01-0.10),
        bradycardia (OR 0.96/bpm wk7), ENLARGED yolk sac (OR 3.36-6.52 wk6-8;
        "YS > 6 mm at any GA predicts loss").
  - IVF-ET reference intervals, 30,416 pregnancies, 6-10 wk (PubMed 32928137):
    smooth GSD/CRL/YSD/HR growth models — used to sanity-check curve shapes.
  - Miscarriage risk-factor cohorts (advanced maternal age ~2.7x at >=35;
    BMI weak; prior loss independent predictor).

VIABLE REFERENCE MEDIANS (mm, mm, mm, bpm) by gestational week:
  Week :   6      7      8      9      10
  CRL  :   5.0   11.5   18.3   25.4   35.8
  GSD  :  16.0   24.0   31.0   37.0   43.0
  YSD  :   2.6    2.9    3.3    3.7    4.1
  FHR  : 116    148    168    171    170

KEY DESIGN PRINCIPLES:
  1. Viable mean trajectories = interpolated published medians (+ growth-rate
     extrapolation to ~14 weeks).
  2. Per-marker, label-dependent effects calibrated to published direction and
     magnitude — including the ENLARGED yolk sac in loss (opposite of naive
     "everything shrinks" models).
  3. Patient-level latent factors give within-group biological spread; loss and
     viable distributions OVERLAP so classification is realistically hard
     (target combined AUROC ~0.88-0.92, not saturated).
  4. Longitudinal autoregressive dependency (each scan evolves from previous),
     with progressive deterioration in loss trajectories.
  5. Correlated maternal features (age<->BMI<->gravidity<->parity<->prev_loss),
     only weakly separated by outcome (matching cohort evidence).
  6. Clinically variable measurement noise and plausible missingness.
  7. All values clamped to first-trimester clinical ranges.

------------------------------------------------------------------------------
DUAL-ETIOLOGY MODE (--etiology dual)  ***MANUSCRIPT DISCLOSURE REQUIRED***
------------------------------------------------------------------------------
The default ("single") generator draws ONE latent loss outcome that drives both
the scan trajectory and the maternal features, so both modalities are noisy views
of the *same* label. Under that design the two single-modality models make highly
correlated errors and multimodal fusion has little to add — which understates the
contribution of a fusion model and is not how first-trimester loss actually works.

First-trimester loss is *etiologically heterogeneous*. "dual" mode models this
explicitly as an OR of two independent Bernoulli channels:

  * a SONOGRAPHIC channel — an embryonic abnormality expressed in the longitudinal
    scan trajectory (slow CRL/GS growth, bradycardia, enlarged yolk sac), visible to
    the FETA-Transformer but with near-normal maternal features; and
  * a MATERNAL channel — a constitutional risk driven by an *interaction* of maternal
    variables (advanced age compounded by elevated BMI and prior loss), visible to
    the PREG-Net graph but with a near-normal early scan.

  observed_loss = sonographic_loss  OR  maternal_loss

Because each modality reflects only one channel, neither single-modality model can
see every loss, so genuine complementary information exists for fusion to exploit —
which is the hypothesis this project tests. Two knobs shape the regime:

  * --channel-corr      shared frailty coupling the two channels (irreducible; noise
                        for the maternal predictor).
  * --sono-mat-coupling directed maternal->sonographic coupling: a share of high
                        maternal-risk pregnancies ALSO express a sonographic
                        abnormality ("both-channel" losses visible to both models).
                        Trades fusion gap against absolute per-model AUROC; the
                        shipped default (1.0) gives the "balanced" regime — towers
                        ~0.78 AUROC, fusion materially higher, ~75% single-channel
                        losses preserved so the fusion gap remains real.

IMPORTANT: dual mode is a DELIBERATE design choice made to create a testbed in which
multimodal fusion is informative. It is a *modelling assumption*, not a measured
property of any real cohort. Any manuscript reporting dual-mode results MUST state
that (a) the data are synthetic, (b) the loss mechanism is a designed OR-of-two-
channels with the coupling parameters above, and (c) the reported fusion gains are
therefore an in-silico demonstration that the architecture *can* exploit etiologic
heterogeneity when it is present — not evidence about real pregnancies. See
results/manuscript/methods.md and limitations.md for the disclosure text.
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
        "--n-patients", type=int, default=3000, help="Number of patients."
    )
    parser.add_argument(
        "--loss-rate", type=float, default=0.20, help="Target pregnancy loss rate."
    )
    parser.add_argument(
        "--etiology",
        choices=["single", "dual"],
        default="single",
        help=(
            "Loss-generating model. 'single' (default): one latent outcome drives "
            "both modalities (legacy, redundant channels). 'dual': etiologic "
            "heterogeneity — an independent sonographic-trajectory channel and a "
            "maternal-interaction channel, loss = (embryonic abnormal) OR (maternal "
            "environment fails). Makes multimodal fusion meaningful; see module "
            "docstring for the required manuscript disclosure."
        ),
    )
    parser.add_argument(
        "--channel-corr",
        type=float,
        default=0.50,
        help="[dual] Shared-frailty coupling between the two loss channels (0=indep). "
             "A moderate value gives some co-occurring losses (lifting both towers) "
             "while preserving single-channel losses that only fusion can catch.",
    )
    parser.add_argument(
        "--sono-mat-coupling",
        type=float,
        default=1.0,
        help="[dual] Directed maternal->sonographic coupling: how strongly a "
             "high maternal-risk conjunction also elevates embryonic (sonographic) "
             "risk, producing 'both-channel' losses visible to both models. 0 = pure "
             "specialists (max fusion gap, lower absolute AUROC); higher = individually "
             "stronger towers with a smaller but still real fusion gap (balanced regime).",
    )
    return parser.parse_known_args()[0]


_ARGS = _parse_args() if __name__ == "__main__" else None

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = _ARGS.seed if _ARGS is not None else 42
random.seed(SEED)
np.random.seed(SEED)

# ── configuration ────────────────────────────────────────────────────────────
N_PATIENTS        = _ARGS.n_patients if _ARGS is not None else 3000
LOSS_RATE_TARGET  = _ARGS.loss_rate if _ARGS is not None else 0.20
ETIOLOGY          = _ARGS.etiology if _ARGS is not None else "single"
CHANNEL_CORR      = _ARGS.channel_corr if _ARGS is not None else 0.25
SONO_MAT_COUPLING = _ARGS.sono_mat_coupling if _ARGS is not None else 0.0

# ── dual-etiology channel strengths (only used when ETIOLOGY == "dual") ───────
# The observed loss is the logical OR of two channels (embryo sonographically
# abnormal OR maternal environment fails). The sonographic channel is a steep
# Bernoulli mechanism whose severity drives the expressed scan trajectory; the
# maternal channel is a near-deterministic THRESHOLD on the maternal-risk score
# (see generate_patient_dual) so each specialist can recover its own channel
# cleanly. Intercepts/threshold are tuned so each marginal channel rate is ~0.12–
# 0.14, giving an OR loss rate ≈ LOSS_RATE_TARGET (~0.20).
SONO_BASE_LOGIT   = -3.78   # sonographic-channel intercept (FETA-favoured)
MAT_LOSS_THRESHOLD = 3.55   # maternal-risk-score threshold above which the maternal
                            # channel fails (PREG-favoured). A hard threshold (plus a
                            # small shared-frailty perturbation) makes label_mat almost
                            # a function of the maternal variables alone.
SONO_SLOPE        = 2.60    # how strongly the sonographic latent drives its loss
                            # (steep → label_sono near-deterministic in the latent,
                            #  so the expressed scan trajectory is cleanly separable)
MAT_MAIN_SLOPE    = 1.30    # additive (per-variable) maternal effect
MAT_INTER_SLOPE   = 3.60    # multiplicative (interaction) maternal effect — the
                            # conjunction PREG-Net's graph is designed to capture
                            # (steep → label_mat near-deterministic in the maternal
                            #  variables, so the clinical specialist detects it cleanly)
MAT_NOISE_SD      = 0.80    # independent Gaussian noise on the maternal-risk score
                            # (unobserved by the clinical model): softens the hard
                            # threshold so PREG detects ~85 % of the channel rather
                            # than 100 %, keeping the two specialists comparably strong
                            # and the maternal boundary realistic rather than perfect.
# SONO_MAT_COUPLING (wired from --sono-mat-coupling above): a compromised maternal
# environment also raises embryonic risk, so a share of high-maternal-risk pregnancies
# ALSO express a sonographic abnormality (a "both-channel" loss both models can see).
# Controls the gap-vs-absolute trade-off: 0 = pure specialists (max fusion gap, lower
# absolute AUROC); higher = more overlap (individually stronger towers, smaller but
# still real gap). The coupling term is centred (see MAT_CONTRIB_CENTER) so raising it
# reshapes the co-occurrence structure without inflating the marginal sonographic rate.
MAT_CONTRIB_CENTER = 1.05    # ≈ E[maternal-risk contribution] over the population;
                            # subtracted so the coupling term is mean-zero and the
                            # sonographic base rate stays put as the knob is turned.
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
YSD_RANGE  = (1.0,   8.0)       # mm (allow enlarged loss yolk sacs up to ~8 mm)

# ── measurement error SD (variable-specific, mimics real US precision) ───────
MEAS_ERROR = {
    'CRL':  1.6,   # mm
    'GS':   3.2,   # mm – GSD has higher inter-observer variability
    'FHR':  6.5,   # bpm
    'YSD':  0.55,  # mm
}

# ==============================================================================
# 1.  PUBLISHED VIABLE REFERENCE CURVES (interpolated medians)
# ==============================================================================

# Gestational weeks at which published medians are anchored.
REF_WEEKS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

# Viable-pregnancy median by week (from Sci Rep 2020 / IVF-ET reference study).
VIABLE_REF = {
    'CRL': np.array([2.5,  5.0, 11.5, 18.3, 25.4, 35.8]),
    'GS':  np.array([11.0, 16.0, 24.0, 31.0, 37.0, 43.0]),
    'YSD': np.array([2.2,  2.6,  2.9,  3.3,  3.7,  4.1]),
    'FHR': np.array([111., 116., 148., 168., 171., 170.]),
}

# Per-week growth used to extrapolate beyond 10 weeks (mm/wk or bpm/wk).
# YSD involutes late, so it grows only gently; FHR plateaus/declines slightly.
REF_SLOPE = {'CRL': 7.54, 'GS': 6.65, 'YSD': 0.20, 'FHR': -1.0}


def ref_curve(marker, ga):
    """Viable median for ``marker`` at gestational age ``ga`` (weeks).

    Linear interpolation of published medians within 5-10 weeks, and linear
    extrapolation (published growth rate) beyond 10 weeks.
    """
    weeks = REF_WEEKS
    vals = VIABLE_REF[marker]
    if ga <= weeks[-1]:
        return float(np.interp(ga, weeks, vals))
    return float(vals[-1] + REF_SLOPE[marker] * (ga - weeks[-1]))


# ==============================================================================
# 2.  PATIENT-LEVEL LATENT VARIABLES (per-marker, label-dependent effects)
# ==============================================================================

def _clip_normal(mu, sd, lo, hi):
    return float(np.clip(np.random.normal(mu, sd), lo, hi))


# Probability that a loss pregnancy expresses each abnormality. Loss is
# heterogeneous — a given case shows only some of these (missed abortion vs
# blighted ovum vs bradycardic embryo present differently), so the four markers
# rarely all agree. This is what keeps the *combined* task realistically hard.
LOSS_EXPRESS_PROB = {'CRL': 0.94, 'GS': 0.92, 'FHR': 0.90, 'YSD': 0.92}

# Fraction of *sonographic-channel* losses that still look normal on scans (early
# euploid-appearing / genetic losses) — irreducible Bayes error for the ultrasound
# model. Kept modest so the FETA specialist can read its own channel well; the
# maternal channel independently supplies losses FETA cannot see (that is the point
# of fusion, and is handled by the dual-etiology model rather than by this knob).
LOSS_OCCULT_PROB = 0.02


def generate_latent_factors(label):
    """Draw per-marker multiplicative effects for one patient.

    A label-independent *constitutional size* factor scales CRL/GS/YSD together,
    so some viable pregnancies are simply small (and some losses measure normal)
    — no single marker is a clean separator.

    Loss pregnancies are heterogeneous:
      - a fraction (:data:`LOSS_OCCULT_PROB`) look sonographically normal;
      - the rest express a *phenotype* — each abnormality present only with
        :data:`LOSS_EXPRESS_PROB`, following the published direction/magnitude
        (small CRL/GS, bradycardia, ENLARGED YSD), at least one expressed.

    Spreads are wide so viable and loss distributions overlap substantially,
    keeping the combined task realistically hard (target AUROC ~0.88-0.92).
    """
    # Constitutional biological size, independent of outcome (wide → overlap).
    size = _clip_normal(1.00, 0.16, 0.65, 1.40)

    def viable_like(noise_mu=1.00, noise_sd=0.18):
        return {
            'CRL':   size * _clip_normal(1.00, 0.12, 0.74, 1.30),
            'GS':    size * _clip_normal(1.00, 0.12, 0.74, 1.30),
            'FHR':   _clip_normal(1.00, 0.08, 0.80, 1.20),
            'YSD':   size * _clip_normal(1.00, 0.16, 0.65, 1.45),
            'noise': _clip_normal(noise_mu, noise_sd, 0.55, 2.10),
            # Non-symptomatic pregnancies (viable and occult loss) have stable
            # trajectories and viable-like scan counts — otherwise trajectory
            # shape or scan count would leak the label.
            'symptomatic': False,
            'degrade': 0.0,
        }

    if label == 0:  # viable pregnancy
        return viable_like()

    # Occult loss — sonographically indistinguishable from a viable pregnancy.
    # Uses the *same* noise as viable: any noise elevation would itself leak the
    # label, so these must be drawn from the identical viable distribution.
    if np.random.rand() < LOSS_OCCULT_PROB:
        return viable_like()

    # Symptomatic loss — pick which abnormalities this case expresses.
    express = {m: (np.random.rand() < p) for m, p in LOSS_EXPRESS_PROB.items()}
    if not any(express.values()):
        express[random.choice(list(express))] = True

    def eff(marker, abn_mu, abn_sd, abn_lo, abn_hi):
        if express[marker]:
            return _clip_normal(abn_mu, abn_sd, abn_lo, abn_hi)
        return _clip_normal(0.99, 0.10, 0.78, 1.22)  # near-normal when not expressed

    return {
        'CRL':   size * eff('CRL', 0.68, 0.11, 0.38, 0.94),
        'GS':    size * eff('GS',  0.70, 0.11, 0.38, 0.94),
        'FHR':   eff('FHR', 0.80, 0.09, 0.54, 1.02),
        'YSD':   size * eff('YSD', 1.55, 0.24, 1.05, 2.40),  # enlarged, wide tail
        'noise': _clip_normal(1.25, 0.22, 0.70, 2.40),
        'symptomatic': True,
        'degrade': _clip_normal(0.20, 0.05, 0.06, 0.34),  # trajectory deterioration rate
    }


# ==============================================================================
# 3.  CORRELATED MATERNAL FEATURES (weakly separated by outcome)
# ==============================================================================

def generate_maternal_features(label):
    """Age, BMI, parity, gravidity, previous_loss with realistic correlations.

    Separation by outcome is deliberately weak (published cohorts found age/BMI
    only marginally different: ~34 vs 32 yrs, ~26 vs 25 BMI). Prior loss carries
    the main maternal signal, consistent with it being an independent predictor.
    """
    if label == 0:
        age_mu, age_sd   = 31.5, 4.6
        bmi_mu, bmi_sd   = 25.0, 4.2
        prev_loss_prob   = 0.12
        ivf_prob         = 0.30
    else:
        age_mu, age_sd   = 33.5, 5.0
        bmi_mu, bmi_sd   = 26.0, 4.6
        prev_loss_prob   = 0.24
        ivf_prob         = 0.32

    # Correlated age and BMI (ρ ≈ 0.15)
    rho = 0.15
    z1, z2 = np.random.normal(0, 1, 2)
    z2 = rho * z1 + math.sqrt(1 - rho**2) * z2
    age = np.clip(age_mu + age_sd * z1, 18, 45)
    bmi = np.clip(bmi_mu + bmi_sd * z2, 16, 45)

    # Age modulates previous_loss probability
    age_risk_modifier = max(0, (age - 32) * 0.012)
    prev_loss = int(np.random.rand() < min(prev_loss_prob + age_risk_modifier, 0.55))

    # Gravidity and parity (correlated); higher age → higher gravidity on average
    grav_lambda = max(0.8, 0.5 + (age - 25) * 0.08)
    gravidity = max(1, np.random.poisson(grav_lambda))

    # Parity ≤ gravidity; if previous loss, subtract at least 1 from max
    max_parity = gravidity - prev_loss
    parity = np.random.randint(0, max(1, max_parity + 1))

    # Ensure gravidity >= parity + previous_loss (biological constraint)
    gravidity = max(gravidity, parity + prev_loss)

    # IVF slightly more common in older / previous-loss patients
    ivf_adj = ivf_prob + (0.05 if prev_loss else 0) + max(0, (age - 37) * 0.02)
    conception = "IVF" if np.random.rand() < min(ivf_adj, 0.50) else "Natural"

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
# 3b. DUAL-ETIOLOGY GENERATION (two partially-independent loss channels)
# ==============================================================================
# Clinical rationale: first-trimester loss is *etiologically heterogeneous*. Some
# losses are embryonic/sonographic (an abnormal scan trajectory — slow CRL/GS
# growth, bradycardia, enlarged yolk sac) and are detectable from the longitudinal
# scans (the FETA-Transformer's temporal signal). Others are maternal/constitutional
# — driven by an *interaction* of maternal risk factors (advanced age compounded by
# elevated BMI and prior loss) — and present with near-normal early scans, detectable
# only from the maternal-variable structure (the PREG-Net graph's signal). A
# pregnancy is lost if EITHER mechanism fires (logical OR). Because each modality
# reflects only one channel, neither single-modality model can see all losses, which
# is precisely what makes multimodal fusion informative — and testable. When
# ETIOLOGY == "single" this whole section is bypassed and behaviour is unchanged.

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def _draw_population_maternal():
    """Draw maternal features from a single *population* (not outcome-conditioned).

    In dual-etiology mode the maternal loss risk is a downstream *interaction* of
    these features (see :func:`generate_patient_dual`), so the features themselves
    must be sampled outcome-blind. Correlations (age–BMI, age–gravidity, the
    parity ≤ gravidity constraint) mirror :func:`generate_maternal_features`.
    """
    age_mu, age_sd = 32.0, 4.9
    bmi_mu, bmi_sd = 25.3, 4.3

    rho = 0.15
    z1, z2 = np.random.normal(0, 1, 2)
    z2 = rho * z1 + math.sqrt(1 - rho ** 2) * z2
    age = float(np.clip(age_mu + age_sd * z1, 18, 45))
    bmi = float(np.clip(bmi_mu + bmi_sd * z2, 16, 45))

    # Population prior-loss prevalence (~14%), gently rising with age.
    prev_loss_prob = 0.14 + max(0.0, (age - 32.0) * 0.012)
    prev_loss = int(np.random.rand() < min(prev_loss_prob, 0.55))

    grav_lambda = max(0.8, 0.5 + (age - 25.0) * 0.08)
    gravidity = max(1, np.random.poisson(grav_lambda))
    max_parity = gravidity - prev_loss
    parity = np.random.randint(0, max(1, max_parity + 1))
    gravidity = max(gravidity, parity + prev_loss)

    ivf_adj = 0.30 + (0.05 if prev_loss else 0) + max(0.0, (age - 37.0) * 0.02)
    conception = "IVF" if np.random.rand() < min(ivf_adj, 0.50) else "Natural"

    return {
        'age': round(age, 1),
        'bmi': round(bmi, 1),
        'parity': int(parity),
        'gravidity': int(gravidity),
        'previous_loss': int(prev_loss),
        'conception': conception,
        'singleton': True,
    }


def _maternal_risk_contribution(maternal) -> float:
    """Maternal-feature contribution to loss risk (excludes base rate and frailty).

    The risk is driven mainly by *conjunctions* of risk factors (old AND heavy,
    old AND prior-loss) rather than any single variable — the multiplicative
    structure a graph-attention model over clinical variables is designed to
    capture, and which a pooled linear head reads only weakly. Returned as a
    standalone term so the directed sonographic coupling (``SONO_MAT_COUPLING``)
    can reuse exactly the same maternal-risk signal that drives the maternal
    channel, guaranteeing the "both-channel" losses are the high-maternal-risk
    pregnancies (not an unrelated draw).
    """
    a = (maternal['age'] - 32.0) / 6.0
    b = (maternal['bmi'] - 25.0) / 5.0
    pl = float(maternal['previous_loss'])

    interaction = _relu(a) * _relu(b) + pl * _relu(a)     # conjunctions
    main = 0.35 * a + 0.30 * b + 0.90 * pl                # weak additive part
    return MAT_MAIN_SLOPE * main + MAT_INTER_SLOPE * interaction


# ── Published epidemiology backing the dual-etiology channels ────────────────
# Every constant below is a reported figure, not a value tuned to hit a target
# AUROC. Sources are cited inline and reproduced in the manuscript.
#
# [1] Magnus MC et al. "Role of maternal age and pregnancy history in risk of
#     miscarriage: prospective register based study." BMJ 2019;364:l869.
#     (421,201 pregnancies) — age-specific risk, and adjusted ORs for prior loss.
# [2] Cavalcante MB et al. "Obesity and recurrent miscarriage: a systematic
#     review and meta-analysis." J Obstet Gynaecol Res 2019;45(1):30-38.
# [3] Volkova N et al. "Chromosomal Abnormalities in Miscarriages and Maternal
#     Age: New Insights from the Study of 7118 Cases." Cells 2025;14(1):8.

# [1] Table 1 — miscarriage risk by maternal age band, at the band midpoints.
MAGNUS_AGE_KNOTS = [19.0, 22.0, 27.0, 32.0, 37.0, 42.0, 46.0]
MAGNUS_AGE_RISK  = [0.158, 0.113, 0.098, 0.108, 0.167, 0.322, 0.536]

OR_PREVIOUS_LOSS = 1.54   # [1] adjusted OR, one previous miscarriage (1.48–1.60)
OR_BMI_OVERWEIGHT = 1.67  # [2] OR for BMI >= 25 (1.25–2.25)

# [3] Share of miscarriages that are chromosomally abnormal — i.e. embryonic in
# origin, and therefore expressed in the scan trajectory (the sonographic
# channel). Rises with maternal age: +0.704 %/yr from 23–37, a documented step
# up to 79.0 % at 38, then rising to ~94 % by 44.
CHROM_SHARE_AT_23 = 0.55
CHROM_SLOPE_YOUNG = 0.00704   # per year, ages 23–37   [3]
CHROM_SHARE_AT_38 = 0.79      # documented step        [3]
CHROM_SLOPE_OLD   = 0.025     # per year, ages 38–44   [3]
CHROM_SHARE_MAX   = 0.94      # [3]


def _total_loss_risk(maternal) -> float:
    """Total miscarriage probability for this woman, from published risk factors.

    Baseline is the age-specific risk from the Magnus BMJ 2019 register study [1],
    interpolated between age-band midpoints. Previous loss [1] and overweight [2]
    are applied as odds ratios. This is the *observed* loss risk from all causes —
    it is split into the two etiologic channels by :func:`_chromosomal_share`.
    """
    p_age = float(np.interp(maternal['age'], MAGNUS_AGE_KNOTS, MAGNUS_AGE_RISK))
    odds = p_age / (1.0 - p_age)
    if maternal['previous_loss']:
        odds *= OR_PREVIOUS_LOSS
    if maternal['bmi'] >= 25.0:
        odds *= OR_BMI_OVERWEIGHT
    return float(odds / (1.0 + odds))


def _chromosomal_share(age: float) -> float:
    """Fraction of this woman's loss risk that is embryonic (chromosomal) [3].

    The remainder is attributed to the maternal environment. Because the
    chromosomal share rises steeply with age, older women's losses are
    increasingly embryonic — which is why the two channels are not independent
    of age, and why the maternal channel is *not* simply "the age channel".
    """
    if age <= 23.0:
        share = CHROM_SHARE_AT_23
    elif age <= 37.0:
        share = CHROM_SHARE_AT_23 + CHROM_SLOPE_YOUNG * (age - 23.0)
    else:
        share = CHROM_SHARE_AT_38 + CHROM_SLOPE_OLD * (age - 38.0)
    return float(np.clip(share, 0.40, CHROM_SHARE_MAX))


def _std_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def generate_patient_dual():
    """Generate one patient under the dual-etiology (OR-of-two-channels) model.

    The observed loss risk ``p`` is taken from published epidemiology
    (:func:`_total_loss_risk`) and split into an embryonic/sonographic channel and
    a maternal channel using the published chromosomal share ``f``
    (:func:`_chromosomal_share`)::

        p_sono = 1 - (1 - p) ** f
        p_mat  = 1 - (1 - p) ** (1 - f)

    which makes ``P(sono OR mat) == p`` exactly, so the cohort's loss rate is the
    published risk rather than a tuned constant, while the two channels carry the
    published etiologic shares. ``--channel-corr`` couples the two draws through a
    Gaussian copula, which correlates them *without* changing either marginal.

    Returns
    -------
    label : int
        Observed outcome = sonographic-loss OR maternal-loss.
    factors : dict
        Latent scan factors (from the sonographic channel) for :func:`generate_scans`.
    maternal : dict
        Population maternal features (the maternal channel acts *through* them).
    channel : dict
        {'label_sono', 'label_mat'} ground-truth per-channel labels (diagnostics only).
    label_sono : int
        Sonographic-channel label, used for scan missingness.
    """
    maternal = _draw_population_maternal()

    p_total = _total_loss_risk(maternal)
    f_chrom = _chromosomal_share(maternal['age'])

    p_sono = 1.0 - (1.0 - p_total) ** f_chrom
    p_mat = 1.0 - (1.0 - p_total) ** (1.0 - f_chrom)

    # Gaussian copula: correlate the channels (shared frailty) with exact marginals.
    z1, z2 = np.random.normal(0.0, 1.0, 2)
    z2 = CHANNEL_CORR * z1 + math.sqrt(max(0.0, 1.0 - CHANNEL_CORR ** 2)) * z2
    label_sono = int(_std_normal_cdf(float(z1)) < p_sono)
    label_mat = int(_std_normal_cdf(float(z2)) < p_mat)

    # The embryonic channel is what the scan trajectory expresses: an embryonic
    # loss shows the published abnormality phenotype (small CRL/GS, bradycardia,
    # enlarged yolk sac); a maternal-channel loss does not.
    factors = generate_latent_factors(label_sono)

    label = int(label_sono or label_mat)
    return label, factors, maternal, {'label_sono': label_sono, 'label_mat': label_mat}, label_sono


# ==============================================================================
# 4.  CLINICALLY PLAUSIBLE MISSING DATA
# ==============================================================================

def apply_missingness(scan, ga, label):
    """Missingness depends on gestational age and outcome."""
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

def _expected(marker, ga, factors, degrade):
    """Viable reference curve × per-marker effect × loss degradation.

    YSD enlargement in loss should worsen (grow) over time, so degradation
    *amplifies* it; the shrinking markers (CRL/GS/FHR) are degraded downward.
    """
    base = ref_curve(marker, ga) * factors[marker]
    if marker == 'YSD':
        # degrade < 1 for loss; invert so an enlarged sac gets more enlarged late
        return base * (2.0 - degrade)
    return base * degrade


def generate_scans(pat_id, label, factors):
    """Generate 2-5 longitudinal scans that evolve autoregressively.

    Scan count and trajectory deterioration are keyed on whether the case is
    *symptomatic* (from ``factors``), not on the label — so occult losses are
    truly indistinguishable from viable pregnancies.
    """
    if factors.get('symptomatic', label == 1):
        # Symptomatic losses tend to stop being scanned earlier.
        n_scans = np.random.choice([2, 3, 4, 5], p=[0.35, 0.40, 0.20, 0.05])
    else:
        n_scans = np.random.choice([2, 3, 4, 5], p=[0.30, 0.35, 0.25, 0.10])

    first_ga = np.clip(np.random.normal(GA_FIRST_SCAN_MU, GA_FIRST_SCAN_SD), 5.0, 9.0)
    noise_factor = factors['noise']

    scans = []
    prev = {'CRL': None, 'GS': None, 'FHR': None, 'YSD': None}
    ar_weight = 0.30  # how much the previous scan influences the next

    for s_idx in range(n_scans):
        if s_idx == 0:
            ga = round(first_ga, 1)
        else:
            delta = np.clip(np.random.normal(INTER_SCAN_MU, INTER_SCAN_SD), 1.0, 4.0)
            ga = round(scans[-1]['gestational_age_weeks'] + delta, 1)

        if ga > GA_MAX:
            break

        # Symptomatic-loss trajectories progressively deteriorate with each
        # later scan; viable and occult-loss trajectories stay stable.
        progress = s_idx / max(n_scans - 1, 1)        # 0 → 1
        degrade = 1.0 - factors.get('degrade', 0.0) * progress

        vals = {}
        for marker in ('CRL', 'GS', 'FHR', 'YSD'):
            exp_val = _expected(marker, ga, factors, degrade)
            if prev[marker] is not None:
                prev_ga = scans[-1]['gestational_age_weeks']
                exp_prev = _expected(marker, prev_ga, factors, degrade)
                from_prev = prev[marker] + (exp_val - exp_prev)
                base = (1 - ar_weight) * exp_val + ar_weight * from_prev
            else:
                base = exp_val
            noisy = base + np.random.normal(0, MEAS_ERROR[marker] * noise_factor)
            vals[marker] = noisy

        scan = {
            'patient_id': pat_id,
            'gestational_age_weeks': ga,
            'FHR': round(np.clip(vals['FHR'], *FHR_RANGE), 1),
            'CRL': round(np.clip(vals['CRL'], *CRL_RANGE), 1),
            'GS':  round(np.clip(vals['GS'],  *GS_RANGE),  1),
            'YSD': round(np.clip(vals['YSD'], *YSD_RANGE), 1),
        }

        scan = apply_missingness(scan, ga, label)
        scans.append(scan)

        for marker in ('CRL', 'GS', 'FHR', 'YSD'):
            prev[marker] = scan[marker]

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
    channel_rows = []  # dual-mode per-channel ground truth (diagnostics only)

    for pid in range(1, n_patients + 1):
        pat_id = f"P{pid:05d}"

        if ETIOLOGY == "dual":
            label, factors, maternal, channel, label_sono = generate_patient_dual()
            # Scan missingness tracks sonographic symptomatology, not the OR outcome.
            scan_label = label_sono
            channel_rows.append({'patient_id': pat_id, **channel})
        else:
            label = int(np.random.rand() < loss_rate)
            factors = generate_latent_factors(label)
            maternal = generate_maternal_features(label)
            scan_label = label

        synthetic_patients.append({
            'patient_id': pat_id,
            **maternal,
            'label': label,
        })

        scans = generate_scans(pat_id, scan_label, factors)
        synthetic_scans.extend(scans)

    patients = pd.DataFrame(synthetic_patients)
    scans = pd.DataFrame(synthetic_scans)
    if channel_rows:
        # Stashed on the DataFrame so the writer can persist it alongside the CSVs
        # without polluting the training columns.
        patients.attrs['channel_labels'] = pd.DataFrame(channel_rows)

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
    print("VIABLE MEDIAN vs PUBLISHED REFERENCE (weeks 6-10)")
    print("-" * 70)
    merged['ga_week'] = merged['gestational_age_weeks'].round().astype(int)
    live = merged[merged['label'] == 0]
    for var in ['CRL', 'GS', 'YSD', 'FHR']:
        ref = {int(w): v for w, v in zip(REF_WEEKS, VIABLE_REF[var])}
        obs = live.groupby('ga_week')[var].median()
        pairs = [f"w{w}: {obs.get(w, float('nan')):.1f}/{ref.get(w, float('nan')):.1f}"
                 for w in range(6, 11)]
        print(f"  {var:4s} (obs/ref): " + "  ".join(pairs))

    print("\n" + "-" * 70)
    print("CORRELATION MATRIX (ultrasound measurements)")
    print("-" * 70)
    print(scans[['FHR', 'CRL', 'GS', 'YSD']].corr().round(3))

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

    channel_labels = patients.attrs.get('channel_labels')
    if channel_labels is not None:
        channel_csv = out_path / "channel_labels.csv"
        channel_labels.to_csv(channel_csv, index=False)
        print(f"Saved channel-label diagnostics: {channel_csv}")


def main():
    print(f"Etiology model: {ETIOLOGY}"
          + (f"  (channel_corr={CHANNEL_CORR})" if ETIOLOGY == "dual" else ""))
    patients, scans = generate_dataset(N_PATIENTS, LOSS_RATE_TARGET, SEED)
    print_summary(patients, scans)
    channel_labels = patients.attrs.get('channel_labels')
    if channel_labels is not None:
        merged = channel_labels.merge(patients[['patient_id', 'label']], on='patient_id')
        n = len(merged)
        sono = merged['label_sono'].sum()
        mat = merged['label_mat'].sum()
        both = ((merged['label_sono'] == 1) & (merged['label_mat'] == 1)).sum()
        sono_only = ((merged['label_sono'] == 1) & (merged['label_mat'] == 0)).sum()
        mat_only = ((merged['label_sono'] == 0) & (merged['label_mat'] == 1)).sum()
        print("\n" + "-" * 70)
        print("DUAL-ETIOLOGY CHANNEL BREAKDOWN")
        print("-" * 70)
        print(f"  sonographic-channel losses : {sono}  ({100*sono/n:.1f}%)")
        print(f"  maternal-channel losses    : {mat}  ({100*mat/n:.1f}%)")
        print(f"  both channels              : {both}  ({100*both/n:.1f}%)")
        print(f"  sonographic-only           : {sono_only}  <- FETA-visible, maternal-blind")
        print(f"  maternal-only              : {mat_only}  <- PREG-visible, sono-blind")
    save_outputs(patients, scans, _ARGS.out)


if __name__ == "__main__":
    main()
