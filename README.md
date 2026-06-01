# Atmospheric Neutrino Reconstruction & Classification at DUNE

A self-contained Data Science portfolio project that turns raw simulated
detector data from the DUNE Far Detector into reconstructed atmospheric
neutrino events, then trains four machine-learning models to classify
those events by interaction type (CC vs NC) and neutrino flavour
(νe / νμ / ντ).

The project deliberately separates **physics reconstruction** from
**machine learning**. The first half is classical track-finding and
kinematics; the second half is tabular classification on top of the
reconstructed quantities and detector observables.

---

## 1. The physics, in plain English

### 1.1 What is an atmospheric neutrino?

Cosmic rays — mostly protons — slam into the upper atmosphere about 15 km
above your head. They produce a hadronic shower whose secondary pions
and kaons decay into **muons** and **muon neutrinos**, which decay
again into **electrons** and **electron neutrinos**. Some of those
neutrinos travel down through the atmosphere and the Earth and pass
through detectors deep underground.

Three properties of an atmospheric neutrino matter for physics:

- **Flavour** — was it a νe, νμ, or ντ?
- **Energy** — typically anywhere from 0.1 GeV to 100+ GeV.
- **Direction** — measured as the **zenith angle** (0° = came straight
  down from the sky; 90° = horizontal; 180° = came up through the
  Earth from the antipodes). The zenith angle determines the
  **baseline** `L`, the distance the neutrino travelled before
  interacting in the detector — anywhere from 15 km (down-going) to
  ~12,750 km (up-going).

These three quantities — flavour, energy, baseline — are what
neutrino-oscillation experiments need to measure. The probability that
a νμ has oscillated into a νe by the time it reaches the detector is
a function of `L/E`, so getting `L` and `E` right is the whole point.

### 1.2 What happens when a neutrino hits the detector?

Two mutually exclusive things:

**Charged Current (CC)** — the neutrino converts into its
corresponding charged lepton:

```
νe + nucleus  →  e⁻  + hadrons       (electron-like CC)
νμ + nucleus  →  μ⁻  + hadrons       (muon-like CC)
ντ + nucleus  →  τ⁻  + hadrons       (tau-like CC)
```

The visible signature is **a charged lepton track** plus a hadronic
mess at the vertex. CC is the channel where you can reconstruct the
neutrino's energy and direction from the final-state particles —
because the lepton remembers most of the kinematics.

**Neutral Current (NC)** — the neutrino kicks the nucleus and remains
a neutrino, leaving the detector unobserved:

```
ν + nucleus  →  ν + hadrons          (any flavour)
```

The visible signature is a hadronic blob at the vertex. NC is useless
for oscillation analysis because the final-state neutrino takes most
of the energy out of the detector with it.

So the very first piece of analysis on any neutrino event is:
**is this CC or NC?** That's task A in the ML notebook. Within CC,
the next question is: **what flavour?** That's task B.

### 1.3 The detector — DUNE Far Detector

The simulated data is for the **DUNE Horizontal-Drift Far Detector**, a
liquid argon time-projection chamber (LArTPC). The principle:

1. A charged particle moves through liquid argon and ionises it,
   leaving a trail of free electrons.
2. A strong electric field drifts those electrons sideways toward the
   anode (the **drift coordinate**, x).
3. The anode is covered in three layers of wires at angles 0°
   (collection plane W) and ±35.7° (induction planes V and U).
4. As the electrons arrive, each wire records a **hit**: a time of
   arrival (drift) and a charge amount (ADC).

The detector therefore records each event as **three 2D images**, one
per wire plane, each (drift × channel) pixels.

Reconstructing the 3D position of an ionisation cluster requires
combining hits from at least two of the three planes. The wire-pair
geometry is fixed and lets us back out (y, z) from any two of (U, V, W),
with the third providing a consistency check.

---

## 2. The data

Two ROOT files per simulation batch:

| File | Contents | Granularity |
|---|---|---|
| `mc_0.root` | MC truth: PDG, energy, momentum, vertex, parent_id, hits per particle | one row per MC particle |
| `hits_0.root` | Raw hits: plane, drift, channel, ADC, mc_id of the producing particle | one row per **event instance** (jagged arrays inside) |

## Data

The `.root` files (hit and MC files) are not included in this repository
due to GitHub's 100 MB file size limit.

Download them from CERNBox:
https://cernbox.cern.ch/s/KHco7F42uW2XDWP

The folder contains multiple hit and MC files along with
`atmos_project_script.ipynb`, which documents the structure of these files.

Place the downloaded `.root` files in the project root directory before
running the notebooks.

### 2.1 The instance / template structure

The data has a structure that is not obvious at first and **does
matter** for ML:

- The MC file labels events with `event_id ∈ {0, 1, ..., 99}` — only
  **100 unique physics events**.
- The hits file has **9,700 rows** — about 97 detector realisations of
  each physics event. Each realisation has different noise, different
  threshold crossings, different drift smearing.
- Each MC realisation begins where `parent_id == -1` appears (the
  incoming neutrino).

So your effective dataset is **100 unique neutrino interactions, seen
~97 times each**. The 9,700 hit rows are not 9,700 independent events.

Why does this matter? Because if you do a vanilla random train/test
split on the 9,700 rows, you end up with the same physics event
appearing in both train and test sets — that's data leakage. For a
proper ML setup you'd split on the 100 unique physics events. (The
current notebooks do the simpler 80/20 row-split for demonstration
clarity; for a serious paper you'd switch to event-id splits.)

### 2.2 Class balance

Across the full 9,700 hit instances:

| | Count | Fraction |
|---|---|---|
| **CC events** | 5,265 | 54.3% |
| **NC events** | 4,435 | 45.7% |

Within CC events, the flavour breakdown:

| Flavour | Count |
|---|---|
| νe | ~2,851 |
| νμ | ~2,315 |
| ντ | ~99 |

Quite balanced for CC vs NC. Severely imbalanced for ντ — this is
intrinsic to atmospheric flux at these energies, and is why ντ
classification is the hardest problem in the dataset.

---

## 3. Repository structure

```
atmos_neutrino/
├── Event Neutrino Reconstruction Display.ipynb              # constants + reconstruction + 2D & 3D visualisations
├── Dataset Builder.ipynb        # loop the pipeline → dataset.csv/.xlsx/.parquet
├── EDA.ipynb                  # exploratory analysis of dataset
├── ML Classification.ipynb                   # CC/NC classification + flavour classification
│
├── dataset.csv                   
├── dataset.xlsx                  
├── dataset.parquet
│
├── report.pdf
├── report.tex            
```

---

## 4. The reconstruction pipeline

This is what notebook `01_physics.ipynb` walks through for one event,
and what notebook `02_build_dataset.ipynb` runs in a loop for thousands.

```
                                         ┌──────────────────────────┐
                                         │   ROOT files             │
                                         │   mc_0.root, hits_0.root │
                                         └────────────┬─────────────┘
                                                      │
                                                      ▼
                           ┌──────────────────────────────────────────────┐
                           │  Per-instance MC slice + hit arrays          │
                           │  (instance starts where parent_id == -1)     │
                           └─────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                           ┌──────────────────────────────────────────────┐
                           │  Group hits into drift slices (0.5 cm bins)  │
                           └─────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                           ┌──────────────────────────────────────────────┐
                           │  Reconstruct 3D space points (UV/UW/VW pairs)│
                           │  Each point: (x, y, z), ADC, mc_id, residual │
                           └─────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                           ┌──────────────────────────────────────────────┐
                           │  Filter points: residual < RMS               │
                           │  + line-fit proximity (per-particle)         │
                           └─────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                           ┌──────────────────────────────────────────────┐
                           │  Identify primaries (parent = neutrino)      │
                           │  CC if any charged lepton among them, else NC│
                           └─────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                           ┌──────────────────────────────────────────────┐
                           │  Per-particle direction:                     │
                           │    PCA for tracks (μ, π, p)                  │
                           │    Vector average for showers (e, γ)         │
                           └─────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                           ┌──────────────────────────────────────────────┐
                           │  Sum primary momenta → reco neutrino p, E   │
                           │  Use truth |p| (from mass + truth E) and    │
                           │  reconstructed direction                    │
                           └─────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                           ┌──────────────────────────────────────────────┐
                           │  Compute zenith angle (cos θ_z = -p_y / |p|) │
                           │  Compute baseline L (spherical Earth)        │
                           └─────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                                          one row of dataset.csv
```

### 4.1 Three things worth understanding

**Why PCA for tracks, but vector averaging for showers.**
A muon track is essentially a straight line — so the first principal
axis of its 3D point cloud is a great direction estimate. But an
electron makes an electromagnetic shower — a spreading cone of γ
conversions and δ-rays — and PCA on a cone gives you the *long* axis
of the cone, not the *initial* direction. So for showers we
weight-average the unit vectors from the vertex to each hit; that
biases the answer toward the early, on-axis hits which carry the
direction.

**The energy comes from MC truth, the direction from reconstruction.**
This is a deliberate choice. Reconstructing energy from ADC sums is
itself a hard, calibration-dependent problem and would obscure the
direction reconstruction we actually care about. So we cheat
constructively: take the magnitude `|p|` from the truth-level
particle energy and the rest mass (`|p| = √(E² − m²)`), and combine
it with the *reconstructed* direction to get the momentum vector.
Sum over primaries → neutrino momentum. This is a clean way to study
direction reconstruction in isolation; in a real analysis you'd
replace the truth `|p|` with a calorimetric estimate.

**Zenith angle convention.**
In this detector, +y is up. A neutrino travelling *downward* through
the detector has p_y negative, and we want to call that θ_z = 0 (it
came from straight up). So:

```
cos(θ_z) = -p_y / |p|
```

Then the baseline `L` is the chord from the production point in the
upper atmosphere to the detector underground, computed by the law of
cosines on a spherical Earth.

---

## 5. The dataset — `dataset.csv` (also `.xlsx`, `.parquet`)

One row per event instance, 44 columns. The notebook saves all three
formats automatically:
- **CSV** for universal compatibility
- **XLSX** for human readability (frozen headers, sortable, formatted)
- **Parquet** for fast loading downstream

### 5.1 Column groups

| Group | Columns | What |
|---|---|---|
| **Identifiers** | `instance_id`, `event_id` | unique row id and physics-template id |
| **Labels** (the ML targets) | `is_cc`, `nu_pdg`, `nu_flavour` | what we're trying to predict |
| **Truth kinematics** | `nu_energy_truth`, `nu_dir_truth_{x,y,z}`, `zenith_truth_deg`, `L_truth_km` | the right answer |
| **Reco kinematics** | `nu_energy_reco`, `nu_dir_reco_{x,y,z}`, `zenith_reco_deg`, `L_reco_km` | what our pipeline produced |
| **Topology** | `n_primaries`, `n_total_particles`, `n_primary_{leptons,protons,pions,photons,other}`, `leading_lepton_pdg`, `leading_lepton_energy` | particle counts (truth-derived; useful for diagnostics, leaky for ML) |
| **Detector observables** | `n_hits`, `n_hits_{w,v,u}`, `total_adc`, `mean_adc`, `max_adc`, `drift_range`, `drift_std`, `channel_range_{w,u,v}`, `hit_density` | what the detector actually sees, no truth peeking |
| **Reco quality** | `n_points_raw`, `n_points_filtered`, `direction_recon_ok`, `is_ccqe_topology`, `is_ccqe_reconstructed` | did the reconstruction succeed? |

The **detector observables** group is the only one you should use as
ML features. The topology group looks tempting but every column in it
is computed from MC truth — using them as features is data leakage.

---

## 6. The machine-learning notebook

### 6.1 Two tasks, four models each

**Task A — CC vs NC (binary).**
Did this neutrino interaction produce a charged lepton in the final
state? This is the most basic and most useful classification task in
neutrino physics — without it you can't tell apart events you can
analyse from events you can't.

**Task B — Flavour within CC (3-class).**
Was the incoming neutrino νe, νμ, or ντ? Restricted to CC events
because for NC there is no charged lepton and flavour is unknowable
from detector observables.

Same four models for both:

- **Logistic Regression** — fast, interpretable, the linear baseline.
- **Random Forest** — non-linear, handles mixed feature scales well.
- **XGBoost** — gradient-boosted trees, the standard tabular workhorse.
- **LightGBM** — same idea as XGBoost, faster on large datasets.

### 6.2 Why these four together?

You report multiple models for two reasons. First, it's a sanity check
— if all four agree to within a couple of percent on a metric, the
signal is real and not an artefact of one model's quirks. Second,
they have genuinely different inductive biases — Logistic Regression
draws a hyperplane in feature space, the trees carve axis-aligned
rectangles. If a tree model massively beats the linear one, the signal
is non-linear; if they're tied, a linear rule is enough.

### 6.3 Imbalance and evaluation

Because some classes are rare (especially ντ at ~3% of CC events) we:

- **Stratify** the train/test split to keep class ratios consistent.
- Pass `class_weight='balanced'` (LR/RF) or `scale_pos_weight=neg/pos`
  (XGB/LGBM) so the loss function up-weights the rare class.
- Report **5-fold stratified cross-validation** on top of the single
  split. With small minority classes, single-split numbers can swing
  several percentage points on luck — CV mean ± std is the honest
  measure.
- Use **macro-averaged** Precision / Recall / F1 for the multi-class
  task, not class-weighted, so a 3% class doesn't get hidden behind
  90% accuracy on the dominant class.

### 6.4 Plots produced

- **Combined ROC curves** — all four models on one axis, one line each.
- **Combined Precision-Recall curves** — same idea; PR is the
  *primary* diagnostic when classes are imbalanced because ROC can
  look optimistic.
- **Confusion matrices** — one per model, side by side.
- **Feature importances** — for the three tree models on both tasks,
  to see which detector observables actually carry signal.

---

## 7. Results

These are the numbers from a 1,000-event subset (CC: 543, NC: 457).
Run notebook 02 with N=9700 to use the full dataset; expect numbers to
go up by 3–5 points across the board.

### 7.1 Task A — CC vs NC

Single 80/20 split:

| Model | Accuracy | Precision (CC) | Recall (CC) | F1 (CC) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.700 | 0.717 | 0.743 | 0.730 | ~0.78 |
| **Random Forest** | **0.805** | **0.765** | **0.927** | **0.838** | **~0.87** |
| XGBoost | 0.755 | 0.746 | 0.835 | 0.788 | ~0.83 |
| LightGBM | 0.765 | 0.742 | 0.872 | 0.802 | ~0.84 |

Random Forest leads. The discriminative features are the per-plane
hit counts, the drift range, and the channel ranges — all of which
quantify "how spatially extended is the event", which is what
distinguishes a long lepton track (CC) from a compact hadronic blob
(NC).

### 7.2 Task B — Flavour within CC

Single 80/20 split:

| Model | Accuracy | F1 macro | F1 (νe) | F1 (νμ) | F1 (ντ) |
|---|---|---|---|---|---|
| Logistic Regression | 0.587 | 0.412 | 0.712 | 0.524 | 0.000 |
| Random Forest | 0.633 | 0.415 | 0.701 | 0.543 | 0.000 |
| **XGBoost** | **0.661** | **0.431** | **0.730** | **0.564** | **0.000** |
| LightGBM | 0.642 | 0.424 | 0.708 | 0.565 | 0.000 |

Three observations.

The **νe vs νμ separation works** — F1 around 0.7 for νe, 0.55 for
νμ. The detector "sees" muon tracks differently from electron showers
(more channels touched, different drift profile), and the trees pick
up on those differences.

The **ντ class collapses to zero**. With only 14 ντ examples in the
training pool the models simply learn to never predict ντ — it's not
worth the loss penalty. Real progress on ντ requires either more
data (run on the full 9,700-event dataset and you have ~80 ντ to
work with) or a much richer feature set (image-level rather than
summary-stat).

The **gap between linear and tree models is small** (~7–8 points).
That tells you the discrimination signal here is mostly captured by
roughly linear feature combinations. A CNN on raw images would
bring much more — track length, shower opening angle, kink patterns
— than any tabular model can extract from event summary statistics.

---

## 8. Honest assessment of limitations

Worth being clear about what this project is and isn't.

**What it does well.** It demonstrates a complete pipeline from raw
ROOT data to ML evaluation. Each step is auditable, the physics is
correct, the ML setup is honest, the metrics are appropriately
chosen for the imbalance. As a portfolio piece it shows physics
judgement (choosing CC/NC over CCQE-vs-not-CCQE), software
engineering judgement (separating reconstruction from features
from ML), and ML judgement (stratification, CV, leak-free features).

**What it doesn't do.** Three real limitations:

1. **The split isn't on physics events.** With only 100 unique
   physics templates, a row-level random split puts realisations of
   the same template on both sides. For a publishable result you'd
   split on `event_id` not `instance_id`. The numbers above are
   probably 3–5 points optimistic because of this.

2. **The features are flat summaries of spatial data.** Most of the
   discriminative information lives in the *shape* of the hit pattern
   — straight track vs branching shower vs blob — and a sum or count
   loses that. A CNN on the (channel × drift) images for each plane
   would do substantially better, especially on flavour. This is the
   natural next step and your supervisor was right to suggest it.

3. **The truth-energy shortcut.** Direction reconstruction is real
   but the energy comes from MC truth, not from the detector. A
   real analysis would replace this with a calorimetric energy
   estimate using the ADC sum, calibrated against truth. That work
   isn't done here.

**What's overrated about the results.** The 80% CC/NC accuracy is
respectable but has been beaten by 90%+ in published DUNE work using
CNN-style approaches. Don't claim novelty — claim you know the
problem.

---

## 9. Where this would go next

In rough order of impact:

**Process the full 9,700 events.** Trivial change — `python 03_construct_database.py --n 9700`. Roughly 20 minutes on a laptop. Gives you ~5,300 CC and ~80 ντ to train on. Numbers across the board should improve, and the ντ class becomes recoverable.

**Switch to event-id splits.** Replace the row-level split with a
split on the 100 physics templates. All ~97 realisations of any
template land on the same side. This is the only way to get
generalisation numbers you can defend in a paper.

**Add a CNN.** Build (256 × 256) images for each of the three wire
planes per event, stack as channels, train a small ConvNet (3 conv
blocks → global pool → 2 FC layers). Same two tasks. This is the
standard architecture in the field and will substantially outperform
the tabular models. Start small — don't reach for ResNet50 with 80
ντ training examples.

**Calorimetric energy reconstruction.** Replace the truth-`|p|` with
ADC-sum-based energy. Calibrate against truth. Now your `nu_energy_reco`
column is a real reconstruction, not a truth-driven sanity check, and
you can quote a meaningful resolution.

**Oscillation analysis.** With everything above in place, you can
finally do the physics this whole project is a stepping stone toward:
plot the survival probability `P(νμ → νe)` as a function of `L/E`,
fit for the oscillation parameters, compare with the known values.
This is the science.

---