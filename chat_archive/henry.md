# Project Archive — Ocean-Current Diffusion Inpainting

> Durable, curated record of everything tried and the results.
> Distilled from the main working chat session (the large active session,
> `0d3590ba…`, ~470 turns). Tool-by-tool narration, SSH/git plumbing, and
> repeated "check on the model" turns removed; **all approaches, decisions,
> measured results, and the north star are kept.**
> The earlier Jun 15 codebase-survey chat is summarized separately at the bottom.

---

## NORTH STAR (the goal everything serves)
Given a few **known pixels** (a sparse robot-path of observations, ~3.6–3.7% of
ocean cells, which is FIXED and won't change), predict a **plausible** ocean-current
vector field. Key principles:
- Observations are **soft** constraints (approximate, not exact).
- The problem is underdetermined → we want the **best realistic guess**, that is
  **divergence-free** and **looks like a real ocean current** (coherent eddies).
- Use diffusion stochasticity to produce **N diverse plausible guesses**.
- **AVOID the blurry mean / climatology** (playing it safe). Each individual draw
  should be full-magnitude and structured even where direction is uncertain.
- Accuracy still matters where the field is informed (near the path).

**Key theory anchors:**
- **Perception–Distortion tradeoff** (Blau & Michaeli 2018): minimizing distortion
  (MSE/angle error) and maximizing realism are *provably* at odds. The min-distortion
  estimator is the posterior **mean** = the blurry thing we want to avoid. So a
  far-field angular-error "floor" (~53°) is EXPECTED, not a failure — the metric was
  the problem.
- **Coverage sweep proved far-field is information-limited, not fit-limited**: the
  deep/unobserved band stays ~53° regardless of total path coverage; only *observed*
  cells improve. Overall-error improvements from longer paths were just more observed
  cells diluting the pool.

---

## THE DATA
- Original: `data.pickle` (raw ROMS, float64, ocean u/v mean −0.051, std 0.116).
- `data_divfree.pickle`: Leray-projected (exactly divergence-free) version of raw.
- **Chrono rebuild** `data_divfree_chrono.pickle` (564 MB, from `ramhead_dataset.mat`,
  17040 hourly frames, 0 gaps): one continuous time-ordered div-free array, splits
  train 11639 / val 1680 / test 1680. Built with 20 POCS Leray iters
  (|div| 0.008 → 0.002). This fixed an earlier corrupted-prior problem
  (prior corr 0.20 → 0.654).
- Grid 94×44, ~3787 ocean / 4136 cells. Land = NaN.
- **Time structure (measured):** adjacent-frame corr 0.907 vs random 0.267 → frames
  are time-ordered, **1 step ≈ 1 hour**. Lag-correlation shows an M2 semidiurnal tide
  (~12.4 h): minima at lag 6, peaks at lag 13 / 25 / 37 (same tidal phase).
- **Lag selection for conditioning:** whole-field corr lag13=0.599 ≳ lag25=0.571;
  vorticity/eddy corr lag13=0.446 ≈ lag25=0.445 (tied). 13h and 25h are essentially
  equivalent for eddies → **condition on BOTH (13h + 25h)**.
- Divergence reality check: Helmholtz decomposition of raw data shows ~12% of the
  energy is divergent (curl-free), ~87% rotational. So a pure stream-function model
  discards ~12% of real signal (concentrated at coasts/upwelling). Verdict: 12% is
  moderate, acceptable for coherent-eddy fields; keep the div-free constraint.

---

## APPROACHES TRIED, IN ORDER, WITH RESULTS

### Baseline to beat
- **Gaussian DDPM + RePaint (r=10)**: mean **RMSE 0.1146** (the GP/old benchmark).
- Geometric-schedule run (older): mean RMSE 0.1635 ± 0.0374.

### 1. Divergence-free noise (Helmholtz / stream-function noise)
- **Problem solved:** RePaint's hard-snap of known cells injects spurious divergence
  at the obs boundary. A divergence-free *noise* keeps the forward/reverse process
  on the div-free manifold.
- **Implementation:** sample isotropic Gaussian, FFT, project out the curl-free part
  per Fourier mode (`dot_k = kx·û + ky·v̂`), renormalize by a **single scalar** for
  both u and v (using separate scalars would break div-free), zero the Nyquist modes
  to preserve Hermitian symmetry.
- **★ Critical bug found & fixed:** the projection originally used the **spectral
  symbol** `kx = fftfreq` (i·k), but the model's curl and the divergence metric use a
  **central-difference** stencil whose Fourier symbol is `i·sin(2π·fftfreq)`, NOT i·k.
  Noise was div-free in the periodic-FFT sense (~3e-7) but had central-diff
  divergence ~0.40 (≈ field magnitude). **Fix:** `kx,ky = sin(2π·fftfreq)`. After fix,
  central-diff interior |div| = 1.4e-7, std = 1.0.

### 2. PPR — Predict-Project-Renoise (new inference method)
- Instead of RePaint's hard-snap, at each reverse step **project the clean estimate
  x̂₀** onto {divergence-free ∩ matches observations} (POCS), then renoise. No masked
  glue → no divergence seam.
- **Results (div-free model):**
  - mean |div| drops to **~0.002** (≈4× cleaner than the ground-truth data itself,
    which is ~0.008–0.025), vs RePaint's ~0.013–0.020 seam.
  - **RMSE 0.109** on the old pipeline — edges out the GP baseline (0.1146) *and* is
    nearly divergence-free. **This was the headline win for the unconditional model.**
  - On a half-trained checkpoint (epoch 78) PPR already hit RMSE 0.1252; the div-free
    model's val loss 0.00271 (ep110) beat the Gaussian's 0.00327 (ep155) by ~17%.
- **RePaint vs PPR, same model:** RePaint produced nearly-blank fields (hard-snap
  drives everything toward zero) with a speckled divergence seam; PPR recovered
  coherent directional structure with near-zero divergence.

### 3. "Colored" div-free noise (spectral filter) — REGRESSION, abandoned
- Added realistic high-frequency energy via a precomputed amplitude spectrum.
- **Result:** RMSE got *worse* (0.109 → 0.155). High-frequency detail in unobserved
  regions is essentially unpredictable; injecting it hurt accuracy. The old squashed
  field had been "cheating" by regressing toward zero (most ocean cells are slow).
- Also exposed a **normalization bug**: un-normalizing made land cells become the mean
  (−0.046) instead of 0, contaminating coastline divergence. Reverted to the clean
  spectral approach (central diffs, FFT projection, 20 POCS iters) → |div| back to 0.003.

### 4. DPS (Diffusion Posterior Sampling) — alternative guidance
- Gradient step toward matching observations through the Tweedie estimate (autograd),
  instead of RePaint's merge trick. Trained/run on the vast.ai GPU.
- Used later with the stream-function model: **stream-fn + DPS ≈ 55.8% overall**
  angle accuracy (still poor in the deep far-field; near-field much better).

### 5. The diagnosis that reframed everything: post-hoc conditioning has a floor
- Across RePaint, PPR, DPS: every *post-hoc* constraint-injection method on an
  *unconditional* model lands at the same wall. This is the signature of a
  fundamental limitation — **the model never learned what "observed" means.**
- → The fix is **conditional training** (channel-concatenation inpainting), the
  field-standard answer. RePaint/PPR/DPS exist as training-free workarounds for when
  you *can't* retrain — but we can.

### 6. ★ Direction + Magnitude fusion (the approach that worked really well)
This is the decomposition you remember. It came from the **angle-only era**:
- **Direction model** — a div-free DDPM trained with a **cosine/angle loss**
  (`Div_Free_DDPM_Angle`). Because cosine **divides out vector length**, the model
  has zero incentive to get magnitude right — it only learns *orientation*. Its output
  vector lengths are meaningless.
- **Magnitude model** — predicts speed **separately**, fully decoupled:
  - **GP magnitude regressor** (chosen first): one Gaussian Process on the scalar
    speed |v|=√(u²+v²) at the ~3.6% known path cells → dense speed field. Deliberately
    ignores the DDPM field (whose magnitudes are noise under angle loss).
    *Results (random val samples):* speed RMSE 0.0372 (val1924), 0.0596 (val1919),
    relRMSE ~0.41–0.45. Recovers the broad slow→fast gradient, smooths fine structure.
  - **Magnitude UNet** (alternative): 1-channel regressor, MSE on held-out speed.
    Only modestly beat climatology (hard, given ~3.6% coverage).
- **Fusion** (`DDPM/testing/direction_magnitude_display.py`):
  $$\text{field} = \underbrace{\text{unit\_direction}}_{\text{angle DDPM}} \times \underbrace{\text{magnitude}}_{\text{GP / Magnitude UNet}}$$
  Direction comes from the diffusion model; calibrated speed from the separate
  regressor. Models on disk: `Div_Free_DDPM_Angle_Old/New.pt`,
  `Magnitude_UNet_Old/New.pt`, plus the GP magnitude scripts under `GP Baseline/`
  (`gp_magnitude.py`, `batch_magnitude.py`).

**Why it worked well:** it cleanly separates the two physically-different sub-problems
— direction is a structured generative problem (diffusion excels), magnitude is a
smooth scalar regression problem (GP/UNet excels). Neither task fights the other.

### 7. Stream-function x0 model (current main pipeline — supersedes the fusion)
- **`StreamFunctionUNet`**: outputs a single scalar stream function ψ; the field is
  its discrete **curl** (u = ∂ψ/∂W, v = −∂ψ/∂H) → **exactly divergence-free by
  construction** (interior |div| ~ machine zero).
- **Key insight that retires the separate magnitude model:** train with **recon-MSE on
  the full (u,v) field + an angle loss** (Min-SNR-γ=5 weighting, λ_angle=1). Recon-MSE
  on both components means **one network learns calibrated magnitude AND direction**.
  The angle/magnitude split was only ever a workaround for the cosine-only loss.
- **Model selection leaderboard** (same 5 val samples, overall mean angle error °):
  - OLD+RePaint 76.8 | OLD+PPR-snap 61.8 | NEW+RePaint 61.0 | NEW+PPR-snap 54.0 |
    NEW-angle+PPR-pocs 50.3 | **NEW-EPS+PPR-pocs 45.2°** (near 22.7, deep 45.7) |
    **STREAM-FN+PPR-pocs 55.8°** (near **8.4**, deep 63.2).
  - **Nuance:** eps "wins" overall (45.2) only by regressing the deep far-field toward
    **climatology** (the mean-prediction the north star rejects). By **near-field**
    accuracy (where data exists) stream-fn **8.4° crushes eps 22.7° (~3×)**, and
    stream-fn is div-free by construction. → **Stream-function selected** on
    north-star metrics (near-field accuracy + div-free + non-mean far field).

### 8. Conditional stream-function DDPM (channel-concat conditioning)
- **12 input channels** to `StreamFunctionUNet` (concat, widen first conv):
  - 2  latent noisy (u,v)
  - 3  observation (soft): obs_u, obs_v, path_mask
  - 4  temporal prior: prev-13h (u,v), prev-25h (u,v)  ← the main far-field info lever
  - 3  geometry (free, from land mask): coord_x, coord_y, distance-to-coast
- Output ψ → curl → (u,v), still exactly div-free. Conditioning doesn't break div-free
  (uncond div ≈ 0.003 preserved). Friend's earlier work was a **FiLM**-conditioned
  Gaussian-noise version with a Voronoi-primed mode; this merges div-free noise + the
  channel-concat conditioning. (FiLM is used only for the **timestep** embedding here,
  not for the conditioning signal.)
- **Training result:** 300/300 epochs, best **epoch 89, val 0.604** — val never
  improved after ep89 (plateaued, *not* undertrained → more epochs refuted).
- **Inference diagnosis — magnitude collapse (exposure bias, NOT a code bug):**
  - Teacher-forced x̂₀ (model fed q_sample of true x₀) has std ~0.9–1.0 at every t →
    the model **can** represent full magnitude. E[x₀|cond] from pure noise also full
    (std 0.974). But the actual sampling trajectory decays std 0.96 → 0.40 in the first
    ~15 steps and stays ~0.40. Each individual draw is weak (~0.35–0.40), not just the
    mean. Ruled out: land-masking, stochasticity (DDIM η=0 → 0.41), step count
    (100 vs 1000 → same), x0↔eps round-trip (exact).
  - Mechanism: cosine ᾱ tiny at high t → early full-magnitude x̂₀ barely enters x_t →
    trajectory slowly embeds the model's own weak output → compounding drift off the
    training manifold.
  - Near-path angle ~24.7°, deep far-field ~34.9° — direction is good and diverse;
    magnitude is the weak point.

### 9. x0 vs v-prediction (fighting magnitude collapse)
- **Plain v-prediction:** fixed magnitude collapse (std-ratio ~1.0 at all t) BUT
  introduced **high-frequency speckle** — measured 53% of energy above k>0.25 vs GT's
  1.4%. Root cause is structural: v reconstructs x̂₀ = √ᾱ·x_t − √(1−ᾱ)·v̂, and at t→0
  this ≈ x_t (the raw noisy final iterate), so it inherits accumulated sampling noise.
  x0-prediction instead returns a fresh curl(ψ) at the last step → smooth. **x0 is the
  structurally better parameterization** for coherent div-free flow.
- **v + Min-SNR-γ:** down-weights high-noise timesteps (where v roughness blows up) via
  `w_v = min(SNR,γ)/(SNR+1)`. Should reduce speckle but can't fully overcome the
  x_t-anchoring → unlikely to reach x0 coherence.
- **★ Chosen fix — improved x0 + magnitude loss:** keep x0; add a scale-invariant
  magnitude term `L_mag = mean[(rms(x̂₀)/rms(x₀) − 1)²]` over ocean cells, per-sample.
  Rationale: the angle loss is scale-invariant, so MSE is the *only* magnitude signal
  and it mean-seeks under directional uncertainty. λ_mag chosen via gradient-norm
  calibration (grad_norm recon 0.26 vs mag 0.56 at ep89 → λ_mag=0.2 ≈ 43% of recon
  gradient). Smoke-tested (mag=0 at pred=truth, 0.25 at half-magnitude, finite grads).
- **Result (trained):** fresh 300-epoch run with the FIXED central-diff noise; later a
  small **spread** term was added to discourage per-draw magnitude collapse. Best usable
  checkpoint = **`Models/StreamFn_Cond_x0_mag_spread.pt` (ep48)** — this is the diffusion
  member-generator used for **all** the calibration / hetero / coupled work below.
  Magnitude collapse on hard samples is mitigated; div-free preserved (~0.003).

---

## RESULTS SUMMARY TABLE (RMSE / mean|div| on ~3.6–3.7% coverage)
| Method | RMSE | mean \|div\| | Notes |
|---|---|---|---|
| Gaussian DDPM + RePaint (r=10) | **0.1146** | ~0.013 | the baseline to beat (eps, 10 val, biased-walk 150) |
| GP full-field baseline (Matérn ν=2.5) | 0.2011 ± 0.0786 | n/a | ~75% worse than DDPM, 3.6× more variance |
| Geometric-schedule DDPM | 0.1635 | — | older run |
| Div-free DDPM + RePaint | 0.118–0.155 | ~0.013–0.020 | hard-snap seam, near-blank fields |
| **Div-free DDPM + PPR** | **0.109** | **0.002** | beat GP baseline + ~div-free ★ |
| Colored div-free noise + PPR | 0.155 | 0.002–0.005 | regression — abandoned |
| GP magnitude regressor | speed RMSE 0.037–0.060 (relRMSE ~0.41–0.45) | n/a | speed only, for fusion |
| Stream-fn + PPR-pocs | 55.8° overall / **8.4° near** | exact 0 | best near-field, div-free by construction |
| eps + PPR-pocs | 45.2° overall / 22.7° near | — | "wins" overall only via climatology regression |
| Conditional stream-fn x0 (ep89) | near 24.7° / deep 34.9° | ~0.003 | magnitude collapse (exposure bias) |
| Plain v-prediction | val 0.572 (ep196) | ~0.024 | magnitude fixed but 53% speckle |
| **x0 + magnitude (+spread) loss** | `StreamFn_Cond_x0_mag_spread.pt` ep48 | ~0.003 | the chosen fix ★ — backbone for calibration era |

*Angle errors and RMSE are different metrics from different eras — angle for the
direction/stream-fn models, RMSE for the full-field reconstruction models.*

---

## CALIBRATION STUDY — r_angle / r_magnitude / r_overall (the uncertainty-map era)

**What the metric means.** Probe `Conditional DDPM/testing/_probe_calib_all.py`.
For each frame we draw an ensemble of model samples and build a **model spread map**
(per-cell dispersion) and compare it, via **Pearson r over ocean cells**, against an
**empirical neighbour-posterior spread map** (the spread of nearby-in-time real
frames given the same observations). Three correlations:
- **r_angle** — directional spread (is the model uncertain where direction is uncertain?)
- **r_magnitude** — speed-only spread (is it uncertain where speed is uncertain?)
- **r_overall** — full-vector spread.
- **RMSE%** — ensemble-mean accuracy (relative). Lower = more accurate; this must
  **not** rise when we add uncertainty calibration.

Decomposition insight: `r_overall ≈ speed-weighted blend of radial (σ_speed²) +
tangential (sbar²·Var_angle)`. Lifting r_overall needs a better **σ field**, which is
what drove the v2 variance head.

### Fusion-mode ablation (12 frames, n_model=10 / n_emp=20)
| fuse_mode | r_angle | r_magnitude | r_overall | RMSE% | off-path dispersion |
|---|---|---|---|---|---|
| replace  | +0.66 | +0.38 | +0.23 | 71.9 | 19.4% (vs raw diffusion 66.7%) |
| reinject | +0.66 | +0.32 | +0.38 | 76.9 | 28.7% |
| none     | +0.66 | −0.03 | +0.06 | 102.5 | — |

**Finding:** `replace` (overwrite every draw with ONE deterministic UNet speed map)
zeroes magnitude diversity → r_magnitude/r_overall stuck. **You cannot calibrate the
uncertainty of a deterministic quantity.** r_angle is high only because direction is
stochastic. No post-hoc trick lifts both magnitude correlations without hurting RMSE.

### Heteroscedastic + coupled progression (6 frames: 2091, 6800, 9312, 11597, 13982, 16306; n_model=40 / n_emp=60)
Fix = **HeteroMagnitudeUNet** (predicts per-cell mean μ AND log-variance → speed
~ N(μ, σ²)), warm-started from `Cond_Magnitude_UNet.pt` with **backbone+mean FROZEN**
so ensemble-mean RMSE is provably unchanged; only σ is learned (Gaussian NLL).

| approach | r_angle | r_magnitude | r_overall | RMSE% |
|---|---|---|---|---|
| hetero v1 (1×1 logvar head, white-noise sampling) | +0.757 | +0.563 | +0.541 | 70.4 |
| coupled v1 (diffusion magnitude anomaly, 1×1 head) | +0.757 | +0.594 | +0.553 | 70.6 |
| **coupled v2 (anomaly-coupled, 3×3 smooth head)** | **+0.757** | **+0.676** | **+0.597** | **71.1** |

**Net (hetero v1 → coupled v2):** r_magnitude **+0.113 (+20%)**, r_overall **+0.056**,
**r_angle untouched**, **RMSE flat** (accuracy preserved). The two design moves:
1. **Coupled sampling** — instead of per-cell Gaussian white noise (salt-and-pepper,
   angle-blind), reuse the diffusion draw's **own** smooth magnitude anomaly
   z_k=(m_k−mbar)/s and affine-map to the UNet's μ,σ: `speed_k = clip(μ + σ·z_k, 0)`.
   Spatially coherent + direction-consistent.
2. **v2 variance head** — two 3×3 convs + 1×1 (27,745 params, vs v1's 65) with a
   **TV-smoothness prior** on logvar → a coherent σ field instead of a patchy one.

### Per-frame coupled v2 (r_magnitude / r_overall)
| frame | r_magnitude | r_overall | note |
|---|---|---|---|
| 2091  | .749 | .564 | |
| 6800  | .260 | .339 | weak/information-limited frame |
| 9312  | .869 | .582 | |
| 11597 | .554 | .719 | |
| 13982 | .739 | .677 | |
| 16306 | .887 | .700 | |

On the strong frames **r_magnitude exceeds r_angle** — calibrated speed uncertainty
is now as good as (or better than) the directional uncertainty.

### Variance-head training (Gaussian NLL, lower = better)
| head | params | best epoch | val_nll | notes |
|---|---|---|---|---|
| v1 (single 1×1 logvar conv) | 65 | ep14 | −0.462 | mean σ (phys) 0.0308 |
| **v2 (3×3×2 + 1×1, TV-smooth prior)** | 27,745 | ep4 | **−0.552** | smoother σ field |

Constants used in fusion: data_std 0.10628, speed_mean 0.1301, speed_std 0.1000.

**Artifacts:** `Models/{StreamFn_Cond_x0_mag_spread.pt (ep48), Cond_Magnitude_UNet.pt}`;
hetero ckpts `Magnitude/checkpoints_cond_mag_hetero/` (v1) and `…_v2/` (v2);
probes `Conditional DDPM/testing/{_probe_calib_all.py, _probe_multidraw.py}`
with `--fuse_mode {replace, reinject, none, hetero, coupled}`; uncertainty maps in
`Conditional DDPM/results/{cond_calib_maps, cond_calib_maps_v2}`.

**Coupled v2 across-frame summary (6 frames):** r_magnitude **+0.676 ± 0.236**,
r_overall **+0.597 ± 0.141** (sample std; the ±0.236 is inflated entirely by the one
information-limited frame 6800 at r_mag 0.260 — the other five sit 0.55–0.89).

---

## DIAGNOSTIC PROBE METRICS (the measurements behind the decisions)

### Magnitude-collapse probe — teacher-forced rms ratio (x0 conditional, ep89)
`utils/_probe_x0mag.py`: feed the model `q_sample(true x₀, t)` and measure
`mean|x̂₀| / mean|x₀|` over ocean vs t (1.0 = full magnitude):
| t | 999 | 500 | 200 | 100 | 50–5 |
|---|---|---|---|---|---|
| rms ratio | 71% | 82% | 86% | 87% | 87% (flat) |
Flat ~87% at low t = mild mean-seek + mild undertrain (model *can* represent
magnitude). The **sampled** trajectory is worse — std decays 0.96 (t999) → 0.40
(t<850) in ~15 steps → trajectory compounding. GT norm std = 0.886; a single sampled
member on a hard frame ≈ 0.35–0.40.

### Structure probe — x0 vs v vs GT (sample 685, normalized units)
`utils/_probe_structure.py` (roughness = mean|Laplacian|/mean|field|, HF = spectral
energy fraction k>0.25, tv = total variation):
| field | magnitude | roughness | HF frac | tv |
|---|---|---|---|---|
| GT | 0.464 | 0.25 | **1.4%** | 0.18 |
| x0 | 0.178 (collapsed here) | 1.57 | 11.6% | 0.86 |
| v | 0.833 | 2.90 | **53%** (half = speckle) | 1.94 |
Confirms "v has no structure" quantitatively: 53% HF vs GT 1.4%. x0 keeps magnitude
low on this hard frame but is far less speckly.

### Coherence diagnostic — roughness vs noise level
`utils/_diag_coherence.py` (teacher-forced, same 8 frames): v roughness-ratio is LOW
at t=50 (~1.4) but **blows up at t≥500 (~18–20)**; x0 is uniform (~17) across t. → v's
speckle is a *high-noise-timestep* phenomenon (exactly what Min-SNR down-weights), but
x_t-anchoring is structural so v can't reach x0 coherence.

### Divergence reality check — Helmholtz on RAW data (per split)
Divergent (curl-free) energy fraction, central-diff sin-symbol operator, ocean cells:
train **11.71%**, val **11.42%**, test **11.31%** (~87% rotational). Intrinsic stream-fn
floor vs raw data ≈ √0.117 ≈ 0.34 of field RMS — but we train/eval on projected
div-free data so metrics never see it. Verdict: 12% is moderate → keep the constraint.

### Land-mask train/infer mismatch A/B (does masking land at inference matter?)
Measured (x0, 5 samples, 100 steps): masked 56.3° vs unmasked 57.8° ocean-angle = a
**wash** (mixed per-sample). Hypothesized big confounder, measured, was wrong.

### Conditional x0 10-sample eval (MPS, 100 steps, seed 20260624)
Near-path angle 12–24° (good); deep/unobserved 22–95° (sample-dependent); mean angle
~27–83° across samples. Genuine model performance, not a pipeline bug
(`results/cond_streamfn_x0_eval10/`).

---

## KEY LESSONS / DECISIONS (don't relearn these the hard way)
- **Measure, don't claim.** The land-mask train/infer mismatch was hypothesized to be
  a big confounder; A/B test showed 56.3° (masked) vs 57.8° (unmasked) = a wash.
- The **div-free noise spectral-symbol bug** (i·k vs i·sin) was the one real noise bug;
  everything downstream uses central differences, so the noise must too.
- The far-field angular **floor is information-limited**, not a model failure — it's
  the perception–distortion tradeoff. Don't chase RMSE < 0.1; the best ever was 0.109.
- **x0-prediction > v-prediction** for coherent div-free flow (v gives speckle).
- The separate **magnitude model is only needed under angle-only loss**; a recon-MSE
  stream-fn x0 model learns magnitude + direction in one network.
- **Devices:** training on the GPU (vast.ai CUDA); inference on Mac **MPS**
  (CUDA > MPS > CPU). div-free noise does FFT on CPU then moves to device (MPS lacks
  complex FFT).
- Keep the div-free constraint now; if coastal/upwelling divergence matters later,
  relax via a **Helmholtz two-potential** model (u,v) = curl(ψ) + grad(φ), φ
  regularized — relax via *architecture*, not by adding divergence to the dataset.

---

## OPEN / NEXT
- Validate the retrained **x0 + magnitude-loss** checkpoint: magnitude probe
  (teacher-forced low-t rms ratio → ~100%), structure probe, 10-sample infer_cond eval
  on MPS, compare against old x0 on hard samples.
- Optionally regenerate fused angle×magnitude views via
  `direction_magnitude_display.py` (`Div_Free_DDPM_Angle_New.pt` + `Magnitude_UNet_New.pt`).
- Inference diversity tooling exists (`samplers.py`: vanilla / particle-filter / DPS;
  `compare_samplers.py`) — run once a good conditional ckpt exists.

---

## APPENDIX — Jun 15 codebase-survey chat (separate, earlier session)
Early "give me a rundown of the codebase" conversation + environment setup. Method
inventory: DDPM core + UNet, RePaint inpainting, noise-schedule study
(cosine/linear/quadratic/sigmoid/geometric), GP baseline (Matérn ν=2.5),
VoronoiNet, auxiliary structural losses (divergence/spectral/gradient/OT-Sinkhorn).
Only committed numeric result at that time: geometric schedule mean RMSE 0.1635 ± 0.0374.
Setup facts: project-local `.venv` at repo root, NumPy pinned `<2` (PyTorch wheel),
`launch.json` runs current file from repo root, PYTHONPATH needs repo root +
`DDPM/model` + `DDPM/testing`.
