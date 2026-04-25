# PSA & VPSA Cyclic Adsorption Simulator

Two Python script sets that solve coupled, stiff, hyperbolic-parabolic PDE systems describing **Pressure Swing Adsorption (PSA)** and **Vacuum Pressure Swing Adsorption (VPSA)** beds. Each cycle is integrated phase-by-phase, with the converged bed state propagated via a fixed-point iteration on cycle-end loadings until **Cyclic Steady State (CSS)** is reached.

| Set | Application | PDE system size | Pressure swing |
|-----|------------|-----------------|----------------|
| **PSA** | H₂ purification from refinery off-gas | 10 × N states (5 species × {gas, solid}) | 15 bar → 1 bar |
| **VPSA** | CO₂ capture from flue gas | 6 × N states (3 species × {gas, solid}) | 1 bar → 0.1 bar |

Both use the same numerical pipeline: **finite-volume spatial discretization → method of lines → BDF time integration with sparse Jacobian → Numba-compiled RHS evaluation**. Cycle-level convergence is enforced by an outer Picard loop on the solid-phase loading.

Each set is driven by a **master orchestrator** that calls phase scripts in sequence, hands bed state from one phase to the next via `.npz` snapshots, and loops the cycle until the CSS residual drops below tolerance.

---

## 1. Repository Layout

```
psa-simulator/
├── master.py                       # PSA orchestrator
├── PSA(ads,New).py                 # PSA adsorption phase
├── PSA(Depressurization).py        # PSA blowdown + purge (combined)
├── PSA(Repressurization).py        # PSA repressurization phase
│
├── master VPSA.py                  # VPSA orchestrator
├── VPSA(ads,New).py                # VPSA adsorption phase
├── VPSA(Rinse).py                  # VPSA heavy reflux (CO₂ rinse)
├── VPSA(Depressurization).py       # VPSA blowdown to vacuum
├── VPSA(Repressurization).py       # VPSA repressurization phase
├── VPSA(CoD).py                    # VPSA cocurrent depressurization (optional)
│
├── master_config.json              # Auto-generated shared parameter file
├── *_end_state.npz                 # Auto-generated bed-state handover files
└── results/
    ├── SCOUT_cycle_0/              # Breakthrough scouting plots
    ├── CSS_cycle_1/ ... CSS_cycle_N/
    ├── FINAL_cycle_FINAL/          # Final diagnostic on converged bed
    └── CSS_Timing_Report.txt       # CapEx/OpEx phase timings
```

The `results/` folder is created on first run and contains every plot, summary, and breakthrough curve, organized by cycle.

---

## 2. Requirements

- **Python ≥ 3.9**
- `numpy`, `scipy`, `matplotlib`, `numba`, `tqdm`

```bash
pip install numpy scipy matplotlib numba tqdm
```

The scripts force `matplotlib.use('Agg')` so they run headless on servers without a display.

---

## 3. How to Run It

From the directory containing the scripts:

```bash
# For hydrogen PSA
python3 "master.py"

# For CO2 capture from flue gas (VPSA)
python3 "master VPSA.py"
```

That is the **only command you need to run**. The master script does everything else: it wipes any leftover bed-state files, writes `master_config.json`, then runs each phase script in the correct order, repeating cycles until convergence. Open `results/CSS_cycle_<last>/` afterward to inspect the converged-cycle plots, and `results/FINAL_cycle_FINAL/` for the full breakthrough diagnostic on the converged bed.

---

## 4. The Three-Phase Workflow

Both masters follow the same overarching workflow:

### Phase A — SCOUT Run (Cycle 0)

The bed starts clean (PSA: pure H₂; VPSA: pure feed). The adsorption script runs for a long time (`t_ads_end` = 2000 s for PSA, 2500 s for VPSA) so the breakthrough curve fully develops.

The script then:

1. Finds the time at which exit purity drops below 99.92 %.
2. Multiplies that breakthrough time by `t_ads_safety_ratio` (0.9–0.95) → this becomes `t_op_ads`, the **operating adsorption time**.
3. Auto-scales the other phase times from configurable ratios:
   - `t_blowdown_end = Blowdown_Ratio × t_total`
   - `t_rep         = Repress_Ratio  × t_total`
   - `tf_des        = Purge_Ratio    × t_total` (PSA) / `Blowdown_Ratio` for VPSA
   - `t_rinse       = Rinse_Ratio    × t_total` (VPSA only)
4. Writes the optimized timings back into `master_config.json`.

The Scout run produces a single breakthrough plot per gas component (`ads_breakthrough_curve.png`) and is the only phase that uses purity-based time selection.

### Phase B — CSS Loop (Cycles 1 … N)

The master wipes the saturated Scout bed (so Cycle 1 starts clean) and then loops:

```
for cycle in 1 .. max_cycles:
    Adsorption        → adsorption_end_state.npz
    [Rinse]           → rinse_end_state.npz       (VPSA only)
    Depressurization  → desorption_end_state.npz  (PSA combines BD + Purge)
    Repressurization  → repressurization_end_state.npz
    Compare q_end vs previous_cycle_state.npz
    if residual < tolerance: break
```

Convergence criterion (the bed is at CSS when):

```
‖ q_end_current − q_end_previous ‖₂  /  ‖ q_end_previous ‖₂   <   tolerance
```

Default tolerance: **1e-3** (VPSA), **8e-3** (PSA). `q` is the solid-phase loading (mol/kg). Once the loading profile stops drifting between cycles, the cycle is operating at steady state and any single cycle from then on is representative of long-term plant behavior.

### Phase C — FINAL Diagnostic Run

With the bed converged, the master runs **one more long adsorption** (`t_ads_end` = 2000–2500 s) on the CSS bed. This produces the *real-plant* breakthrough curve, the spatial concentration profiles, and the Ergun pressure drop along the column for a steady-state cycle. Output goes to `results/FINAL_cycle_FINAL/`.

---

## 5. How the Scripts Talk to Each Other

There is no inter-process Python state. Each phase is a standalone script. They coordinate through three channels:

**1. `master_config.json`** — written by the master at start-up and updated by the Scout adsorption with optimized timings. Every phase script reads this for bed dimensions, pressures, ratios, and the Sips/Toth isotherm constants.

**2. `*_end_state.npz`** — bed-state handover. Each phase ends by saving the gas concentration field `C_end` (5 species × N nodes) and the solid loading field `q_end` to a `.npz` file. The next phase loads it as the initial condition. Files used:

| File | Written by | Read by |
|------|-----------|---------|
| `adsorption_end_state.npz` | Adsorption | Rinse / Blowdown |
| `rinse_end_state.npz` | Rinse (VPSA) | Blowdown |
| `desorption_end_state.npz` | Blowdown/Purge | Repressurization |
| `repressurization_end_state.npz` | Repressurization | Adsorption (next cycle) |
| `previous_cycle_state.npz` | Master | Master (convergence check) |

**3. Environment variables** — the master sets two before each `subprocess.run`:

- `RUN_TYPE` ∈ `{SCOUT, CSS, FINAL}` — tells the adsorption script whether to run long-and-optimize, run-to-`t_op_ads`, or run long-on-converged-bed.
- `PSA_CYCLE` — used for naming the per-cycle results folder.

---

## 6. Numerical Method

Each phase reduces a coupled PDE system to a stiff ODE system via the **method of lines**, then integrates in time with an implicit BDF solver. All phases share the same discretization machinery but differ in boundary conditions, source terms, and the analytic form of the pressure trajectory.

### 6.1 Spatial Discretization — Cell-Centered Finite Volume

The column **z ∈ [0, L]** is split into **N = 100** equal cells of width `dz = L/N`, with nodes placed at cell centers `z_j = (j + ½)·dz`. State variables are the cell-averaged gas concentration `C_{i,j}` and solid loading `q_{i,j}` for each species *i* and cell *j*.

Convective transport is handled with a **first-order upwind flux**: for cocurrent (feed-driven) flow the inlet flux at cell *j* is the outlet flux of cell *j-1*; for countercurrent flow (blowdown, purge, vacuum regeneration) the indexing is reversed. Upwinding is chosen over higher-order schemes (e.g. WENO) because the steep concentration fronts in PSA breakthrough are stable and monotone under upwind, and a flux-limiter `C / (C + 1e-6)` further enforces non-negativity at near-zero concentrations.

Diffusion is neglected — Peclet numbers in industrial-scale PSA columns are O(10³–10⁴), so axial dispersion is dominated by numerical diffusion from the upwind discretization itself, which is the standard simplification.

### 6.2 Velocity Field — Cumulative Continuity

The superficial velocity `u(z)` is not a primary state variable. Instead, `du/dz` is reconstructed at every RHS call from the local total-moles balance, and `u(z)` is rebuilt by **cumulative summation** (`np.cumsum`) starting from a known boundary:

- Adsorption: `u(z=0) = u_feed` (Dirichlet inlet), march forward.
- Blowdown / purge / repressurization: `u` is integrated backward from the open boundary using `np.cumsum(du[::-1])[::-1]`.

The resulting velocity is then **clipped** to `[1e-6, 5.0] m/s` to keep the integrator out of unphysical regions during early transient steps.

### 6.3 Pressure Sub-Model

Two different treatments depending on the phase:

- **Adsorption** — Pressure is solved by the **Ergun equation** integrated cell-by-cell along *z*, treating the high-pressure end as the boundary. To avoid an impulsive Δp at *t = 0* (when `u = u_feed` slams the empty bed), Ergun is multiplied by a **soft-start ramp** `(1 − exp(−t/5))`.
- **Blowdown / Repressurization** — Pressure follows a prescribed analytic trajectory `P(t) = P_low + (P_high − P_low)·exp(−t/τ_bd)`. This decouples pressure from the ODE state, turning what would otherwise be a DAE into a pure ODE and dramatically improving solver stiffness behaviour.

### 6.4 Time Integration — `scipy.integrate.solve_ivp` with BDF

The MOL system is **stiff** (LDF rate constants `k_LDF` differ from convective time scales by 3–4 orders of magnitude), so an implicit method is mandatory:

```python
solve_ivp(rhs, [t0, tf], y0, method='BDF',
          t_eval=..., rtol=1e-2, atol=1e-2,    # adsorption
          rtol=5e-2, atol=1e-4, first_step=1e-11)  # blowdown / purge
```

Tolerances are deliberately loose (`rtol = 1e-2 … 5e-2`). Tighter tolerances buy little physical accuracy because the isotherm parameters themselves carry ~5 % uncertainty, and they sharply increase Jacobian-evaluation cost. The very small `first_step = 1e-11 s` for blowdown forces BDF past the discontinuity at *t = 0* when pressure starts collapsing.

### 6.5 Sparse Jacobian Hint

For the regeneration phases, a **lower-triangular block sparsity pattern** is precomputed and passed to BDF:

```python
sparsity_matrix[eq_type*N + j, var_type*N + j : var_type*N + N] = 1
jac_sparsity = sp.csc_matrix(sparsity_matrix)
```

This reflects the fact that with cocurrent upwind flow, the value at cell *j* depends only on cells *j … N-1* (or symmetrically for countercurrent). BDF then evaluates the finite-difference Jacobian column-by-column over the sparse pattern instead of as a full `(10N × 10N)` matrix — order-of-magnitude speedup for `N = 100`.

### 6.6 RHS Acceleration — Numba JIT

The adsorption RHS is decorated with `@njit(cache=True)` and **fully unrolled** — no array broadcasting, just explicit inner loops over species and cells. Vectorized NumPy ends up slower here because the RHS is called thousands of times during a single integration and each call allocates many small temporaries; Numba compiles the loop down to a single tight C kernel and reuses pre-allocated buffers.

### 6.7 Numerical Regularization

Several smoothing tricks keep the integrator stable across regime changes:

- `leaky_max(x, ε, α)` — a differentiable substitute for `max(x, ε)` that keeps a small gradient on the clipped side, preventing the BDF Jacobian from going singular when concentrations approach zero.
- **Sigmoid layer blend** `w(z) = 1 / (1 + exp(−10·(z − z_mid)))` — replaces a step transition between bed layers with a smooth ramp over ~2 cells, eliminating spurious oscillations at the layer interface.
- **Tanh activation gate** on the LDF rate during adsorption — turns uptake on smoothly as the local concentration crosses the trace threshold.
- **Soft Ergun ramp** `(1 − exp(−t/5))` — avoids a step jump in pressure drop at *t = 0*.
- **Bounded p-ratio** `np.clip(P_partial / P_atm, 1e-10, 1000)` — keeps `^n_param` away from `0^negative` and `large^large`.

### 6.8 Outer Loop — Cyclic Steady State as a Picard Iteration

The full cycle is treated as a **fixed-point map** on the solid-phase loading field:

```
q_{k+1}(z) = Φ_cycle( q_k(z) )
```

where `Φ_cycle` is the composition of all phase integrators (Ads ∘ Rinse ∘ BD ∘ Rep). Convergence is monitored by the relative L₂ residual:

```
‖ q_{k+1} − q_k ‖₂ / ‖ q_k ‖₂  <  tol
```

with `tol = 1e-3` (VPSA) or `8e-3` (PSA). This is a Picard iteration; convergence is linear with rate equal to the spectral radius of the linearization of `Φ_cycle`, which in practice is 0.3–0.7 for well-designed cycles (10–30 iterations to converge).

### 6.9 SCOUT — Threshold-Based Time Selection

The SCOUT run replaces a continuous optimization with a simple **first-passage time detection**: integrate adsorption past breakthrough, then locate the last index where exit purity ≥ 99.92 %, multiply by `t_ads_safety_ratio ∈ [0.9, 0.95]`, and write that back to the config. All other phase times scale linearly off this single number through the user-supplied ratio set. This sidesteps the need for a gradient-based optimizer in the outer loop while still adapting cycle timing to the actual breakthrough physics.

### 6.10 Diagnostic Quadratures

Cumulative quantities (moles in, moles out, recovery) are computed by **trapezoidal integration** (`np.trapz`) over the BDF-returned `t_eval` grid. A grounded mass balance is enforced at each output time:

```
moles_out_gross(t) = ∫₀ᵗ ṅ_in dτ  −  Δ(bed_inventory)
```

so any rounding drift in the integrator is absorbed into the inventory term rather than the product-stream metric.

### 6.11 Phase-Specific Notes (Numerical)

- **Adsorption** — solves the full 10·N (PSA) or 6·N (VPSA) system; PSA additionally evaluates a **Peng-Robinson cubic** (`np.roots`) for the gas-phase compressibility factor at each call, picking the largest real root (vapour-like).
- **Rinse (VPSA)** — runs adsorption-style PDE but with a pure-CO₂ inlet at intermediate pressure; reuses the adsorption RHS with swapped boundary conditions.
- **Blowdown / Purge** — the pressure trajectory becomes an explicit `t`-dependent forcing term (no longer a state), shrinking the Jacobian footprint and stiffness.
- **Repressurization** — same kernel as blowdown with the sign of the pressure ramp reversed.
- **CoD (VPSA, optional)** — cocurrent depressurization between rinse and blowdown; commented out in `master VPSA.py` line 126, enable to add the step.

---

## 7. Configuration: What You Can Tune

Edit the `master_params` dict at the top of `master.py` or `master VPSA.py`. The most useful knobs:

**Bed geometry & operating conditions**

| Parameter | Meaning | Default (PSA / VPSA) |
|-----------|---------|----------------------|
| `L` | Bed length (m) | 12 / 12 |
| `d` | Bed diameter (m) | 3 / 4.5 |
| `P_high` | High pressure (Pa) | 15 atm / 1 atm |
| `P_low` | Low pressure (Pa) | 1 atm / 0.1 atm |
| `T` | Temperature (K) | 303.15 / 293.15 |
| `Nsets` | Number of bed pairs in the train | 3 / 12 |

**Phase time ratios** (must sum to 1.0 across the cycle)

| Parameter | Meaning |
|-----------|---------|
| `Adsorption_Ratio` | Fraction of cycle on adsorption (default 0.50) |
| `Blowdown_Ratio` | Fraction on depressurization |
| `Purge_Ratio` (PSA) / `Rinse_Ratio` (VPSA) | Fraction on regeneration |
| `Repress_Ratio` | Fraction on repressurization |
| `CoD_Ratio` (VPSA) | Fraction on cocurrent depressurization |

**Solver / convergence**

- `t_ads_end` — length of the SCOUT and FINAL adsorption runs (must be long enough to see breakthrough)
- `t_ads_safety_ratio` — multiplied by breakthrough time to set the operating adsorption time
- `tau_bd` — blowdown time constant for the exponential pressure decay
- `convergence_tolerance` — CSS convergence threshold (in `master.py`/`master VPSA.py`)
- `max_cycles` — abort the CSS loop after this many cycles even if not converged
- `N` — number of spatial nodes (default 100; raise for accuracy, lower for speed)

**Material properties** (PSA has two layers blended by a sigmoid at `ratio_layer1`; VPSA is single-layer)

- `eps_1`, `eps_2`, `rho_s_1`, `rho_s_2` (PSA) — bed voidage and solid density per layer
- `eps`, `rho_s` (VPSA) — single-layer values
- `dp` — particle diameter
- `k_ldf_*`, `k1…k6` — isotherm constants (defined in each phase script, not in master)

---

## 8. Outputs

Per cycle (`results/<RUN_TYPE>_cycle_<n>/`):

- `ads_performance.png` — Purity & Recovery vs. time, plus instantaneous mass flows
- `ads_breakthrough_curve.png` — C/C₀ at z = L for each species (SCOUT + FINAL only)
- `ads_profiles_2d.png` — spatial mol-fraction profiles + breakthrough curve (CSS + FINAL)
- `ads_pressure_profile.png` — Ergun pressure along the column at several times
- `des_flow_pressure_metrics.png` — blowdown & purge tail-gas flows and inlet/exit pressures
- `des_spatial_profiles.png` — combined regeneration elution curve + per-species spatial loading
- `adsorption_summary.txt` — purity, recovery, annual H₂ (PSA) or CO₂ (VPSA) production (ton/yr)
- `desorption_summary.txt` — tail-gas composition (BD vs. Purge), instantaneous purge front

Once converged, the master also writes:

- `results/CSS_Timing_Report.txt` — final timings for **CapEx / OpEx sizing**, including:
  - Adsorption / Rinse / Blowdown / Purge / Repress times
  - Total cycle time
  - Cycles simulated to converge
  - Bed L, d, L/D ratio, number of pairs

---

## 9. Common Issues

- **`FileNotFoundError: …_end_state.npz`** — a phase ran before the prior phase finished. The master clears these on the right transitions; if you run a phase script directly, you must set `PSA_CYCLE` and `RUN_TYPE` env vars and ensure the upstream `.npz` exists.
- **Velocity warning ("Bed Fluidisized") in VPSA** — `u_feed > 1.3 m/s` means feed flow is too high for the bed cross-section. Increase `d` or reduce `feed_input` in the master.
- **CSS never converges** — usually means the phase ratios don’t balance (bed is over- or under-regenerated each cycle). Try: lower `Adsorption_Ratio`, raise `Purge_Ratio`/`Rinse_Ratio`, or raise `t_ads_safety_ratio` away from 1.
- **`RuntimeWarning` from `np.roots` in Peng-Robinson** — harmless; the warning filter suppresses it. Z-factor selects the largest real root (vapour-like).
- **Long runtime** — first call to a Numba-compiled function compiles it (~5–10 s). Subsequent cycles reuse the cache.

---

## 10. Why It’s Structured This Way

A monolithic single-file simulator would be easier to read but harder to debug — when the recovery is wrong, you want to know whether the bug is in adsorption, regeneration, or repressurization without re-running the full cycle.

Splitting each phase into its own script means:

- Each phase can be **rerun independently** for debugging given the right `*_end_state.npz` and `master_config.json`.
- The master is a thin orchestrator: it owns the cycle logic and convergence check; it owns no physics.
- The CSS loop is a clean `for cycle in range(...)` whose inner body is four `subprocess.run` calls. You can add a phase (the VPSA Rinse and CoD steps were added this way) without touching the others.
- Results are organized per cycle, so you can scrub through `cycle_1, cycle_2, …` and watch the bed approach steady state visually.
