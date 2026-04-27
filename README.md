# PSA & VPSA Cyclic Adsorption Simulator

Two Python script sets that solve coupled, stiff, hyperbolic-parabolic PDE systems describing **Pressure Swing Adsorption (PSA)** and **Vacuum Pressure Swing Adsorption (VPSA)** beds. Each cycle is integrated phase-by-phase, with the converged bed state propagated via a fixed-point iteration on cycle-end loadings until **Cyclic Steady State (CSS)** is reached.

| Set | Application | PDE system size | Pressure swing |
|-----|------------|-----------------|----------------|
| **PSA** | H₂ purification from refinery off-gas | 10 × N states (5 species × {gas, solid}) | 15 bar → 1 bar |
| **VPSA** | CO₂ capture from flue gas | 6 × N states (3 species × {gas, solid}) | 1 bar → 0.03 bar |

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

**1. `master_config.json`** — written by the master at start-up and updated by the Scout adsorption with optimized timings. Every phase script reads this for bed dimensions, pressures, ratios, and isotherm constants (Sips/Langmuir hard-coded inside each phase script).

**2. `*_end_state.npz`** — bed-state handover. Each phase ends by saving the gas concentration field `C_end` (3 species for VPSA, 5 for PSA × N nodes) and the solid loading field `q_end` to a `.npz` file. The next phase loads it as the initial condition. Some VPSA states also pass **analytic mass-balance scalars** alongside the field data (`co2_moles_fed`, `co2_moles_rinse`) so the blowdown audit can compute true cycle recovery without re-integrating ramped flows.

| File | Written by | Read by | Extra payload |
|------|-----------|---------|---------------|
| `adsorption_end_state.npz` | Adsorption | Rinse / Blowdown | `co2_moles_fed` (VPSA) |
| `rinse_end_state.npz` | Rinse (VPSA) | Blowdown | `co2_moles_rinse` (VPSA) |
| `desorption_end_state.npz` | Blowdown/Purge | Repressurization | — |
| `repressurization_end_state.npz` | Repressurization | Adsorption (next cycle) | — |
| `previous_cycle_state.npz` | Master | Master (convergence check) | — |

**3. Environment variables** — the master sets two before each `subprocess.run`:

- `RUN_TYPE` ∈ `{SCOUT, CSS, FINAL}` — tells the adsorption script whether to run long-and-optimize, run-to-`t_op_ads`, or run long-on-converged-bed.
- `PSA_CYCLE` — used for naming the per-cycle results folder.

---

## 6. Numerical Method

Each phase reduces a coupled PDE system to a stiff ODE system via the **method of lines**, then integrates in time with an implicit BDF solver. All phases share the same discretization machinery but differ in boundary conditions, source terms, and the analytic form of the pressure trajectory.

### 6.1 Spatial Discretization — Cell-Centered Finite Volume

The column **z ∈ [0, L]** is split into **N = 100** equal cells of width `dz = L/N`, with nodes placed at cell centers `z_j = (j + ½)·dz`. State variables are the cell-averaged gas concentration `C_{i,j}` and solid loading `q_{i,j}` for each species *i* and cell *j*.

Convective transport uses a **first-order upwind flux**, but with two distinct flavours depending on the phase:

- **Static upwind** (PSA adsorption) — flow is always cocurrent, so the inlet flux at cell *j* is just the outlet flux of cell *j-1*. The `flux_in[i] = flux_out_i` rolling assignment runs forward through the loop.
- **Dynamic / sign-aware upwind** (VPSA adsorption, rinse, blowdown, repressurization) — at each cell the local velocity sign `sign(next_u)` is checked, and the upwind donor cell is selected accordingly. This matters because in VPSA the reflux step (rinse) and vacuum blowdown can locally reverse direction, and forcing a fixed upwind direction would inject mass against the actual flow.

A flux-limiter `C / (C + 1e-6)` is added in PSA blowdown to suppress oscillations when concentrations approach zero. VPSA achieves the same goal via the simpler `max(C, 0)` clamp inside the dynamic-upwind branch.

Diffusion is neglected — Peclet numbers in industrial PSA columns are O(10³–10⁴), so axial dispersion is dominated by the numerical diffusion of the upwind scheme itself.

### 6.2 Velocity Field — Cumulative Continuity

Superficial velocity `u(z)` is not a primary state variable. At each RHS call, `du/dz` is reconstructed from the local total-moles balance (LDF source + gas-density change) and `u(z)` is rebuilt by **cumulative summation** anchored at a known boundary:

- **PSA adsorption / VPSA adsorption / rinse** — `u(z=0) = u_feed` (Dirichlet inlet), march forward through the loop.
- **VPSA blowdown** — `u(z=L) = 0` (closed product end), integrate **backward** using a reverse-loop `for j in range(N-2, -1, -1)`. The bottom of the bed is the open vacuum port, so velocity grows in magnitude (negative sign convention) as the loop walks down.
- **VPSA repressurization** — `u(z=0) = 0` (closed feed end), integrate forward from cell 0 toward the open product-end inlet at z = L. Pure-N₂ feed is injected only when `v_inlet ≤ 0` (gas flowing inward).
- **PSA blowdown / purge / repressurization** — `np.cumsum(du[::-1])[::-1]` integrates from the open boundary back toward the closed end.

PSA additionally **clips** the result to `[1e-6, 5.0] m/s` to keep BDF out of unphysical regimes during the startup transient.

### 6.3 Pressure Sub-Model

Three different treatments depending on the phase:

- **Adsorption (both)** — Pressure is solved by the **Ergun equation** integrated cell-by-cell along *z*, anchored at the high-pressure inlet. To avoid an impulsive Δp at *t = 0*, Ergun is multiplied by a **soft-start ramp** `(1 − exp(−t/5))` (PSA) or `(1 − exp(−t/2))` (VPSA).
- **Blowdown** — Pressure follows a prescribed **first-order exponential decay** `P(t) = P_low + (P_high − P_low)·exp(−t/τ_bd)`. The corresponding `dP/dt` is fed into the total-moles balance to drive the gas-expansion velocity term.
- **Repressurization** — In VPSA, `P(t) = P_low + ((P_high − P_low)/t_rep)·t` is a **linear ramp**, giving a constant `dP/dt` in the velocity equation. PSA uses an exponential ramp via the same `tau` formulation as blowdown.

In every regeneration phase the pressure is an explicit function of time rather than a state variable — this collapses what would otherwise be a stiff differential-algebraic system into a plain ODE.

### 6.4 Time Integration — `scipy.integrate.solve_ivp` with BDF

The MOL system is **stiff** (LDF rate constants differ from convective time scales by 3–4 orders of magnitude), so an implicit method is mandatory. The two script sets use slightly different tolerance regimes:

```python
# PSA
solve_ivp(rhs, [t0, tf], y0, method='BDF',
          rtol=1e-2, atol=1e-2)                         # adsorption
solve_ivp(..., rtol=5e-2, atol=1e-4, first_step=1e-11)  # blowdown / purge

# VPSA
solve_ivp(..., rtol=1e-3, atol=1e-5)                    # adsorption / rinse
solve_ivp(..., rtol=1e-3, atol=1e-6)                    # blowdown
solve_ivp(..., rtol=1e-3, atol=1e-5, first_step=1e-6)   # repressurization
```

VPSA runs tighter because the inventory-difference mass-balance audit (Section 6.10) is sensitive to integrator drift. PSA can afford looser tolerances because its mass-balance metric uses a grounded inventory term that absorbs drift. The very small `first_step` values force BDF to take a tiny first step past the *t = 0* discontinuities (pressure ramp activation, ramped feed startup).

### 6.5 Sparse Jacobian Hint

A block sparsity pattern is precomputed and passed to BDF as `jac_sparsity`. The triangular structure flips depending on flow direction:

- **VPSA adsorption** uses a banded **lower-triangular** pattern — row `eq*N + j` depends on cells `0 … j+1`, reflecting forward-marching upwind:
  ```python
  for k in range(min(j + 2, N)):
      sparsity_matrix[eq*N + j, var*N + k] = 1
  ```
- **PSA blowdown / purge** uses an **upper-triangular** pattern — row `eq*N + j` depends on cells `j … N-1`, reflecting countercurrent (top-down) upwind:
  ```python
  sparsity_matrix[eq*N + j, var*N + j : var*N + N] = 1
  ```

BDF then computes a finite-difference Jacobian column-by-column **only at the nonzero entries** instead of evaluating the full `(6N × 6N)` (VPSA) or `(10N × 10N)` (PSA) dense matrix — an order-of-magnitude speedup at `N = 100`.

### 6.6 RHS Acceleration — Numba JIT

Every RHS that runs in a tight integration loop is decorated with `@njit(cache=True)` and **fully unrolled** by species — no `np.sum`, no broadcasting, just explicit scalar variables `raw_C_0, raw_C_1, raw_C_2, …`. The reason: BDF calls the RHS thousands of times per integration and vectorized NumPy allocates a fresh temporary array on each call; Numba compiles the unrolled loop into a single tight kernel that operates entirely on stack-allocated scalars. The `cache=True` flag persists the compiled artifact across runs, so only the very first invocation pays the ~5–10 s compilation cost.

### 6.7 Numerical Regularization

Several smoothing tricks keep the integrator stable across regime changes:

- **Mole-fraction renormalization** (VPSA) — at each cell, raw concentrations are converted to fractions `y_i = max(C_i,0) / max(ΣC, 1e-10)` and partial pressures are reconstructed as `y_i · P(t)`. This breaks a positive-feedback loop where a numerically negative `C_i` would otherwise inflate the local total pressure and ruin the isotherm evaluation.
- `leaky_max(x, ε, α)` (PSA) — a differentiable substitute for `max(x, ε)` that keeps a small gradient on the clipped side, preventing the BDF Jacobian from going singular when concentrations approach zero.
- **Sigmoid layer blend** `w(z) = 1 / (1 + exp(−10·(z − z_mid)))` (PSA only — VPSA is single-material) — replaces a step transition between bed layers with a smooth ramp over ~2 cells, eliminating spurious oscillations at the layer interface.
- **Tanh activation gate** on the LDF rate during PSA adsorption — turns uptake on smoothly as the local concentration crosses the trace threshold.
- **Soft startup ramps** — Ergun `(1 − exp(−t/5))` PSA / `(1 − exp(−t/2))` VPSA, plus a feed-composition ramp `feed_ramp = 1 − exp(−t/0.5)` in VPSA that bleeds the inlet from pure-N₂ residue into the actual flue-gas mix over the first ~1.5 s.
- **Bounded p-ratio** `np.clip(P_partial / P_atm, 1e-10, 1000)` — keeps `^n_param` away from `0^negative` and `large^large` in PSA's Sips evaluation.
- **Velocity clipping** (PSA blowdown) `np.clip(u, 1e-4, 5.0)` — keeps the integrator out of unphysical regions during the early transient.

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

### 6.10 Diagnostic Quadratures & Mass Balance

Two distinct strategies are used:

- **PSA — instantaneous-flow trapezoidal integration.** Tail-gas moles per species in blowdown and purge are computed by `np.trapz(species_flow, t_actual, axis=0)` over the BDF output grid. Adsorption recovery uses a grounded mass balance `moles_out_gross(t) = ∫₀ᵗ ṅ_in dτ − Δ(bed_inventory)` so integrator drift is absorbed into the inventory term.
- **VPSA — inventory differencing + analytic ramp integrals.** The blowdown audit computes `moles_vacuumed = max(initial_inventory − final_inventory, 0)` directly from the saved bed states, completely sidestepping the BDF time grid for cumulative quantities. The "fed CO₂" and "rinsed CO₂" totals needed for true cycle recovery are computed from the **closed-form integral** of the soft-start ramp:

  ```
  ∫₀^t_end  ṅ · (1 − exp(−τ/τ_ramp))  dτ  =  ṅ · ( t_end + τ_ramp · (exp(−t_end/τ_ramp) − 1) )
  ```

  This avoids trapezoid error on the early-time exponential and is what gets stored as `co2_moles_fed` / `co2_moles_rinse` in the npz handover files.

The VPSA blowdown reports **three independent recovery metrics** from the same audit:

1. **Purity** = `co2_moles_collected / total_moles_collected` — composition of the vacuum tail gas.
2. **Gross step recovery** = `co2_moles_collected / initial_co2_in_bed` — how much of the loaded CO₂ the vacuum actually pulled off (vacuum efficiency).
3. **Cycle recovery** = `(co2_moles_collected − co2_moles_rinse) / co2_moles_fed` — true capture rate, netted against the CO₂ recycled as rinse to avoid double-counting.

### 6.11 Phase-Specific Notes (Numerical)

- **Adsorption** — solves the full 10·N (PSA) or 6·N (VPSA) ODE system. PSA additionally evaluates a **Peng-Robinson cubic** (`np.roots` of the EOS polynomial) for the gas-phase compressibility factor at each RHS call, picking the largest real root (vapour-like). VPSA uses ideal gas throughout. Both use **extended Sips** (PSA) or **single-site Langmuir** (VPSA) isotherm equilibrium — note the VPSA arrays are still named `b_toth` for legacy reasons even though the actual `q_star = (b·P·q_s) / (1 + b·P)` form is Langmuir.
- **Rinse (VPSA)** — adsorption-style RHS with the inlet boundary set to `y_feed = [0, 1, 0]` (pure CO₂) at `P_mid`, and `u_feed_rinse` read directly from the config rather than back-calculated from a target molar flow.
- **Blowdown / Purge** — pressure trajectory becomes an explicit `t`-dependent forcing term, shrinking the effective state vector. VPSA blowdown also pre-allocates the `dqdt_*` arrays inside the JIT-compiled RHS to keep Numba from re-allocating on each call.
- **Repressurization** — same kernel as blowdown with the sign of `dP/dt` reversed; in VPSA the linear pressure ramp gives a constant compression source term, and pure N₂ is fed at the open product end only when the inlet velocity is inward (`v_inlet ≤ 0`).
- **CoD (VPSA, optional)** — cocurrent depressurization between rinse and blowdown; commented out in `master VPSA.py` line 126, enable by uncommenting `subprocess.run([..., cod_path], ...)`.

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
