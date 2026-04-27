import os
import json
from math import pi
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.sparse as sp 
import warnings
from tqdm import tqdm
from numba import njit 

warnings.filterwarnings('ignore')

# =============================================================================
# 1. RUN ROUTER & PATHS
# =============================================================================
run_type = os.environ.get("RUN_TYPE", "CSS") 
current_cycle = os.environ.get("PSA_CYCLE", "1")

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "results")
cycle_folder = os.path.join(base_dir, f"{run_type}_cycle_{current_cycle}")
os.makedirs(cycle_folder, exist_ok=True)

print(f"\n--- Starting Adsorption: [{run_type} MODE] Cycle {current_cycle} ---")

config_path = os.path.join(script_dir, "master_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

L = config["L"]; T = config["T"]; R = config["R"]
P_high = config["P_high"]; P_low = config["P_low"]
d = config["d"]; Nsets = config["Nsets"]

# FIXED: Defaults to 0.0 if the purge step was removed from the master config
purge_fraction = float(config.get("purge_fraction", 0.0))
t_ads_safety_ratio = float(config.get("t_ads_safety_ratio", 0.90))

N = int(config["N"])
dp = float(config.get("dp", 0.002))
mu = float(config.get("mu", 1.135086e-05))
P_atm_Pa = float(config.get("P_atm_Pa", 101325.0))

eps = float(config.get("eps", 0.35))
rho_s = float(config.get("rho_s", 2000))

if run_type in ["SCOUT", "FINAL"]:
    t_end = config["t_ads_end"] 
else:
    t_end = config.get("t_op_ads", 400) 
t_ads_eval = np.linspace(0, t_end, config.get("t_ads_steps", 500))

# =============================================================================
# 2. FEED SIZING & SYSTEM PARAMETERS
# =============================================================================
A = pi*(d/2)**2 
labels = ['N2', 'CO2', 'O2']
colors = ['gray', 'blue', 'red']
MW = np.array([0.028014, 0.044009, 0.031998])
y_feed = np.array([0.804585, 0.1006, 0.0304]) 
y_feed /= np.sum(y_feed)

feed_molar_flow = float(config.get("feed_molar_flow", 450.0)) 

# --- IDEAL GAS LAW USED HERE ---
C_in_total = P_high / (R * T)
u_feed = feed_molar_flow / (A * C_in_total) 
C_in = y_feed * C_in_total


# --- NEW DIAGNOSTIC READOUT ---
print(f"\n--- FEED DIAGNOSTICS ---")
print(f"Target Feed Flow:     {feed_molar_flow:.2f} mol/s")
print(f"Column Cross Section: {A:.4f} m^2")
print(f"Inlet Gas Density:    {C_in_total:.2f} mol/m^3")
print(f"-> INLET SUPERFICIAL VELOCITY: {u_feed:.4f} m/s <-")
if u_feed > 1.3:
    print(f"⚠️ WARNING: Velocity is > 1.3 m/s! Bed Fluidisized.")
print(f"------------------------\n")

dz = L / N
z_nodes = np.linspace(dz/2, L - dz/2, N) 

k_ldf = np.array([[0.0021], [0.0143], [0.002]]).repeat(N, axis=1) 

k_qs_1 = np.array([[1.89],[2.82],[1e-8]]).repeat(N, axis=1)
k_qs_2 = np.array([[-2.25e-4],[-3.50e-4],[1e-8]]).repeat(N, axis=1)
q_s = k_qs_1 + k_qs_2 * T    

b_0 = np.array([[1.16e-9], [2.83e-9], [1e-8]]).repeat(N, axis=1)  
B_val = np.array([[1944.61], [2598.2], [1e-8]]).repeat(N, axis=1)                 
b_toth = b_0 * np.exp(B_val / T)

a_ergun_arr = (150 * (1 - eps)**2) / (4 * (dp/2)**2 * eps**3) * np.ones(N)
b_ergun_arr = (1.75 * (1 - eps)) / (2 * (dp/2) * eps**3) * np.ones(N)

# =============================================================================
# 3. PDE SOLVER LOGIC (Fully Unrolled for Maximum Speed)
# =============================================================================
@njit(cache=True)
def calc_rhs(t, y, N, P_low, P_high, P_atm_Pa, u_feed, eps, rho_s, dz, MW, Rp, mu, y_feed, k_ldf, q_s, b_toth, R, T, a_ergun_arr, b_ergun_arr):
    res = np.empty(6 * N) 
    current_P = P_high
    current_u = u_feed
    # At the top of calc_rhs, replace your static flux_in with a ramped version:
    feed_ramp = 1.0 - np.exp(-t / 0.5) # Reaches ~95% after 1.5 seconds

    flux_in_0 = (current_u / eps) * (y_feed[0] * feed_ramp + (1 - feed_ramp)) * (current_P / (R * T)) # Starts as pure N2
    flux_in_1 = (current_u / eps) * (y_feed[1] * feed_ramp) * (current_P / (R * T))
    flux_in_2 = (current_u / eps) * (y_feed[2] * feed_ramp) * (current_P / (R * T))
    
    mass_transfer_coef = ((1 - eps) / eps) * rho_s

    for j in range(N):
        raw_C_0 = y[0 * N + j]
        raw_C_1 = y[1 * N + j]
        raw_C_2 = y[2 * N + j]
        
        # --- THE FIX: Break the Concentration-Pressure Feedback Loop ---
        # Normalize to mole fractions to prevent runaway partial pressures
        C_tot_raw = max(raw_C_0 + raw_C_1 + raw_C_2, 1e-10)
        y_0 = max(raw_C_0, 0.0) / C_tot_raw
        y_1 = max(raw_C_1, 0.0) / C_tot_raw
        y_2 = max(raw_C_2, 0.0) / C_tot_raw
        
        # 1. Calculate partial pressures using the STABLE Ergun pressure
        P_0_kPa = (y_0 * current_P)
        P_1_kPa = (y_1 * current_P)  
        P_2_kPa = (y_2 * current_P) 
        # 2. Toth Isotherm (with absolute value on the base to prevent NaN in exponent)
        # Langmuir Isotherm 
        denom_0 = 1.0 + (b_toth[0, j] * P_0_kPa)
        denom_1 = 1.0 + (b_toth[1, j] * P_1_kPa)
        denom_2 = 1.0 + (b_toth[2, j] * P_2_kPa)
        
        q_star_0 = (b_toth[0, j] * P_0_kPa * q_s[0, j]) / denom_0
        q_star_1 = (b_toth[1, j] * P_1_kPa * q_s[1, j]) / denom_1
        q_star_2 = (b_toth[2, j] * P_2_kPa * q_s[2, j]) / denom_2
        
        dqdt_0 = k_ldf[0, j] * (q_star_0 - y[3 * N + j])
        dqdt_1 = k_ldf[1, j] * (q_star_1 - y[4 * N + j])
        dqdt_2 = k_ldf[2, j] * (q_star_2 - y[5 * N + j])
        
        res[3 * N + j] = dqdt_0
        res[4 * N + j] = dqdt_1
        res[5 * N + j] = dqdt_2
        
        sum_dqdt = dqdt_0 + dqdt_1 + dqdt_2
        
        # 7. Velocity Update
        du = -dz * ((1 - eps) * rho_s * sum_dqdt / (current_P / (R * T)))
        next_u = current_u + du
        # Floor the velocity to prevent numerical collapse at the adsorption front.
        # Without this clamp, when local adsorption demand exceeds feed supply,
        # next_u drops to zero/negative and gas piles up unphysically in 1-2 cells.
        u_floor = 0.05 * u_feed
        if next_u < u_floor:
            next_u = u_floor
        
        # 8. Gas Phase Balances (DYNAMIC UPWINDING)
        # Check flow direction to prevent pushing negative concentrations forward
        if next_u >= 0.0:
            upwind_C_0 = raw_C_0
            upwind_C_1 = raw_C_1
            upwind_C_2 = raw_C_2
        else:
            if j < N - 1:
                upwind_C_0 = y[0 * N + j + 1]  # Pull from the cell ahead
                upwind_C_1 = y[1 * N + j + 1]
                upwind_C_2 = y[2 * N + j + 1]
            else:
                upwind_C_0 = raw_C_0
                upwind_C_1 = raw_C_1
                upwind_C_2 = raw_C_2
        
        flux_out_0 = (next_u / eps) * upwind_C_0 
        flux_out_1 = (next_u / eps) * upwind_C_1 
        flux_out_2 = (next_u / eps) * upwind_C_2 
        
        res[0 * N + j] = -((flux_out_0 - flux_in_0) / dz) - mass_transfer_coef * dqdt_0
        res[1 * N + j] = -((flux_out_1 - flux_in_1) / dz) - mass_transfer_coef * dqdt_1
        res[2 * N + j] = -((flux_out_2 - flux_in_2) / dz) - mass_transfer_coef * dqdt_2
        
        flux_in_0 = flux_out_0
        flux_in_1 = flux_out_1
        flux_in_2 = flux_out_2
        
        # 9. Pressure Update
        if j < N - 1:
            ergun_ramp = (1.0 - np.exp(-t / 2.0))
            
            # Protect gas density from numerical concentration spikes
            rho_gas = (y_0 * MW[0] + y_1 * MW[1] + y_2 * MW[2]) * (current_P / (R * T))
            
            dPdz = - (a_ergun_arr[j] * mu * current_u + b_ergun_arr[j] * rho_gas * current_u * abs(current_u)) * ergun_ramp
            next_P = current_P + dPdz * dz
            if next_P < P_low * 0.1: next_P = P_low * 0.1
            current_P = next_P
            
        current_u = next_u
        
    return res

pbar = tqdm(total=t_end, desc="Simulation Progress", unit="s")
last_t = [0.0]
def pde_layered(t, y):
    if t > last_t[0]: pbar.update(t - last_t[0]); last_t[0] = t
    return calc_rhs(t, y, N, P_low, P_high, P_atm_Pa, u_feed, eps, rho_s, dz, MW, dp/2, mu, y_feed, k_ldf, q_s, b_toth, R, T, a_ergun_arr, b_ergun_arr)

rep_state_file = os.path.join(script_dir, 'repressurization_end_state.npz')
if os.path.exists(rep_state_file) and run_type != "SCOUT":
    data = np.load(rep_state_file)
    y0 = np.concatenate([data['C_end'].flatten(), data['q_end'].flatten()])
    print("-> Loading bed state from Repressurization.")
else:
    print("-> Initializing clean bed with pure N2 environment.")
    y_pure = np.array([1.0, 1e-10, 1e-10])
    C_pure_total = P_high / (R * T) # Ideal Gas Law
    C_init = np.tile(y_pure * C_pure_total, (N, 1)).T
    
    P_partial_init_Pa = np.array([y_pure[i] * P_high * np.ones(N) for i in range(3)])
    q_init = np.zeros((3, N))
    for j in range(N):
        P_i_kPa = np.maximum(P_partial_init_Pa[:, j], 1e-12)
        for i in range(3):
            denom = 1.0 + (b_toth[i, j] * P_i_kPa[i])
            q_init[i, j] = (b_toth[i, j] * P_i_kPa[i] * q_s[i, j]) / denom

    y0 = np.concatenate([C_init.flatten(), q_init.flatten()])

sol = solve_ivp(pde_layered, [0, t_end], y0, method='BDF', t_eval=t_ads_eval, rtol=1e-3, atol=1e-4, first_step=0.01)
print("\n✅ Integration complete! Generating plots and saving data...")
pbar.close()

# =============================================================================
# 5. POST-PROCESSING (Exhaust Consistency Metrics)
# =============================================================================
t = sol.t; n_t = len(t)
C_history = sol.y[:3*N, :].T.reshape((-1, 3, N))

mass_flow_out_exhaust_gross = np.zeros(n_t)
mass_flow_out_exhaust_net = np.zeros(n_t)
mass_flow_purge_exhaust = np.zeros(n_t) 

for i in range(1, n_t):
    C_at_t = C_history[i]
    C_total_exit = np.sum(C_at_t[:, -1])
    
    inst_gross_flow = (u_feed * (1.0 - np.exp(-t[i]/2.0))) * A * C_total_exit * np.mean(MW)
    
    mass_flow_out_exhaust_gross[i] = inst_gross_flow
    mass_flow_out_exhaust_net[i] = inst_gross_flow * (1.0 - purge_fraction)
    mass_flow_purge_exhaust[i] = inst_gross_flow * purge_fraction

# =============================================================================
# 6. DIAGNOSTIC PLOTS
# =============================================================================
fig, ax_mass = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax_mass.plot(t, np.full_like(t, u_feed * A * C_in_total * np.mean(MW)), 'k--', label='Total Feed Capacity')
ax_mass.plot(t, mass_flow_out_exhaust_gross, 'gray', linestyle=':', label='Gross Exhaust Flow')
ax_mass.plot(t, mass_flow_out_exhaust_net, 'r-', linewidth=2, label='Net Exhaust Flow (Raffinate)')
ax_mass.set_title(f'[{run_type}] Instantaneous Mass Flow Rates'); ax_mass.set_ylabel('Mass Flow (kg/s)'); ax_mass.legend(); ax_mass.grid(True)
fig.savefig(os.path.join(cycle_folder, "ads_performance.png"))

if run_type == "SCOUT":
    fig_break, ax_break = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for i in range(3):
        ax_break.plot(t, np.maximum(C_history[:, i, -1]/C_in[i], 0.0), label=labels[i], color=colors[i], linewidth=2.5)
    ax_break.set_title('Gas Breakthrough Curve (z = L)')
    ax_break.set_xlabel('Time (s)'); ax_break.set_ylabel('C/C0'); ax_break.set_xlim(0, t_end); ax_break.legend(); ax_break.grid(True)
    fig_break.savefig(os.path.join(cycle_folder, "ads_breakthrough_curve.png"))

if run_type != "SCOUT":
    fig2d, axes2d = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes_flat = axes2d.flatten()
    
    ax_break = axes_flat[0]
    for i in range(3):
        ax_break.plot(t, np.maximum(C_history[:, i, -1]/C_in[i], 0.0), label=labels[i], color=colors[i], linewidth=2.5)
    ax_break.set_title('A. Gas Breakthrough Curve (z = L)')
    ax_break.set_xlabel('Time (s)'); ax_break.set_ylabel('C/C0'); ax_break.set_xlim(0, t_end); ax_break.legend(); ax_break.grid(True)

    time_points_sec = [0.4*t_end, 0.6*t_end, 0.8*t_end, 1.0*t_end]
    time_indices = [int(np.argmin(np.abs(t - t_sec))) for t_sec in time_points_sec]
    cmap_lines = plt.cm.plasma(np.linspace(0, 0.8, len(time_indices)))
    
    for i in range(3):
        ax = axes_flat[i+1]
        for idx, t_idx in enumerate(time_indices):
            y_local = C_history[t_idx, i, :] / np.maximum(np.sum(C_history[t_idx], axis=0), 1e-10)
            ax.plot(z_nodes, y_local, label=f't={t[t_idx]:.0f}s', color=cmap_lines[idx], linewidth=2)
        ax.set_title(f'Molar Frac: {labels[i]}')
        ax.set_xlabel('Column Length z (m)'); ax.set_ylabel('Mol Fraction'); ax.legend(fontsize=8); ax.grid(True, alpha=0.5); ax.set_xlim(0, L)

    fig2d.savefig(os.path.join(cycle_folder, "ads_profiles_2d.png"))

    def get_pressure_profile(idx):
        t_val = t[idx]
        C_at_t = C_history[idx]
        P_total = np.zeros(N)
        P_total[0] = P_high
        current_u = u_feed
        ergun_ramp = (1.0 - np.exp(-t_val / 2.0))
        for j in range(N - 1):
            C_safe = np.maximum(C_at_t, 1e-12)
            rho_gas = np.sum(C_safe[:, j] * MW)
            dPdz = - (a_ergun_arr[j] * mu * current_u + b_ergun_arr[j] * rho_gas * current_u**2) * ergun_ramp
            P_total[j+1] = max(P_total[j] + dPdz * dz, P_low * 0.9)
        return P_total

    fig_p, ax_p = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for idx, t_idx in enumerate(time_indices):
        P_profile = get_pressure_profile(t_idx)
        ax_p.plot(z_nodes, P_profile / 1e5, label=f't={t[t_idx]:.0f}s', color=cmap_lines[idx], linewidth=2.5)

    ax_p.set_title('Ergun Pressure Profile Along the Column', fontsize=14, fontweight='bold')
    ax_p.set_xlabel('Column Length z (m)', fontsize=12); ax_p.set_ylabel('Pressure (bar)', fontsize=12)
    ax_p.legend(); ax_p.grid(True, linestyle='--', alpha=0.6); ax_p.set_xlim(0, L)
    fig_p.savefig(os.path.join(cycle_folder, "ads_pressure_profile.png"))

# =============================================================================
# 7. REPORTING & STATE SAVING
# =============================================================================
eval_idx = -1

initial_inventory = np.zeros(3)
final_inventory = np.zeros(3)
moles_in_total = np.zeros(3)

C_init_reshaped = y0[:3*N].reshape((3, N))
q_init_reshaped = y0[3*N:].reshape((3, N))

C_final_reshaped = sol.y[:3*N, eval_idx].reshape((3, N))
q_final_reshaped = sol.y[3*N:, eval_idx].reshape((3, N))

tau_ramp = 0.5
ramp_integral = t_end + tau_ramp * (np.exp(-t_end / tau_ramp) - 1.0)
for i in range(3):
    initial_inventory[i] = np.sum(C_init_reshaped[i, :] * dz * A * eps) + np.sum(q_init_reshaped[i, :] * dz * A * (1 - eps) * rho_s)
    final_inventory[i] = np.sum(C_final_reshaped[i, :] * dz * A * eps) + np.sum(q_final_reshaped[i, :] * dz * A * (1 - eps) * rho_s)
    moles_in_total[i] = y_feed[i] * C_in_total * u_feed * A * ramp_integral
    
moles_exhaust_gross = np.maximum(moles_in_total - (final_inventory - initial_inventory), 0.0)

total_gross_exhaust = np.sum(moles_exhaust_gross)
exhaust_mix_pct = (moles_exhaust_gross / total_gross_exhaust) * 100 if total_gross_exhaust > 0 else np.zeros(3)

print(f"\n--- EXHAUST (RAFFINATE) CONSISTENCY RESULTS ---")
for i in range(3):
    print(f"{labels[i]:<4}: {exhaust_mix_pct[i]:>6.2f}% | Slipped Moles: {moles_exhaust_gross[i]:.3e}")

if run_type == "SCOUT":
    C_CO2_exit = C_history[:, 1, -1]
    purity_over_time = np.zeros(n_t)
    for i in range(1, n_t):
        C_total_exit = np.sum(C_history[i, :, -1])
        purity_over_time[i] = (C_CO2_exit[i] / C_total_exit) * 100 if C_total_exit > 1e-9 else 0.0
        
    valid = np.where(purity_over_time >= 0.1)[0] 
    breakthrough_time = t[valid[-1]] if len(valid) > 0 else t_end
    config["t_op_ads"] = float(breakthrough_time * t_ads_safety_ratio)
    
    t_tot = config["t_op_ads"] / config.get("Adsorption_Ratio", 0.5)
    config["t_tot"] = t_tot # <--- ADD THIS LINE
    
    config["tf_des"] = config.get("Purge_Ratio", 0.0) * t_tot
    config["t_blowdown_end"] = config.get("Blowdown_Ratio", 0.1) * t_tot
    config["t_rep"] = config.get("Repress_Ratio", 0.05) * t_tot
    config["t_rinse"] = config.get("Rinse_Ratio", 0.0) * t_tot
    
    with open(config_path, "w") as f: json.dump(config, f, indent=4)

elif run_type in ["CSS", "FINAL"]: 
    
    with open(os.path.join(cycle_folder, "adsorption_summary.txt"), "w") as f:
        f.write("\n" + "="*55 + "\n")
        f.write(f"{'RAFFINATE (EXHAUST) CONSISTENCY SUMMARY':^55}\n")
        f.write("="*55 + "\n")
        f.write(f"Cycle: {current_cycle}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'Species':<10} | {'Slipped Moles':<15} | {'Mix %':<10}\n")
        f.write("-" * 55 + "\n")
        for i in range(3):
            f.write(f"{labels[i]:<10} | {moles_exhaust_gross[i]:<15.3e} | {exhaust_mix_pct[i]:<10.2f}\n")
        f.write("="*55 + "\n\n")
    
    # Ensure t_tot exists: try config value, otherwise compute from t_op_ads and Adsorption_Ratio, fallback to t_end
    t_tot = config.get("t_tot", None)
    if t_tot is None:
        t_op_ads_val = config.get("t_op_ads", t_end)
        t_tot = t_op_ads_val / config.get("Adsorption_Ratio", 0.5)

    tau_ramp = 0.5
    co2_moles_fed_exact = y_feed[1] * feed_molar_flow * (t_end + tau_ramp * (np.exp(-t_end / tau_ramp) - 1.0))
    
    ads_state_file = os.path.join(script_dir, 'adsorption_end_state.npz')
    np.savez(ads_state_file,
             L=L, T=T, R=R, P_end=P_high, d=d,
             C_end=sol.y[:3*N, eval_idx],
             q_end=sol.y[3*N:, eval_idx],
             final_mass_flow_purge=mass_flow_purge_exhaust[eval_idx],
             t_tot=t_tot,
             co2_moles_fed=co2_moles_fed_exact,
             co2_moles_exhaust_ads=float(moles_exhaust_gross[1]),
             ads_end_inventory=final_inventory) # <--- Pass exact moles
    
    print(f"✅ Adsorption State Saved.")

plt.close('all'); os._exit(0)