import os
import json
from math import pi
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
from tqdm import tqdm
from numba import njit

warnings.filterwarnings('ignore')

# =============================================================================
# 1. SETUP
# =============================================================================
run_type = os.environ.get("RUN_TYPE", "CSS")
current_cycle = int(os.environ.get("PSA_CYCLE", 1))

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "results")
cycle_folder = os.path.join(base_dir, f"{run_type}_cycle_{current_cycle}")
os.makedirs(cycle_folder, exist_ok=True)

print(f"\n--- Starting Blowdown (CO2 Recovery) for [{run_type}] Cycle {current_cycle} ---")

# =============================================================================
# 2. LOAD PARAMETERS
# =============================================================================
config_path = os.path.join(script_dir, "master_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

tau_bd = config["tau_bd"]
t_blowdown_end = config["t_blowdown_end"]
t_eval_bd = np.linspace(0, t_blowdown_end, config.get("t_bd_steps", 150))

# Load the saturated bed from the Rinse step
state_file = os.path.join(script_dir, 'rinse_end_state.npz')
if not os.path.exists(state_file):
    raise FileNotFoundError(f"Cannot find {state_file}.")

state_data = np.load(state_file)

L = float(state_data['L'])
T = float(state_data['T'])
R = float(state_data['R'])
d = float(state_data['d'])
A = pi * (d / 2)**2
N = int(config["N"])
co2_moles_rinse = float(state_data['co2_moles_rinse'])
co2_moles_exhaust_rinse = float(state_data['co2_moles_exhaust_rinse']) if 'co2_moles_exhaust_rinse' in state_data.files else 0.0

eps = float(config.get("eps", 0.35))
rho_s = float(config.get("rho_s", 2000))
P_start = float(state_data['P_end'])
P_low = float(config["P_low"])
ads_state_file = os.path.join(script_dir, 'adsorption_end_state.npz')
ads_data = np.load(ads_state_file)
co2_moles_fed = float(ads_data['co2_moles_fed'])
co2_moles_exhaust_ads = float(ads_data['co2_moles_exhaust_ads']) if 'co2_moles_exhaust_ads' in ads_data.files else 0.0

C_initial = state_data['C_end']
q_initial = state_data['q_end']
y0_bd = np.concatenate([C_initial, q_initial])

labels = ['N2', 'CO2', 'O2']
colors = ['gray', 'blue', 'red']
MW = np.array([0.028014, 0.044009, 0.031998]) 

dz = L / N
z_nodes = np.linspace(dz/2, L - dz/2, N)
mass_transfer_coeff = ((1 - eps) / eps) * rho_s

# Toth Constants
k_ldf = np.array([[0.0021], [0.0143], [0.002]]).repeat(N, axis=1) 

k_qs_1 = np.array([[1.89],[2.82],[1e-8]]).repeat(N, axis=1)
k_qs_2 = np.array([[-2.25e-4],[-3.50e-4],[1e-8]]).repeat(N, axis=1)
q_s = k_qs_1 + k_qs_2 * T

b_0 = np.array([[1.16e-9], [2.83e-9], [1e-8]]).repeat(N, axis=1)
B_val = np.array([[1944.61], [2598.2], [1e-8]]).repeat(N, axis=1)
b_toth = b_0 * np.exp(B_val / T)

# Track exact moles in the bed BEFORE blowdown starts
initial_gas_moles = np.sum(C_initial[:3*N].reshape(3, N) * (A * dz * eps), axis=1)
initial_solid_moles = np.sum(q_initial[:3*N].reshape(3, N) * (A * dz * (1 - eps) * rho_s), axis=1)

# This is the 'initial_inventory' Pylance was looking for
initial_inventory = initial_gas_moles + initial_solid_moles
initial_co2_moles = initial_inventory[1]

# =============================================================================
# 3. PDE SOLVER (Optimized & Directionally Corrected)
# =============================================================================
@njit(cache=True)
def calc_rhs_blowdown(t, y, N, P_low, P_start, tau_bd, eps, rho_s, dz, k_ldf, q_s, b_toth, R, T):
    res = np.empty(6 * N)
    
    # 1. Pressure & Expansion Kinetics
    P_t = P_low + (P_start - P_low) * np.exp(-t / tau_bd)
    dPdt = -(P_start - P_low) / tau_bd * np.exp(-t / tau_bd)
    dCtot_dt_expansion = dPdt / (R * T)
    
    mass_transfer_coef = ((1 - eps) / eps) * rho_s
    
    sum_dqdt = np.zeros(N)
    dqdt_0 = np.zeros(N)
    dqdt_1 = np.zeros(N)
    dqdt_2 = np.zeros(N)
    
    # 2. Mass Transfer Rates (LDF)
    for j in range(N):
        raw_C_0 = y[0 * N + j]
        raw_C_1 = y[1 * N + j]
        raw_C_2 = y[2 * N + j]
        
        C_tot_raw = max(raw_C_0 + raw_C_1 + raw_C_2, 1e-10)
        y_frac_0 = max(raw_C_0, 0.0) / C_tot_raw
        y_frac_1 = max(raw_C_1, 0.0) / C_tot_raw
        y_frac_2 = max(raw_C_2, 0.0) / C_tot_raw
        
        P_0_kPa = (y_frac_0 * P_t)
        P_1_kPa = (y_frac_1 * P_t)
        P_2_kPa = (y_frac_2 * P_t)
        
        denom_0 = 1.0 + b_toth[0, j] * P_0_kPa
        denom_1 = 1.0 + b_toth[1, j] * P_1_kPa
        denom_2 = 1.0 + b_toth[2, j] * P_2_kPa
        
        q_star_0 = (b_toth[0, j] * P_0_kPa * q_s[0, j]) / denom_0
        q_star_1 = (b_toth[1, j] * P_1_kPa * q_s[1, j]) / denom_1
        q_star_2 = (b_toth[2, j] * P_2_kPa * q_s[2, j]) / denom_2
        
        dq0 = k_ldf[0, j] * (q_star_0 - y[3 * N + j])
        dq1 = k_ldf[1, j] * (q_star_1 - y[4 * N + j])
        dq2 = k_ldf[2, j] * (q_star_2 - y[5 * N + j])
        
        dqdt_0[j] = dq0
        dqdt_1[j] = dq1
        dqdt_2[j] = dq2
        
        res[3 * N + j] = dq0
        res[4 * N + j] = dq1
        res[5 * N + j] = dq2
        
        sum_dqdt[j] = dq0 + dq1 + dq2

    # 3. Velocity Integration (Anchored at 0 at the closed ceiling, flowing DOWN)
    C_tot_theo = P_t / (R * T)
    v_local = np.zeros(N)
    
    dv_dz_top = - (dCtot_dt_expansion + mass_transfer_coef * sum_dqdt[N-1]) / C_tot_theo
    # v(z=L)=0 at the closed wall; cell N-1 is centered half a step below the wall
    v_local[N-1] = -dv_dz_top * (dz / 2.0)
    
    for j in range(N-2, -1, -1):
        dv_dz_j = - (dCtot_dt_expansion + mass_transfer_coef * sum_dqdt[j]) / C_tot_theo
        v_local[j] = v_local[j+1] - dv_dz_j * dz
        
    # 4. IMMUNE DYNAMIC UPWINDING
    F_face_0 = np.zeros(N + 1)
    F_face_1 = np.zeros(N + 1)
    F_face_2 = np.zeros(N + 1)

    # Calculate fluxes at every cell face
    for j in range(N):
        v_face = v_local[j]
        
        if v_face >= 0.0:
            # Flowing UPWARDS (only happens during solver probing)
            if j == 0:
                C_up_0, C_up_1, C_up_2 = 0.0, 0.0, 0.0 # Vacuum backflow assumption
            else:
                C_up_0 = y[0 * N + j - 1]
                C_up_1 = y[1 * N + j - 1]
                C_up_2 = y[2 * N + j - 1]
        else:
            # Flowing DOWNWARDS (Standard Blowdown behavior)
            C_up_0 = y[0 * N + j]
            C_up_1 = y[1 * N + j]
            C_up_2 = y[2 * N + j]

        F_face_0[j] = v_face * C_up_0
        F_face_1[j] = v_face * C_up_1
        F_face_2[j] = v_face * C_up_2

    # Ceiling face (z=L) is always closed
    F_face_0[N] = 0.0
    F_face_1[N] = 0.0
    F_face_2[N] = 0.0

    # 5. Flux Balance
    for j in range(N):
        F_bottom_0 = F_face_0[j]
        F_bottom_1 = F_face_1[j]
        F_bottom_2 = F_face_2[j]
        
        F_top_0 = F_face_0[j+1]
        F_top_1 = F_face_1[j+1]
        F_top_2 = F_face_2[j+1]
            
        res[0 * N + j] = -((F_top_0 - F_bottom_0) / dz) - mass_transfer_coef * dqdt_0[j]
        res[1 * N + j] = -((F_top_1 - F_bottom_1) / dz) - mass_transfer_coef * dqdt_1[j]
        res[2 * N + j] = -((F_top_2 - F_bottom_2) / dz) - mass_transfer_coef * dqdt_2[j]
        
    return res

print(f"\n--- Solving Blowdown phase ({P_start/1e5:.2f} bar to {P_low/1e5:.2f} bar) ---")
pbar_bd = tqdm(total=t_blowdown_end, desc="Blowdown", unit="s", bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]")
last_t_bd = [0.0] 

def pde_blowdown(t, y): 
    if t > last_t_bd[0]:
        pbar_bd.update(t - last_t_bd[0])
        last_t_bd[0] = t
    return calc_rhs_blowdown(t, y, N, P_low, P_start, tau_bd, eps, rho_s, dz, k_ldf, q_s, b_toth, R, T)

sol_bd = solve_ivp(pde_blowdown, [t_eval_bd[0], t_eval_bd[-1]], y0_bd, method='BDF',
                   t_eval=t_eval_bd, rtol=1e-3, atol=1e-4, first_step=0.01)
pbar_bd.close() 

#=============================================================================
# 4. TAIL GAS INTEGRATION (Perfect Mass Balance)
# =============================================================================
actual_steps_bd = sol_bd.y.shape[1]
C_history_bd = sol_bd.y[:3*N, :].T.reshape((actual_steps_bd, 3, N))
q_history_bd = sol_bd.y[3*N:, :].T.reshape((actual_steps_bd, 3, N))
t_actual_bd = sol_bd.t
species_flow_bd = np.zeros((actual_steps_bd, 3))
P_exit_bd = np.zeros(actual_steps_bd)

for i, t in enumerate(t_actual_bd):
    P_t = P_low + (P_start - P_low) * np.exp(-t / tau_bd)
    dPdt = -(P_start - P_low) / tau_bd * np.exp(-t / tau_bd)
    dCtot_dt_expansion = dPdt / (R * T)
    P_exit_bd[i] = P_t / 1e5
    
    C_at_t = C_history_bd[i]
    C_safe = np.maximum(C_at_t, 1e-12)
    C_tot_actual = np.sum(C_safe, axis=0)
    
    sum_dqdt = np.sum(k_ldf * (((b_toth * ((C_safe / C_tot_actual) * P_t) * q_s) / (1.0 + b_toth * ((C_safe / C_tot_actual) * P_t))) - q_history_bd[i]), axis=0)
    dv_dz_viz = - (dCtot_dt_expansion + mass_transfer_coeff * sum_dqdt) / (P_t / (R * T))
    
    v_local_viz = np.zeros(N)
    # v(z=L)=0 at the closed wall; cell N-1 is centered half a step below the wall
    v_local_viz[N-1] = -dv_dz_viz[N-1] * (dz / 2.0)
    for j in range(N-2, -1, -1):
        v_local_viz[j] = v_local_viz[j+1] - dv_dz_viz[j] * dz
    
    u_exit = abs(v_local_viz[0] * eps)  
    species_flow_bd[i, :] = u_exit * A * C_at_t[:, 0]

# --- 2. The Source of Truth: Inventory-Based Mass Balance ---
C_final = sol_bd.y[:3*N, -1].reshape(3, N)
q_final = sol_bd.y[3*N:, -1].reshape(3, N)

# Exact moles in column at the very last second of simulation
final_inventory = np.sum(C_final * (A * dz * eps), axis=1) + \
                  np.sum(q_final * (A * dz * (1 - eps) * rho_s), axis=1)

# Difference between start and end (initial_inventory calculated at start of script)
moles_vacuumed_exact = np.maximum(initial_inventory - final_inventory, 0.0)
co2_moles_collected = moles_vacuumed_exact[1] 
total_moles_collected = np.sum(moles_vacuumed_exact)

# --- 3. The Three Evaluation Tiers ---
# A. Purity
co2_purity = (co2_moles_collected / total_moles_collected) * 100 if total_moles_collected > 0 else 0.0

# B. Bed Sweep Efficiency (fraction of CO2 in the bed at start of blowdown
# that is actually vacuumed out — a working-capacity / vacuum-sweep metric,
# not a true recovery: the denominator includes pure CO2 added by the rinse
# step, so this is typically below cycle_recovery and that is expected.)
bed_sweep_efficiency = (co2_moles_collected / initial_co2_moles) * 100 if initial_co2_moles > 0 else 0.0


# C. True Cycle Recovery (Net new CO2 captured vs Flue Gas input)
# Use the exhaust-side mass balance, which is conserved cycle-to-cycle:
#     fresh CO2 in (ads) = CO2 out (ads exhaust) + CO2 out (rinse exhaust) + product
# The earlier formula (collected − rinse_in) relied on the bed-inventory snapshot
# at the start of blowdown, which has small numerical drift that gets amplified
# because rinse_in ≫ fresh_in — that's how cycle_recovery was breaking 100%.
net_co2_produced = max(co2_moles_fed - co2_moles_exhaust_ads - co2_moles_exhaust_rinse, 0.0)
cycle_recovery = (net_co2_produced / co2_moles_fed) * 100 if co2_moles_fed > 0 else 0.0


# =============================================================================
# 5. SUMMARY TXT EXPORT
# =============================================================================
summary_path = os.path.join(cycle_folder, "desorption_summary.txt")
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("\n" + "="*55 + "\n")
    f.write(f"{'VPSA CYCLE PERFORMANCE AUDIT':^55}\n")
    f.write("="*55 + "\n")
    f.write(f"Cycle: {current_cycle}\n")
    f.write(f"CO2 Purity:           {co2_purity:>10.2f} %\n")
    f.write("-" * 55 + "\n")
    f.write(f"Bed Sweep Efficiency: {bed_sweep_efficiency:>10.2f} % (CO2 vacuumed / CO2 in bed at start of blowdown)\n")
    f.write(f"Cycle Recovery:       {cycle_recovery:>10.2f} % (Net fresh CO2 captured / fresh CO2 fed)\n")
    f.write("-" * 55 + "\n")
    f.write(f"Fresh CO2 Fed:        {co2_moles_fed:>10.2e} mol\n")
    f.write(f"CO2 Slipped (Ads):    {co2_moles_exhaust_ads:>10.2e} mol\n")
    f.write(f"CO2 Slipped (Rinse):  {co2_moles_exhaust_rinse:>10.2e} mol\n")
    f.write(f"Pure CO2 Rinsed In:   {co2_moles_rinse:>10.2e} mol\n")
    f.write(f"Total CO2 Vacuumed:   {co2_moles_collected:>10.2e} mol\n")
    f.write(f"NET CO2 PRODUCED:     {net_co2_produced:>10.2e} mol\n")
    f.write("="*55 + "\n\n")

print(f"\n--- PERFORMANCE AUDIT RESULTS ---")
print(f"Purity:           {co2_purity:.2f}%")
print(f"Bed Sweep Eff.:   {bed_sweep_efficiency:.2f}% (CO2 vacuumed / CO2 in bed at start of BD)")
print(f"Cycle Recovery:   {cycle_recovery:.2f}% (Net fresh CO2 captured / fresh CO2 fed)")


# =============================================================================
# 6. VISUALIZATION & STATE SAVING
# =============================================================================
C_plot_combined = np.maximum(C_history_bd, 1e-8) 
y_history_combined = C_plot_combined / np.sum(C_plot_combined, axis=1, keepdims=True)

fig_metrics, axes_metrics = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)
axes_metrics[0].plot(t_actual_bd, np.sum(species_flow_bd, axis=1), color='orange', linewidth=2.5)
axes_metrics[0].set_title('A. Blowdown: Total Flow at Exit (z=0)')
axes_metrics[0].set_xlabel('Time (s)')
axes_metrics[0].set_ylabel('Flow Rate (mol/s)')
axes_metrics[0].grid(True)

axes_metrics[1].plot(t_actual_bd, P_exit_bd, color='black', linestyle='--', label='Exit (z=0)')
axes_metrics[1].set_title('B. Blowdown: Pressure Profile')
axes_metrics[1].set_xlabel('Time (s)')
axes_metrics[1].set_ylabel('Pressure (bar)')
axes_metrics[1].legend()
axes_metrics[1].grid(True)

fig2d, axes2d = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)
axes_flat = axes2d.flatten()

ax_elution = axes_flat[0]
for i in range(3): 
    ax_elution.plot(t_actual_bd, y_history_combined[:, i, 0] * 100, label=labels[i], color=colors[i], linewidth=2.5)

ax_elution.set(title='Blowdown Elution Curve (z=0)', xlabel='Time (s)', ylabel='Molar Fraction (%)')
ax_elution.legend()
ax_elution.grid(True)

target_times = [0.0, t_actual_bd[-1]*0.25, t_actual_bd[-1]*0.5, t_actual_bd[-1]*0.75, t_actual_bd[-1]]
time_indices = [int(np.argmin(np.abs(t_actual_bd - t_sec))) for t_sec in target_times]
time_cmap = plt.cm.plasma(np.linspace(0, 0.8, len(time_indices)))

for i in range(3):
    ax_sp = axes_flat[i+1]
    for t_step_idx, t_data_idx in enumerate(time_indices):
        if t_data_idx < len(t_actual_bd): 
            ax_sp.plot(z_nodes, q_history_bd[t_data_idx, i, :], 
                       label=f't={t_actual_bd[t_data_idx]:.0f}s', color=time_cmap[t_step_idx], linewidth=2)
            
    ax_sp.set(title=f'Spatial Profile: {labels[i]}', xlabel='Column Length z (m)', ylabel='Solid Loading q')
    ax_sp.legend(fontsize=8)
    ax_sp.grid(True, alpha=0.5)
    
metrics_path = os.path.join(cycle_folder, "des_flow_pressure_metrics.png")
fig_metrics.savefig(metrics_path)
profiles_path = os.path.join(cycle_folder, "des_spatial_profiles.png")
fig2d.savefig(profiles_path)

# SAVING THE CLEAN STATE
des_state_file = os.path.join(script_dir, 'desorption_end_state.npz')
np.savez(des_state_file, 
         C_end=sol_bd.y[:3*N, -1], 
         q_end=sol_bd.y[3*N:, -1])

print(f"✅ Blowdown state saved to {des_state_file}")
plt.close('all')
os._exit(0)
