import os
from math import pi
import numpy as np
import matplotlib
import json

# --- 1. THE MATPLOTLIB SILVER BULLET MUST GO HERE ---
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
import scipy.sparse as sp 
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# HELPER FUNCTIONS FOR SOLVER STABILITY
# =============================================================================
def leaky_max(x, eps=1e-10, alpha=1e-6):
    return np.where(x > eps, x, eps + alpha * (x - eps))

# =============================================================================
# 2. CYCLE DETECTION & ABSOLUTE PATHS
# =============================================================================
run_type = os.environ.get("RUN_TYPE", "CSS")
current_cycle = int(os.environ.get("PSA_CYCLE", 1))

# 1. Get the exact folder where this script actually lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Force the 'results' folder to be built inside that specific directory
base_dir = os.path.join(script_dir, "results")
cycle_folder = os.path.join(base_dir, f"{run_type}_cycle_{current_cycle}")

# 3. Create the directories
os.makedirs(cycle_folder, exist_ok=True)
(f"\n--- Starting Desorption/Blowdown for [{run_type}] Cycle {current_cycle} ---")
print(f"--- Results will be saved to: {cycle_folder} ---")

# =============================================================================
# 3. LOAD BED STATE FROM ADSORPTION & SETUP PARAMETERS
# =============================================================================

# --- LOAD PARAMETERS FROM MASTER CONFIG ---
config_path = os.path.join(script_dir, "master_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

tau_bd = config["tau_bd"]
t_blowdown_end = config["t_blowdown_end"]
t_eval_bd = np.linspace(0, t_blowdown_end, config.get("t_bd_steps", 150))
tf_des = config["tf_des"]

state_file = os.path.join(script_dir, 'adsorption_end_state.npz')

if not os.path.exists(state_file):
    raise FileNotFoundError(f"Cannot find {state_file}. Please run the adsorption script first.")

print("Loading saturated bed state from Adsorption phase...")
state_data = np.load(state_file)

L = float(state_data['L'])
T = float(state_data['T'])
R = float(state_data['R'])
d = float(state_data['d'])
A = pi * (d / 2)**2

# --- LOAD DYNAMIC CONFIG VARIABLES ---
N = int(config["N"])
dp = float(config.get("dp", 0.002))
mu = float(config.get("mu", 1.46276889631402e-5))
P_atm_Pa = float(config.get("P_atm_Pa", 101325.0))

eps_1 = float(config.get("eps_1", 0.40))
eps_2 = float(config.get("eps_2", 0.40))
rho_s_1 = float(config.get("rho_s_1", 2000))
rho_s_2 = float(config.get("rho_s_2", 2500))
P_high = float(state_data['P_end'])
P_low = 1e5


C_initial = state_data['C_end']
q_initial = state_data['q_end']
final_mass_flow_purge = float(state_data['final_mass_flow_purge'])

# Combine states for the solver
y0_bd = np.concatenate([C_initial, q_initial])

labels = ['H2', 'CO', 'CO2', 'CH4', 'N2']
colors = ['gray', 'blue', 'red', 'green', 'purple']
MW = np.array([0.002016, 0.02801, 0.04401, 0.01604, 0.02801])

# -- PARAMETERS FOR PENG-ROBINSON EOS --
Tc = np.array([33.19, 132.92, 304.2, 190.56, 126.2])
Pc = np.array([12.98e5, 34.99e5, 73.83e5, 45.99e5, 34.0e5])
omega = np.array([-0.219, 0.045, 0.224, 0.011, 0.037])
kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
alpha = (1 + kappa*(1 - np.sqrt(T/Tc)))**2
a_i = 0.45724 * (R**2 * Tc**2) / Pc * alpha  
b_i = 0.07780 * (R * Tc) / Pc                

def get_Z_PR(P, T, y):
    a_mix, b_mix = np.sum(y * np.sqrt(a_i))**2, np.sum(y * b_i)              
    A_pr, B_pr = a_mix * P / (R**2 * T**2), b_mix * P / (R * T)
    roots = np.roots([1, B_pr-1, A_pr - 2*B_pr - 3*B_pr**2, B_pr**3 + B_pr**2 - A_pr*B_pr])
    return np.max(roots[np.isreal(roots)].real) 

# --- DISCRETIZATION & MATERIAL PROPERTIES ---
dz = L / N
z_nodes = np.linspace(dz/2, L - dz/2, N)

ratio_layer1 = float(config.get("ratio_layer1", 0.6))
z_mid = L * ratio_layer1
k_smooth = 10.0 
weight = 1.0 / (1.0 + np.exp(-k_smooth * (z_nodes - z_mid)))    
weight_2d = weight.reshape(1, N) 

eps = eps_1 + (eps_2 - eps_1) * weight
rho_s = rho_s_1 + (rho_s_2 - rho_s_1) * weight
mass_transfer_coeff = ((1 - eps) / eps) * rho_s

# --- 2D PARAMETERS (Extended Sips isotherms) ---
k_ldf_L1 = np.array([[0.700], [0.150], [0.036], [0.195], [0.261]]) 
k1_L1 = np.array([[16.943], [33.850], [28.797], [23.860], [1.644]])
k2_L1 = np.array([[-0.02100], [-0.09072], [-0.07000], [-0.05621], [-0.00073]]) 
k3_L1 = np.array([[0.625e-4], [2.311e-4], [100.0e-4], [34.780e-4], [545.0e-4]]) 
k4_L1 = np.array([[1229], [1751], [1030], [1159], [326]])
k5_L1 = np.array([[0.980], [3.053], [0.999], [1.618], [0.908]])
k6_L1 = np.array([[43.03], [-654.4], [-37.04], [-248.9], [0.991]])

k_ldf_L2 = np.array([[0.700], [0.063], [0.014], [0.147], [0.099]])  
k1_L2 = np.array([[4.314], [11.845], [10.030], [5.833], [4.813]])
k2_L2 = np.array([[-0.01060], [-0.03130], [-0.01858], [-0.01192], [-0.00668]]) 
k3_L2 = np.array([[25.15e-4], [202.0e-4], [1.578e-4], [6.507e-4], [5.695e-4]]) 
k4_L2 = np.array([[458], [763], [207], [1731], [1531]])
k5_L2 = np.array([[0.986], [3.823], [-5.648], [0.820], [0.842]])
k6_L2 = np.array([[43.03], [-931.3], [2098.0], [53.15], [-7.467]])

k_ldf = k_ldf_L1 + (k_ldf_L2 - k_ldf_L1) * weight_2d
k1    = k1_L1    + (k1_L2    - k1_L1)    * weight_2d
k2    = k2_L1    + (k2_L2    - k2_L1)    * weight_2d
k3    = k3_L1    + (k3_L2    - k3_L1)    * weight_2d
k4    = k4_L1    + (k4_L2    - k4_L1)    * weight_2d
k5    = k5_L1    + (k5_L2    - k5_L1)    * weight_2d
k6    = k6_L1    + (k6_L2    - k6_L1)    * weight_2d

qm = k1 + k2 * T
B  = k3 * np.exp(k4 / T)
n_param  = k5 + k6 / T

# --- UNIFIED SPARSITY MATRIX (Top-Down Flow) ---
total_vars = 10 * N
sparsity_matrix_top_down = np.zeros((total_vars, total_vars))
for eq_type in range(10): 
    for var_type in range(10): 
        for j in range(N):
            sparsity_matrix_top_down[eq_type*N + j, var_type*N + j : var_type*N + N] = 1
jac_sparsity_top_down = sp.csc_matrix(sparsity_matrix_top_down)


# =============================================================================
# 4. BLOWDOWN (DEPRESSURIZATION) PHASE
# =============================================================================

print(f"\n--- Solving Blowdown phase ({P_high/1e5:.1f} bar to {P_low/1e5:.1f} bar) ---")
pbar_bd = tqdm(total=t_blowdown_end, desc="Blowdown", unit="s", bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]")
last_t_bd = [0.0] 

def pde_blowdown(t, y): 
    if t > last_t_bd[0]:
        pbar_bd.update(t - last_t_bd[0])
        last_t_bd[0] = t

    C = y[:5*N].reshape((5, N))
    q = y[5*N:].reshape((5, N))
    
    C_safe = leaky_max(C, 1e-10) 
    C_total_num = np.sum(C_safe, axis=0) 
    y_frac = C_safe / C_total_num        
    
    P_t = P_low + (P_high - P_low) * np.exp(-t / tau_bd)
    dPdt = -(P_high - P_low) / tau_bd * np.exp(-t / tau_bd)
    
    dCtot_dt_expansion = dPdt / (R * T) 
    C_total_theo = P_t / (R * T) 
    
    P_partial = y_frac * P_t 
    p_ratio = np.clip(P_partial / P_atm_Pa, 1e-10, 1000.0)
    
    denom = 1.0 + np.sum(B * (p_ratio ** n_param), axis=0) 
    q_star = (qm * B * (p_ratio ** n_param)) / denom 
    
    dqdt_raw = k_ldf * (q_star - q) 
    availability_ads = C_safe / (C_safe + 1e-3)
    dqdt = np.maximum(dqdt_raw, 0.0) * availability_ads + np.minimum(dqdt_raw, 0.0) 
    
    sum_dqdt = np.sum(dqdt, axis=0) 
    du = -dz * (eps * dCtot_dt_expansion + (1 - eps) * rho_s * sum_dqdt) / C_total_theo
    
    u_superficial = np.maximum(np.cumsum(du[::-1])[::-1], 0.0) 
    v_local = u_superficial / eps 
    
    flux_limiter = np.maximum(C, 0.0) / (np.maximum(C, 0.0) + 1e-6)
    flux = v_local * C * flux_limiter
    
    flux_in = np.column_stack((flux[:, 1:], np.zeros(5))) 
    
    d_vC_dz = (flux - flux_in) / dz
    dCdt = -d_vC_dz - mass_transfer_coeff * dqdt
        
    return np.concatenate([dCdt.flatten(), dqdt.flatten()])

sol_bd = solve_ivp(pde_blowdown, [t_eval_bd[0], t_eval_bd[-1]], y0_bd, method='BDF', 
                   t_eval=t_eval_bd, 
                   rtol=5e-2, atol=1e-4, first_step=1e-11)
pbar_bd.close() 

actual_steps_bd = sol_bd.y.shape[1]
C_history_bd = sol_bd.y[:5*N, :].T.reshape((actual_steps_bd, 5, N))
q_history_bd = sol_bd.y[5*N:, :].T.reshape((actual_steps_bd, 5, N))
t_actual_bd = sol_bd.t

# --- Calculate Blowdown Tail Gas & Pressure Drop ---
species_flow_bd = np.zeros((actual_steps_bd, 5))
delta_P_history_bd = np.zeros(actual_steps_bd)

for i, t in enumerate(t_actual_bd):
    P_t = P_low + (P_high - P_low) * np.exp(-t / tau_bd)
    dPdt = -(P_high - P_low) / tau_bd * np.exp(-t / tau_bd)
    dCtot_dt_expansion = dPdt / (R * T)
    C_total_theo = P_t / (R * T)
    
    C_at_t = np.maximum(C_history_bd[i], 1e-10)
    C_tot = np.sum(C_at_t, axis=0)
    y_frac_viz = C_at_t / C_tot
    p_ratio_viz = np.clip((y_frac_viz * P_t) / P_atm_Pa, 1e-10, 1000.0)
    denom_viz = 1.0 + np.sum(B * (p_ratio_viz ** n_param), axis=0)
    q_star_viz = (qm * B * (p_ratio_viz ** n_param)) / denom_viz
    
    dqdt_viz = k_ldf * (q_star_viz - q_history_bd[i])
    sum_dqdt = np.sum(dqdt_viz, axis=0)
    
    du = -dz * (eps * dCtot_dt_expansion + (1 - eps) * rho_s * sum_dqdt) / C_total_theo
    
    u_superficial_bd = np.maximum(np.cumsum(du[::-1])[::-1], 0.0)
    u_exit = u_superficial_bd[0]  
    
    species_flow_bd[i, :] = u_exit * A * C_at_t[:, 0]
    
    P_total_bd = np.zeros(N)
    P_total_bd[0] = P_t 
    
    for j in range(N - 1):
        rho_gas = np.sum(C_at_t[:, j] * MW)
        Re_prime = (dp * u_superficial_bd[j] * rho_gas) / mu
        Re_prime = max(Re_prime, 1e-4)
        
        f_fric = ((1 - eps[j]) / eps[j]**3) * (1.75 + 150 * (1 - eps[j]) / Re_prime)
        dPdz = (f_fric * u_superficial_bd[j]**2 * rho_gas) / dp 
        P_total_bd[j+1] = P_total_bd[j] + dPdz * dz 
        
    delta_P_history_bd[i] = P_total_bd[-1] - P_total_bd[0]

moles_bd = np.trapz(species_flow_bd, t_actual_bd, axis=0)


# =============================================================================
# 5. DESORPTION (PURGE) PHASE
# =============================================================================
y0_des = sol_bd.y[:, -1] 
t_eval_des = np.linspace(0, tf_des, 150) 
y_feed = np.array([1.0, 1e-8, 1e-8, 1e-8, 1e-8])

molar_flow_H2 = final_mass_flow_purge / MW[0] 
C_total_calc = P_low / (R * T)         
u_feed = molar_flow_H2 / (A * C_total_calc)
C_in = y_feed * (P_low / (get_Z_PR(P_low, T, y_feed) * R * T))

print(f"\n--- Solving Desorption phase ({tf_des} seconds) ---")
pbar_des = tqdm(total=tf_des, desc="Desorption", unit="s", bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]")
last_t_des = [0.0] 

def pde_desorption(t, y): 
    if t > last_t_des[0]:
        pbar_des.update(t - last_t_des[0])
        last_t_des[0] = t

    C = y[:5*N].reshape((5, N))
    q = y[5*N:].reshape((5, N))
    
    C_safe = leaky_max(C, 1e-10)
    q_safe = leaky_max(q, 0.0)

    C_tot = np.sum(C_safe, axis=0) 
    y_frac = C_safe / C_tot 
    u_inlet = u_feed * (1.0 - np.exp(-t / 5.0))
    
    P_partial = y_frac * P_low
    p_ratio = np.clip(P_partial / P_atm_Pa, 1e-10, 1000.0)
    denom = 1.0 + np.sum(B * (p_ratio ** n_param), axis=0)
    q_star = (qm * B * (p_ratio ** n_param)) / denom
    
    dqdt = k_ldf * (q_star - q_safe)
    
    sum_dqdt = np.sum(dqdt, axis=0)
    du = -dz * ((1 - eps) * rho_s * sum_dqdt) / (P_low / (R * T))
    du = np.clip(du, -u_inlet * 0.1, u_inlet * 0.1) 
    
    u_superficial = u_inlet + np.cumsum(du[::-1])[::-1]
    u_superficial = np.clip(u_superficial, 1e-4, 5.0) 
    
    v_local = u_superficial / eps
    flux = v_local * C_safe
    flux_boundary_L = (u_inlet / eps[-1]) * C_in
    flux_in = np.column_stack((flux[:, 1:], flux_boundary_L))
    
    d_vC_dz = (flux - flux_in) / dz
    dCdt = -d_vC_dz - mass_transfer_coeff * dqdt
    
    return np.concatenate((dCdt.ravel(), dqdt.ravel()))

sol_des = solve_ivp(pde_desorption, [t_eval_des[0], t_eval_des[-1]], y0_des, method='BDF', 
                    t_eval=t_eval_des, 
                    rtol=5e-2, atol=1e-4, first_step=1e-11)
pbar_des.close()

actual_steps_des = sol_des.y.shape[1]
t_actual_des = sol_des.t  
C_history_des = sol_des.y[:5*N, :].T.reshape((actual_steps_des, 5, N))
q_history_des = sol_des.y[5*N:, :].T.reshape((actual_steps_des, 5, N)) 

# --- Calculate Desorption Tail Gas & Pressure Drop ---
species_flow_des = np.zeros((actual_steps_des, 5))
delta_P_history_des = np.zeros(actual_steps_des)

for i, t in enumerate(t_actual_des):
    C_t = np.maximum(C_history_des[i], 1e-10)
    q_t = np.maximum(q_history_des[i], 0.0)
    
    C_tot_t = np.sum(C_t, axis=0)
    y_frac = C_t / C_tot_t
    u_inlet = u_feed * (1.0 - np.exp(-t / 5.0))
    
    P_partial = y_frac * P_low
    p_ratio = np.clip(P_partial / P_atm_Pa, 1e-10, 1000.0)
    denom = 1.0 + np.sum(B * (p_ratio ** np.maximum(n_param, 0.01)), axis=0)
    q_star = (qm * B * (p_ratio ** n_param)) / denom
    
    dqdt = k_ldf * (q_star - q_t)
    sum_dqdt = np.sum(dqdt, axis=0)
    
    du = -dz * ((1 - eps) * rho_s * sum_dqdt) / C_total_calc
    du = np.clip(du, -u_feed * 0.5, 0.5)
    u_superficial_des = u_inlet + np.cumsum(du[::-1])[::-1]
    u_superficial_des = np.clip(u_superficial_des, 1e-4, 5.0)
    u_exit = u_superficial_des[0]  
    
    species_flow_des[i, :] = u_exit * A * C_total_calc * y_frac[:, 0]

    P_total_des = np.zeros(N)
    P_total_des[0] = P_low 
    
    for j in range(N - 1):
        rho_gas = np.sum(C_t[:, j] * MW)
        Re_prime = (dp * u_superficial_des[j] * rho_gas) / mu
        Re_prime = max(Re_prime, 1e-4)
        
        f_fric = ((1 - eps[j]) / eps[j]**3) * (1.75 + 150 * (1 - eps[j]) / Re_prime)
        dPdz = (f_fric * u_superficial_des[j]**2 * rho_gas) / dp 
        P_total_des[j+1] = P_total_des[j] + dPdz * dz 
        
    delta_P_history_des[i] = P_total_des[-1] - P_total_des[0] 

moles_des = np.trapz(species_flow_des, t_actual_des, axis=0)

# =============================================================================
# 6. COMPREHENSIVE TAIL GAS REPORT
# =============================================================================

sum_moles_bd = np.sum(moles_bd)
comp_bd = (moles_bd / sum_moles_bd) * 100

sum_moles_des = np.sum(moles_des)
comp_des = (moles_des / sum_moles_des) * 100

total_moles_all = moles_bd + moles_des
total_sum_all = np.sum(total_moles_all)
comp_total = (total_moles_all / total_sum_all) * 100


sample_indices = np.linspace(0, actual_steps_des - 1, 10, dtype=int)
for idx in sample_indices:
    t_val = t_actual_des[idx]
    C_exit = np.maximum(C_history_des[idx, :, 0], 1e-10)
    y_exit = C_exit / np.sum(C_exit)
    Instant_Flow = np.sum(species_flow_des[idx])
    DP_val = delta_P_history_des[idx]


# =============================================================================
# 7. VISUALIZATION & PLOTTING
# =============================================================================

# --- Combine BD and DES arrays for unified charting ---
# We shift the Desorption time by the length of the Blowdown time to make a continuous timeline
t_combined = np.concatenate([t_actual_bd, t_actual_des + t_actual_bd[-1]])
C_combined = np.concatenate([C_history_bd, C_history_des], axis=0)
q_combined = np.concatenate([q_history_bd, q_history_des], axis=0)

C_plot_combined = np.maximum(C_combined, 1e-8) 
y_history_combined = C_plot_combined / np.sum(C_plot_combined, axis=1, keepdims=True)

# Calculate Absolute Pressures
P_exit_bd = (P_low + (P_high - P_low) * np.exp(-t_actual_bd / tau_bd)) / 1e5
P_inlet_bd = P_exit_bd + (delta_P_history_bd / 1e5)

P_exit_des = np.full_like(t_actual_des, P_low / 1e5)
P_inlet_des = P_exit_des + (delta_P_history_des / 1e5)

fig_metrics, axes_metrics = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)

axes_metrics[0, 0].plot(t_actual_bd, np.sum(species_flow_bd, axis=1), color='orange', linewidth=2.5)
axes_metrics[0, 0].set_title('A. Blowdown: Total Molar Flow at Exit (z=0)')
axes_metrics[0, 0].set_xlabel('Time (s)')
axes_metrics[0, 0].set_ylabel('Flow Rate (mol/s)')
axes_metrics[0, 0].grid(True)

axes_metrics[0, 1].plot(t_actual_des, np.sum(species_flow_des, axis=1), color='blue', linewidth=2.5)
axes_metrics[0, 1].set_title('B. Desorption/Purge: Total Molar Flow at Exit (z=0)')
axes_metrics[0, 1].set_xlabel('Time (s)')
axes_metrics[0, 1].set_ylabel('Flow Rate (mol/s)')
axes_metrics[0, 1].grid(True)

axes_metrics[1, 0].plot(t_actual_bd, P_inlet_bd, color='red', label='Inlet (z=L)', linewidth=2)
axes_metrics[1, 0].plot(t_actual_bd, P_exit_bd, color='black', linestyle='--', label='Exit (z=0)', linewidth=2)
axes_metrics[1, 0].set_title('C. Blowdown: Absolute Pressure Profile')
axes_metrics[1, 0].set_xlabel('Time (s)')
axes_metrics[1, 0].set_ylabel('Pressure (bar)')
axes_metrics[1, 0].legend()
axes_metrics[1, 0].grid(True)

axes_metrics[1, 1].plot(t_actual_des, P_inlet_des, color='purple', label='Inlet (z=L)', linewidth=2)
axes_metrics[1, 1].plot(t_actual_des, P_exit_des, color='black', linestyle='--', label='Exit (z=0)', linewidth=2)
axes_metrics[1, 1].set_title('D. Desorption/Purge: Absolute Pressure Profile')
axes_metrics[1, 1].set_xlabel('Time (s)')
axes_metrics[1, 1].set_ylabel('Pressure (bar)')
axes_metrics[1, 1].legend()
axes_metrics[1, 1].grid(True)

fig2d, axes2d = plt.subplots(3, 2, figsize=(14, 15), tight_layout=True)
axes_flat = axes2d.flatten()

ax_elution = axes_flat[0]
for i in range(5): 
    # PLOT THE COMBINED TIMELINE
    ax_elution.plot(t_combined, y_history_combined[:, i, 0] * 100, label=labels[i], color=colors[i], linewidth=2.5)

# Add a vertical line to show exactly where Blowdown ends and Purge begins
ax_elution.axvline(x=t_actual_bd[-1], color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Purge Start')
ax_elution.set(title='Total Regeneration Elution Curve (z=0)', xlabel='Time (s)', ylabel='Molar Fraction (%)')
ax_elution.legend()
ax_elution.grid(True, which="major", ls="-", alpha=0.5)

# Pick 5 times spread across BOTH phases
target_times = [0.0, t_actual_bd[-1]*0.5, t_actual_bd[-1], t_actual_bd[-1] + 0.5*tf_des, t_combined[-1]]
time_indices = [int(np.argmin(np.abs(t_combined - t_sec))) for t_sec in target_times]
time_cmap = plt.cm.plasma(np.linspace(0, 0.8, len(time_indices)))

for i in range(5):
    ax_sp = axes_flat[i+1]
    for t_step_idx, t_data_idx in enumerate(time_indices):
        if t_data_idx < len(t_combined): 
            # Dynamically label the legend so you know if it's Blowdown or Purge
            phase = "BD" if t_combined[t_data_idx] <= t_actual_bd[-1] else "Purge"
            ax_sp.plot(z_nodes, q_combined[t_data_idx, i, :], 
                       label=f'{phase} t={t_combined[t_data_idx]:.0f}s', color=time_cmap[t_step_idx], linewidth=2)
            
    ax_sp.set(title=f'Spatial Profile: {labels[i]}', xlabel='Column Length z (m)', ylabel='Solid Loading q (mol/kg)')
    ax_sp.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
    ax_sp.legend(fontsize=8)
    ax_sp.grid(True, alpha=0.5)
    ax_sp.axvline(x=L * ratio_layer1, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
    
# =============================================================================
# 8. SAVE RESULTS & PREPARE FOR NEXT CYCLE
# =============================================================================

# 1. Save the Flow & Pressure Metrics Figure
metrics_path = os.path.join(cycle_folder, "des_flow_pressure_metrics.png")
fig_metrics.savefig(metrics_path)
print(f"Saved metrics to {metrics_path}")

# 2. Save the Elution & Spatial Profiles Figure
profiles_path = os.path.join(cycle_folder, "des_spatial_profiles.png")
fig2d.savefig(profiles_path)
print(f"Saved profiles to {profiles_path}")

# --- CRITICAL: FORCE ABSOLUTE PATH FOR DESORPTION STATE HANDOVER ---
des_state_file = os.path.join(script_dir, 'desorption_end_state.npz')
np.savez(des_state_file, 
         C_end=sol_des.y[:5*N, -1], 
         q_end=sol_des.y[5*N:, -1])

summary_path = os.path.join(cycle_folder, "desorption_summary.txt")
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("\n" + "="*115 + "\n")
    f.write(f"{'PHASE SUMMARY':^115}\n")
    f.write("="*115 + "\n")
    header = f"{'Species':<10} | {'BD Moles':<12} | {'BD Mix %':<10} | {'DES Moles':<12} | {'DES Mix %':<10} | {'TOTAL Moles':<12} | {'FINAL Mix %':<10}"
    f.write(header + "\n")
    f.write("-" * 115 + "\n")

    for i in range(5):
        f.write(f"{labels[i]:<10} | {moles_bd[i]:<12.3f} | {comp_bd[i]:<10.2f} | {moles_des[i]:<12.3f} | {comp_des[i]:<10.2f} | {total_moles_all[i]:<12.3f} | {comp_total[i]:<10.2f}\n")

    f.write("-" * 115 + "\n")
    f.write(f"{'TOTAL':<10} | {sum_moles_bd:<12.3f} | {'100.00':<10} | {sum_moles_des:<12.3f} | {'100.00':<10} | {total_sum_all:<12.3f} | {'100.00':<10}\n")
    f.write("="*115 + "\n\n")

    f.write("="*102 + "\n")
    f.write(" INSTANTANEOUS PURGE FRONT PROGRESSION (z = 0)\n")
    f.write("="*102 + "\n")
    f.write(f"{'Time (s)':<10} | {'Total Flow (mol/s)':<18} | {'ΔP (Pa)':<9} | {'H2 (%)':<8} | {'CO (%)':<8} | {'CO2 (%)':<8} | {'CH4 (%)':<8} | {'N2 (%)':<8}\n")
    f.write("-" * 102 + "\n")

    sample_indices = np.linspace(0, actual_steps_des - 1, 10, dtype=int)
    for idx in sample_indices:
        t_val = t_actual_des[idx]
        C_exit = np.maximum(C_history_des[idx, :, 0], 1e-10)
        y_exit = C_exit / np.sum(C_exit)
        Instant_Flow = np.sum(species_flow_des[idx])
        DP_val = delta_P_history_des[idx]
        
        f.write(f"{t_val:<10.1f} | {Instant_Flow:<18.4f} | {DP_val:<9.1f} | {y_exit[0]*100:<8.2f} | {y_exit[1]*100:<8.2f} | {y_exit[2]*100:<8.2f} | {y_exit[3]*100:<8.2f} | {y_exit[4]*100:<8.2f}\n")
    f.write("="*102 + "\n\n")

print(f"✅ Desorption state saved to {des_state_file}")
print(f"Ready for Cycle {current_cycle + 1} Adsorption.")

plt.close('all')
os._exit(0)