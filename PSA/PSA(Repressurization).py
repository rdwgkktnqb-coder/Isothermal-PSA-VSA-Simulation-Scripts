import os
from math import pi
import numpy as np
import matplotlib
import json

# --- 1. THE MATPLOTLIB SILVER BULLET MUST GO HERE ---
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

def leaky_max(x, eps=1e-10, alpha=1e-6):
    return np.where(x > eps, x, eps + alpha * (x - eps))

# =============================================================================
# 2. CYCLE DETECTION & SETUP
# =============================================================================
run_type = os.environ.get("RUN_TYPE", "CSS") # Defaults to CSS
current_cycle = int(os.environ.get("PSA_CYCLE", 1)) # <--- FIXED: Cast to int

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "results")
cycle_folder = os.path.join(base_dir, f"{run_type}_cycle_{current_cycle}")
os.makedirs(cycle_folder, exist_ok=True)

print(f"\n--- Starting Repressurization for [{run_type}] Cycle {current_cycle} ---")

# =============================================================================
# 3. LOAD PARAMETERS & BED STATE
# =============================================================================
config_path = os.path.join(script_dir, "master_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# --- STRICT TIMING ENFORCEMENT ---
tf_rep = float(config.get("t_rep", 60.0))
t_eval_rep = np.linspace(0, tf_rep, 150)

L = float(config["L"])
T = float(config["T"])
R = float(config["R"])
d = float(config["d"])
A = pi * (d / 2)**2
N = int(config["N"])
P_high = float(config["P_high"])
P_low = float(config["P_low"])
P_atm_Pa = config.get("P_atm_Pa", 101325.0)


state_file = os.path.join(script_dir, 'desorption_end_state.npz')
if not os.path.exists(state_file):
    raise FileNotFoundError(f"Cannot find {state_file}.")

state_data = np.load(state_file)
y0_rep = np.concatenate([state_data['C_end'], state_data['q_end']])

labels = ['H2', 'CO', 'CO2', 'CH4', 'N2']
colors = ['gray', 'blue', 'red', 'green', 'purple']
MW = np.array([0.002016, 0.02801, 0.04401, 0.01604, 0.02801])
dz = L / N
z_nodes = np.linspace(dz/2, L - dz/2, N)

# --- Discretization (Same as previous) ---
z_mid = L * 0.6
weight = 1.0 / (1.0 + np.exp(-10.0 * (z_nodes - z_mid)))    
weight_2d = weight.reshape(1, N) 

eps = 0.40 + 0.0 * weight
rho_s = 2000 + 500 * weight
mass_transfer_coeff = ((1 - eps) / eps) * rho_s

# --- Sips Isotherm Arrays ---
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

# =============================================================================
# 4. REPRESSURIZATION PHASE (IMPOSED RAMP)
# =============================================================================

print(f"\n--- Solving Repressurization phase (Target: {P_high/1e5:.1f} bar in {tf_rep:.1f}s) ---")
pbar_rep = tqdm(total=P_high, desc="Pressurizing", unit="Pa", bar_format="{l_bar}{bar}| {n:.2e}/{total_fmt} [{elapsed}<{remaining}]")
last_P_mean = [P_low] 

def pde_repressurization(t, y): 
    C = y[:5*N].reshape((5, N))
    q = y[5*N:].reshape((5, N))
    
    C_safe = leaky_max(C, 1e-10) 
    C_tot = np.sum(C_safe, axis=0) 
    y_frac = C_safe / C_tot        
    
    P_local = C_tot * R * T
    P_mean = np.mean(P_local)
    
    if P_mean > last_P_mean[0]:
        pbar_rep.update(P_mean - last_P_mean[0])
        last_P_mean[0] = P_mean

    # Readsorption thermodynamics
    P_partial = y_frac * P_local 
    p_ratio = np.clip(P_partial / P_atm_Pa, 1e-10, 1000.0)
    denom = 1.0 + np.sum(B * (p_ratio ** n_param), axis=0) 
    q_star = (qm * B * (p_ratio ** n_param)) / denom 
    dqdt = k_ldf * (q_star - q) 
    sum_dqdt = np.sum(dqdt, axis=0) 
    
    # --- EXACT LINEAR PRESSURE RAMP ---
    dPdt_target = (P_high - P_low) / tf_rep
    dCtot_dt_compression = dPdt_target / (R * T)
    
    # Calculate localized velocity accumulation (Counter-Current)
    du = dz * (eps * dCtot_dt_compression + (1 - eps) * rho_s * sum_dqdt) / C_tot
    
    # Build velocity profile naturally from the wall outward
    u_superficial = np.zeros(N)
    u_superficial[0] = 0.0 # Strict zero at closed end
    for j in range(1, N):
        u_superficial[j] = u_superficial[j-1] + du[j-1]
        
    v_local = u_superficial / eps 
    
    # Flux & Upwind Discretization (Counter-Current)
    flux_limiter = np.maximum(C, 0.0) / (np.maximum(C, 0.0) + 1e-6)
    flux = v_local * C * flux_limiter
    
    # The exact dynamic velocity required at the product end to maintain the ramp
    u_feed_dynamic = u_superficial[-1] + du[-1]
    
    C_in_L = np.array([(P_local[-1] / (R * T)), 0, 0, 0, 0])
    flux_boundary_L = (u_feed_dynamic / eps[-1]) * C_in_L
    
    flux_in = np.column_stack((flux[:, 1:], flux_boundary_L)) 
    
    d_vC_dz = (flux - flux_in) / dz
    dCdt = -d_vC_dz - mass_transfer_coeff * dqdt
        
    return np.concatenate([dCdt.flatten(), dqdt.flatten()])

# Because we enforce the ramp, it is mathematically guaranteed to reach P_high at tf_rep.
sol_rep = solve_ivp(pde_repressurization, [t_eval_rep[0], t_eval_rep[-1]], y0_rep, method='BDF', 
                   t_eval=t_eval_rep,
                   rtol=1e-3, atol=1e-5, first_step=1e-6) 

pbar_rep.close() 

actual_steps_rep = sol_rep.y.shape[1]
C_history_rep = sol_rep.y[:5*N, :].T.reshape((actual_steps_rep, 5, N))
t_actual_rep = sol_rep.t

P_history = np.sum(np.maximum(C_history_rep, 1e-10), axis=1) * R * T

# =============================================================================
# 5. VISUALIZATION & PLOTTING
# =============================================================================
fig_p, ax_p = plt.subplots(figsize=(10, 6), tight_layout=True)

sample_indices = np.linspace(0, actual_steps_rep - 1, 5, dtype=int)
time_cmap = plt.cm.viridis(np.linspace(0, 0.9, len(sample_indices)))

for i, idx in enumerate(sample_indices):
    t_val = t_actual_rep[idx]
    ax_p.plot(z_nodes, P_history[idx] / 1e5, 
              label=f't={t_val:.1f}s', color=time_cmap[i], linewidth=2.5)

ax_p.set_title(f'Repressurization: Imposed Pressure Build-up ({tf_rep}s)', fontsize=14, fontweight='bold')
ax_p.set_xlabel('Column Length z (m)', fontsize=12)
ax_p.set_ylabel('Pressure (bar)', fontsize=12)
ax_p.axvline(x=L, color='red', linestyle='--', label='Product End (Pure H2 Inlet)', alpha=0.7)
ax_p.axvline(x=0, color='black', linestyle='-', label='Feed End (Closed)', alpha=0.7)
ax_p.set_ylim(0, P_high/1e5 * 1.1) # Anchor the Y axis nicely
ax_p.legend()
ax_p.grid(True, linestyle=':', alpha=0.6)

# =============================================================================
# 6. SAVE RESULTS
# =============================================================================
pressure_path = os.path.join(cycle_folder, "rep_pressure_wave.png")
fig_p.savefig(pressure_path)

rep_state_file = os.path.join(script_dir, 'repressurization_end_state.npz')
np.savez(rep_state_file, 
         C_end=sol_rep.y[:5*N, -1], 
         q_end=sol_rep.y[5*N:, -1])

print(f"✅ State saved to {rep_state_file}")

plt.close('all')
os._exit(0)