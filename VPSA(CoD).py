import os
from math import pi
import numpy as np
import matplotlib
import json
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
from tqdm import tqdm

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

print(f"\n--- Starting Co-Current Depressurization (CoD) for [{run_type}] Cycle {current_cycle} ---")

# =============================================================================
# 2. LOAD PARAMETERS & RINSE STATE
# =============================================================================
config_path = os.path.join(script_dir, "master_config.json")
with open(config_path, "r") as f:
    config = json.load(f) 

state_file = os.path.join(script_dir, 'adsorption_end_state.npz')
if not os.path.exists(state_file):
    raise FileNotFoundError(f"Cannot find {state_file}. Did Rinse run?")

state_data = np.load(state_file)

L = float(state_data['L'])
T = float(state_data['T'])
R = float(state_data['R'])
d = float(state_data['d'])
A = pi * (d / 2)**2
N = int(config["N"])

eps = float(config.get("eps", 0.35))
rho_s = float(config.get("rho_s", 2000))
P_high = float(state_data['P_end'])

# --- CoD Specific Parameters ---
# If not in master_config, we set default values here
P_mid = config.get("P_mid", 0.4 * 101325.0)  # Drop to 0.4 bar
t_cod_end = config.get("t_cod_end", 20.0)    # 20 seconds duration
tau_cod = config.get("tau_cod", 10.0)        # Time constant

t_eval_cod = np.linspace(0, t_cod_end, 150)

C_initial = state_data['C_end']
q_initial = state_data['q_end']
y0_cod = np.concatenate([C_initial, q_initial])

labels = ['N2', 'CO2', 'O2']
colors = ['gray', 'blue', 'red']
MW = np.array([0.028014, 0.044009, 0.031998]) 

dz = L / N
z_nodes = np.linspace(dz/2, L - dz/2, N)
mass_transfer_coeff = ((1 - eps) / eps) * rho_s

# --- TOTH ISOTHERM CONSTANTS ---
k_ldf = np.array([[0.0021], [0.0143], [0.002]]).repeat(N, axis=1) 

k_qs_1 = np.array([[1.89],[2.82],[1e-8]]).repeat(N, axis=1)
k_qs_2 = np.array([[-2.25e-4],[-3.50e-4],[1e-8]]).repeat(N, axis=1)
q_s = k_qs_1 + k_qs_2 * T    

b_0 = np.array([[1.16e-9], [2.83e-9], [1e-8]]).repeat(N, axis=1)  
B_val = np.array([[1944.61], [2598.2], [1e-8]]).repeat(N, axis=1)                 
b_toth = b_0 * np.exp(B_val / T)

# =============================================================================
# 3. CoD PDE SOLVER (Forward Flow)
# =============================================================================
print(f"\n--- Solving CoD phase ({P_high/1e5:.2f} bar to {P_mid/1e5:.2f} bar) ---")
pbar_cod = tqdm(total=t_cod_end, desc="CoD Phase", unit="s", bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]")
last_t_cod = [0.0] 
def pde_cod(t, y): 
    if t > last_t_cod[0]:
        pbar_cod.update(t - last_t_cod[0])
        last_t_cod[0] = t

    C = y[:3*N].reshape((3, N))
    q = y[3*N:].reshape((3, N))
    
    # FIX 1: Clamp concentrations to prevent Jacobian inversions
    C_safe = np.maximum(C, 1e-12) 
    C_total_num = np.sum(C_safe, axis=0) 
    y_frac = C_safe / C_total_num        
    
    P_t = P_mid + (P_high - P_mid) * np.exp(-t / tau_cod)
    dPdt = -(P_high - P_mid) / tau_cod * np.exp(-t / tau_cod)
    
    dCtot_dt_expansion = dPdt / (R * T) 
    C_total_theo = P_t / (R * T) 
    
    P_partial_kPa = (y_frac * P_t) 
    
    base = np.abs(b_toth * P_partial_kPa)
    denom = 1.0 + (b_toth * P_partial_kPa)
    q_star = (b_toth * P_partial_kPa * q_s) / denom
    
    dqdt = k_ldf * (q_star - q) 
    sum_dqdt = np.sum(dqdt, axis=0) 
    
    du = -dz * (eps * dCtot_dt_expansion + (1 - eps) * rho_s * sum_dqdt) / C_total_theo
    
    # FIX 2: Strictly enforce forward velocity so the upwinding math never breaks
    u_superficial = np.maximum(np.cumsum(du), 0.0)
    v_local = u_superficial / eps 
    
    flux = v_local * C 
    flux_in = np.column_stack((np.zeros(3), flux[:, :-1])) 
    
    d_vC_dz = (flux - flux_in) / dz
    dCdt = -d_vC_dz - mass_transfer_coeff * dqdt
        
    return np.concatenate([dCdt.flatten(), dqdt.flatten()])

sol_cod = solve_ivp(pde_cod, [t_eval_cod[0], t_eval_cod[-1]], y0_cod, method='BDF', 
                   t_eval=t_eval_cod, 
                   rtol=1e-4, atol=1e-8)
pbar_cod.close() 

# =============================================================================
# 4. REPORTING & VISUALIZATION
# =============================================================================
actual_steps = sol_cod.y.shape[1]
C_history_cod = sol_cod.y[:3*N, :].T.reshape((actual_steps, 3, N))
t_actual = sol_cod.t

# Calculate gas flow leaving at z=L (Waste / Void Gas)
flow_out_L = np.zeros((actual_steps, 3))
for i, t in enumerate(t_actual):
    P_t = P_mid + (P_high - P_mid) * np.exp(-t / tau_cod)
    dPdt = -(P_high - P_mid) / tau_cod * np.exp(-t / tau_cod)
    dCtot_dt_expansion = dPdt / (R * T)
    C_total_theo = P_t / (R * T)
    
    C_at_t = np.maximum(C_history_cod[i], 1e-10)
    C_tot = np.sum(C_at_t, axis=0)
    y_frac_viz = C_at_t / C_tot
    
    P_partial_kPa_viz = (y_frac_viz * P_t) 
    denom_viz = 1.0 + (b_toth * P_partial_kPa_viz)
    q_star_viz = (b_toth * P_partial_kPa_viz * q_s) / denom_viz
    
    dqdt_viz = k_ldf * (q_star_viz - sol_cod.y[3*N:, i].reshape(3, N))
    sum_dqdt = np.sum(dqdt_viz, axis=0)
    
    du = -dz * (eps * dCtot_dt_expansion + (1 - eps) * rho_s * sum_dqdt) / C_total_theo
    u_superficial_cod = np.maximum(np.cumsum(du), 0.0) 
    u_exit = u_superficial_cod[-1]
    
    flow_out_L[i, :] = u_exit * A * C_at_t[:, -1]

moles_waste = np.trapz(flow_out_L, t_actual, axis=0)

print(f"\n--- CoD WASTE GAS EXPELLED (z=L) ---")
print(f"N2 expelled:  {moles_waste[0]:.4e} mol")
print(f"CO2 expelled: {moles_waste[1]:.4e} mol (Loss)")
print(f"O2 expelled:  {moles_waste[2]:.4e} mol")

fig, ax = plt.subplots(figsize=(8, 5))
for i in range(3):
    ax.plot(t_actual, flow_out_L[:, i], label=labels[i], color=colors[i], linewidth=2)
ax.set_title('CoD Flow Rate Expelled at z=L (Void Gas Flush)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Flow Rate (mol/s)')
ax.grid(True)
ax.legend()
fig.savefig(os.path.join(cycle_folder, "cod_exhaust_flow.png"))

# =============================================================================
# 5. SAVE STATE FOR BLOWDOWN
# =============================================================================
cod_state_file = os.path.join(script_dir, 'cod_end_state.npz')
np.savez(cod_state_file, 
         L=L, T=T, R=R, P_end=P_mid, d=d, # Note P_end is now P_mid
         C_end=sol_cod.y[:3*N, -1], 
         q_end=sol_cod.y[3*N:, -1])

print(f"✅ CoD state saved to {cod_state_file}. Ready for Rinse!")
plt.close('all')
os._exit(0)