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
# 1. AUTONOMOUS RUN ROUTER & PATHS
# =============================================================================
run_type = os.environ.get("RUN_TYPE", "CSS") 
current_cycle = os.environ.get("PSA_CYCLE", "1")

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "results")
cycle_folder = os.path.join(base_dir, f"{run_type}_cycle_{current_cycle}")
os.makedirs(cycle_folder, exist_ok=True)

print(f"\n--- Starting Adsorption: [{run_type} MODE] Cycle {current_cycle} ---")

# --- LOAD MASTER CONFIG ---
config_path = os.path.join(script_dir, "master_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

L = config["L"]; T = config["T"]; R = config["R"]
P_high = config["P_high"]; P_low = config["P_low"]
d = config["d"]; Nsets = config["Nsets"]; ratio_layer1 = config["ratio_layer1"]
purge_fraction = config["purge_fraction"]
t_ads_safety_ratio = float(config.get("t_ads_safety_ratio", 0.90))

N = int(config["N"])
dp = float(config.get("dp", 0.002))
mu = float(config.get("mu", 1.135086e-05))
P_atm_Pa = float(config.get("P_atm_Pa", 101325.0))

eps_1 = float(config.get("eps_1", 0.40)); eps_2 = float(config.get("eps_2", 0.40))
rho_s_1 = float(config.get("rho_s_1", 2000)); rho_s_2 = float(config.get("rho_s_2", 2500))

if run_type in ["SCOUT", "FINAL"]:
    t_end = config["t_ads_end"] 
else:
    t_end = config.get("t_op_ads", 400) 
t_ads_eval = np.linspace(0, t_end, config.get("t_ads_steps", 500))

# =============================================================================
# 2. FEED SIZING & SYSTEM PARAMETERS
# =============================================================================
A = pi*(d/2)**2 
labels = ['H2', 'CO', 'CO2', 'CH4', 'N2']
colors = ['gray', 'blue', 'red', 'green', 'purple']
MW = np.array([0.002016, 0.02801, 0.04401, 0.01604, 0.02801])
y_feed = np.array([0.783843, 0.003649, 0.193034, 0.017725, 0.001951])
y_feed /= np.sum(y_feed)

# Peng-Robinson Z-factor logic
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
    roots = np.roots([1, -(1-B_pr), A_pr - 2*B_pr - 3*B_pr**2, -(A_pr*B_pr - B_pr**2 - B_pr**3)])
    return np.max(roots[np.isreal(roots)].real) 

mass_out_total = 100000e3/(330*24*3600)  
target_mass_out_H2 = mass_out_total/Nsets
expected_recovery = 0.73

required_molar_in_H2 = (target_mass_out_H2 / MW[0]) / expected_recovery
required_molar_in_total = required_molar_in_H2 / y_feed[0]

Z_high = get_Z_PR(P_high, T, y_feed)
C_in_total = P_high / (Z_high * R * T)
u_feed = required_molar_in_total / (A * C_in_total)
C_in = y_feed * C_in_total

dz = L / N
z_nodes = np.linspace(dz/2, L - dz/2, N)
weight = 1.0 / (1.0 + np.exp(-10.0 * (z_nodes - L*ratio_layer1)))   
eps = eps_1 + (eps_2 - eps_1) * weight
rho_s = rho_s_1 + (rho_s_2 - rho_s_1) * weight

# --- ISOTHERMS ---
k_ldf_L1 = np.array([[0.7], [0.15], [0.036], [0.195], [0.261]])
k1_L1 = np.array([[16.943], [33.85], [28.797], [23.86], [1.644]])
k2_L1 = np.array([[-0.021], [-0.09072], [-0.07], [-0.05621], [-0.00073]])
k3_L1 = np.array([[0.625e-4], [2.311e-4], [100.0e-4], [34.78e-4], [545.0e-4]])
k4_L1 = np.array([[1229], [1751], [1030], [1159], [326]])
k5_L1 = np.array([[0.98], [3.053], [0.999], [1.618], [0.908]])
k6_L1 = np.array([[43.03], [-654.4], [-37.04], [-248.9], [0.991]])

k_ldf_L2 = np.array([[0.7], [0.063], [0.014], [0.147], [0.099]])
k1_L2 = np.array([[4.314], [11.845], [10.03], [5.833], [4.813]])
k2_L2 = np.array([[-0.0106], [-0.0313], [-0.01858], [-0.01192], [-0.00668]])
k3_L2 = np.array([[25.15e-4], [202.0e-4], [1.578e-4], [6.507e-4], [5.695e-4]])
k4_L2 = np.array([[458], [763], [207], [1731], [1531]])
k5_L2 = np.array([[0.986], [3.823], [-5.648], [0.82], [0.842]])
k6_L2 = np.array([[43.03], [-931.3], [2098.0], [53.15], [-7.467]])

weight_2d = weight.reshape(1, N) 
k_ldf = k_ldf_L1 + (k_ldf_L2 - k_ldf_L1) * weight_2d
k1 = k1_L1 + (k1_L2 - k1_L1) * weight_2d; k2 = k2_L1 + (k2_L2 - k2_L1) * weight_2d
k3 = k3_L1 + (k3_L2 - k3_L1) * weight_2d; k4 = k4_L1 + (k4_L2 - k4_L1) * weight_2d
k5 = k5_L1 + (k5_L2 - k5_L1) * weight_2d; k6 = k6_L1 + (k6_L2 - k6_L1) * weight_2d
qm = k1 + k2 * T; B = k3 * np.exp(k4 / T); n_param = k5 + k6 / T

a_ergun_arr = (150 * (1 - eps)**2) / (4 * (dp/2)**2 * eps**3)
b_ergun_arr = (1.75 * (1 - eps)) / (2 * (dp/2) * eps**3)

# =============================================================================
# 3. PDE SOLVER LOGIC
# =============================================================================
@njit(cache=True)
def calc_rhs(t, y, N, P_low, P_high, P_atm_Pa, u_feed, eps, rho_s, dz, MW, Rp, mu, y_feed, k_ldf, qm, B, n_param, R, T, a_ergun_arr, b_ergun_arr):
    res = np.zeros(10 * N); current_P = P_high; current_u = u_feed
    flux_in = (current_u / eps[0]) * (y_feed * (current_P / (R * T)))
    C_safe_arr = np.zeros(5)
    for j in range(N):
        C_total = 0.0
        for i in range(5):
            C_safe_arr[i] = max(y[i * N + j], 1e-10); C_total += C_safe_arr[i]
        if C_total < 1e-3: C_total = 1e-3
        denom = 1.0 + np.sum(B[:, j] * ((C_safe_arr * R * T / P_atm_Pa) ** n_param[:, j]))
        sum_dqdt = 0.0
        for i in range(5):
            q_star = (qm[i, j] * B[i, j] * ((C_safe_arr[i] * R * T / P_atm_Pa) ** n_param[i, j])) / denom
            dq_val = k_ldf[i, j] * (q_star - max(y[5 * N + i * N + j], 0.0))
            if dq_val > 0: dq_val *= 0.5 * (1.0 + np.tanh((y[i * N + j] - 0.01) / 0.01))
            res[5 * N + i * N + j] = dq_val; sum_dqdt += dq_val
        du = -dz * ((1 - eps[j]) * rho_s[j] * sum_dqdt / (current_P / (R * T)))
        next_u = min(max(current_u + du, 1e-6), 5.0)
        rho_gas = np.sum(C_safe_arr * MW)
        for i in range(5):
            flux_out_i = (current_u / eps[j]) * y[i * N + j] 
            res[i * N + j] = -((flux_out_i - flux_in[i]) / dz) - ((1 - eps[j]) / eps[j]) * rho_s[j] * res[5 * N + i * N + j]
            flux_in[i] = flux_out_i 
        if j < N - 1:
            ergun_ramp = (1.0 - np.exp(-t / 5.0))
            dPdz = - (a_ergun_arr[j] * mu * current_u + b_ergun_arr[j] * rho_gas * current_u**2) * ergun_ramp
            current_P = max(current_P + dPdz * dz, P_low * 0.1)
        current_u = next_u
    return res

pbar = tqdm(total=t_end, desc="Simulation Progress", unit="s")
last_t = [0.0]
def pde_layered(t, y):
    if t > last_t[0]: pbar.update(t - last_t[0]); last_t[0] = t
    return calc_rhs(t, y, N, P_low, P_high, P_atm_Pa, u_feed, eps, rho_s, dz, MW, dp/2, mu, y_feed, k_ldf, qm, B, n_param, R, T, a_ergun_arr, b_ergun_arr)

# --- FIX: ROBUST PURE HYDROGEN INITIALIZATION ---
rep_state_file = os.path.join(script_dir, 'repressurization_end_state.npz')
if os.path.exists(rep_state_file) and run_type != "SCOUT":
    data = np.load(rep_state_file)
    y0 = np.concatenate([data['C_end'].flatten(), data['q_end'].flatten()])
    print("-> Loading bed state from Repressurization.")
else:
    print("-> Initializing clean bed with 100% pure Hydrogen.")
    y_pure = np.array([1.0, 1e-10, 1e-10, 1e-10, 1e-10])
    Z_pure = get_Z_PR(P_high, T, y_pure)
    C_pure_total = P_high / (Z_pure * R * T)
    C_init = np.tile(y_pure * C_pure_total, (N, 1)).T
    
    # Calculate pure H2 adsorption loading to prevent numerical shocks
    P_partial_init = np.array([y_pure[i] * P_high * np.ones(N) for i in range(5)])
    q_init = np.zeros((5, N))
    for j in range(N):
        P_i_atm = np.maximum(P_partial_init[:, j], 1e-12) / P_atm_Pa
        denom = 1.0 + np.sum(B[:, j] * (P_i_atm ** n_param[:, j]))
        q_init[:, j] = (qm[:, j] * B[:, j] * (P_i_atm ** n_param[:, j])) / denom

    y0 = np.concatenate([C_init.flatten(), q_init.flatten()])

sol = solve_ivp(pde_layered, [0, t_end], y0, method='BDF', t_eval=t_ads_eval, rtol=1e-2, atol=1e-2)
pbar.close()

# =============================================================================
# 4. TIME-SERIES POST-PROCESSING (Grounded Mass Balance + Surge Tank)
# =============================================================================
t = sol.t; n_t = len(t)
C_history = sol.y[:5*N, :].T.reshape((-1, 5, N))
purity_over_time = np.zeros(n_t); recovery_over_time = np.zeros(n_t)
mass_flow_out_H2_gross = np.zeros(n_t); mass_flow_out_H2_net = np.zeros(n_t)
mass_flow_purge_H2 = np.zeros(n_t) 

n_dot_in = y_feed[0] * C_in_total * u_feed * A * np.ones_like(t)
moles_in_array = [np.trapz(n_dot_in[:i+1], t[:i+1]) for i in range(n_t)]
n_H2_inventory = np.sum(sol.y[:N, 0] * dz * A * eps) + np.sum(sol.y[5*N:6*N, 0] * dz * A * (1 - eps) * rho_s)

# Repressurization Debt
n_H2_rep_penalty = ( (A * L * np.mean(eps)) * C_in_total * ((P_high - P_low)/P_high) ) + np.sum(np.mean(sol.y[5*N:6*N, 0]) * dz * A * (1-eps) * rho_s)

for i in range(1, n_t):
    C_at_t = C_history[i]; q_at_t = sol.y[5*N:, i].reshape((5, N))
    C_H2_exit = C_at_t[0, -1]; C_total_exit = np.sum(C_at_t[:, -1])
    purity_over_time[i] = (C_H2_exit / C_total_exit) * 100 if C_total_exit > 1e-9 else 100.0
    
    delta_storage = (np.sum(C_at_t[0, :] * dz * A * eps) + np.sum(q_at_t[0, :] * dz * A * (1-eps) * rho_s)) - n_H2_inventory
    moles_out_gross = moles_in_array[i] - delta_storage
    
    # --- STRICT PENALTY (No Surge Tank) ---
    moles_out_net = (moles_out_gross * (1.0 - purge_fraction)) - n_H2_rep_penalty
    recovery_over_time[i] = (max(0, moles_out_net) / max(moles_in_array[i], 1e-9)) * 100
    
    inst_gross_flow = (u_feed * (1.0 - np.exp(-t[i]/5.0))) * A * C_H2_exit * MW[0]
    
    mass_flow_out_H2_gross[i] = inst_gross_flow
    mass_flow_out_H2_net[i] = inst_gross_flow * (1.0 - purge_fraction)
    mass_flow_purge_H2[i] = inst_gross_flow * purge_fraction


# =============================================================================
# 5. ALL DIAGNOSTIC PLOTS (Purity/Recovery, Mass Flow, Spatial, Pressure)
# =============================================================================

# --- 5.1 Performance Plot (Always) ---
fig, (ax1, ax_mass) = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
ax1.plot(t, purity_over_time, 'b-', label='Purity (η)')
ax1.set_ylabel('Purity (%)', color='b'); ax1.set_ylim(40, 105)
ax2 = ax1.twinx(); ax2.plot(t, recovery_over_time, 'g--', label='Net Recovery (φ)')
ax2.set_ylabel('Recovery (%)', color='g'); ax2.set_ylim(40, 105)
ax1.set_title(f'[{run_type}] Performance: Purity & Recovery'); ax1.grid(True, linestyle=':')

ax_mass.plot(t, np.full_like(t, u_feed * A * C_in[0] * MW[0]), 'k--', label='H2 Feed Capacity')
ax_mass.plot(t, mass_flow_out_H2_gross, 'gray', linestyle=':', label='Gross H2 Out')
ax_mass.plot(t, mass_flow_out_H2_net, 'r-', linewidth=2, label='Net H2 Product')
ax_mass.set_title('Instantaneous Mass Flow Rates'); ax_mass.set_ylabel('Mass Flow (kg/s)'); ax_mass.legend(); ax_mass.grid(True)
fig.savefig(os.path.join(cycle_folder, "ads_performance.png"))

# --- 5.2 Standalone Breakthrough Curve (For SCOUT Mode) ---
if run_type == "SCOUT":
    fig_break, ax_break = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for i in range(5):
        ax_break.plot(t, np.maximum(C_history[:, i, -1]/C_in[i], 0.0), label=labels[i], color=colors[i], linewidth=2.5)
    ax_break.set_title('Gas Breakthrough Curve (z = L)')
    ax_break.set_xlabel('Time (s)')
    ax_break.set_ylabel('C/C0')
    ax_break.set_xlim(0, t_end)
    ax_break.legend()
    ax_break.grid(True)
    fig_break.savefig(os.path.join(cycle_folder, "ads_breakthrough_curve.png"))

# --- 5.3 Spatial Profiles & Ergun Pressure (For CSS/FINAL Mode) ---
if run_type != "SCOUT":
    # A. 2D Spatial Grid (Including the Breakthrough Curve in Slot 0)
    fig2d, axes2d = plt.subplots(3, 2, figsize=(14, 15), constrained_layout=True)
    axes_flat = axes2d.flatten()
    
    # Breakthrough Curve
    ax_break = axes_flat[0]
    for i in range(5):
        ax_break.plot(t, np.maximum(C_history[:, i, -1]/C_in[i], 0.0), label=labels[i], color=colors[i], linewidth=2.5)
    ax_break.set_title('A. Gas Breakthrough Curve (z = L)')
    ax_break.set_xlabel('Time (s)')
    ax_break.set_ylabel('C/C0')
    ax_break.set_xlim(0, t_end) 
    ax_break.legend()
    ax_break.grid(True)

    # Spatial Molar Profiles
    time_points_sec = [0.4*t_end, 0.6*t_end, 0.8*t_end, 1.0*t_end]
    time_indices = [int(np.argmin(np.abs(t - t_sec))) for t_sec in time_points_sec]
    cmap_lines = plt.cm.plasma(np.linspace(0, 0.8, len(time_indices)))
    
  # Reshape the 1D sigmoid weight array into a 2D format for imshow
    background_gradient = weight.reshape(1, -1)
    
    for i in range(5):
        ax = axes_flat[i+1]
        
        # 1. Plot the standard breakthrough data FIRST so it scales naturally
        for idx, t_idx in enumerate(time_indices):
            y_local = C_history[t_idx, i, :] / np.maximum(np.sum(C_history[t_idx], axis=0), 1e-10)
            ax.plot(z_nodes, y_local, label=f't={t[t_idx]:.0f}s', color=cmap_lines[idx], linewidth=2)
            
        # 2. Capture the natural auto-scaled Y-limits for this specific gas
        natural_ylim = ax.get_ylim()
        
        # 3. Paint the sigmoid background behind everything (zorder=0)
        ax.imshow(background_gradient, aspect='auto', extent=[0, L, 0, 1], 
                  transform=ax.get_xaxis_transform(), cmap='coolwarm', alpha=0.15, zorder=0)
        
        # Optional: Keep a faint marker at the exact theoretical midpoint
        ax.axvline(x=L * ratio_layer1, color='black', linestyle=':', linewidth=1.0, alpha=0.4, zorder=1)
        
        # 4. RESTORE the natural limits so your trace gases don't get squashed!
        ax.set_ylim(natural_ylim)
            
        ax.set_title(f'Molar Frac: {labels[i]}')
        ax.set_xlabel('Column Length z (m)')
        ax.set_ylabel('Mol Fraction') 
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.5)
        ax.set_xlim(0, L)

    fig2d.savefig(os.path.join(cycle_folder, "ads_profiles_2d.png"))

    # B. Ergun Pressure Profile
    def get_pressure_profile(idx):
        t_val = t[idx]
        C_at_t = C_history[idx]
        P_total = np.zeros(N)
        P_total[0] = P_high
        current_u = u_feed
        ergun_ramp = (1.0 - np.exp(-t_val / 5.0))
        for j in range(N - 1):
            C_safe = np.maximum(C_at_t, 1e-12)
            rho_gas = np.sum(C_safe[:, j] * MW)
            dPdz = - (a_ergun_arr[j] * mu * current_u + b_ergun_arr[j] * rho_gas * current_u * abs(current_u)) * ergun_ramp
            P_total[j+1] = max(P_total[j] + dPdz * dz, P_low * 0.9)
        return P_total

    fig_p, ax_p = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for idx, t_idx in enumerate(time_indices):
        P_profile = get_pressure_profile(t_idx)
        ax_p.plot(z_nodes, P_profile / 1e5, label=f't={t[t_idx]:.0f}s', color=cmap_lines[idx], linewidth=2.5)

    ax_p.axvline(x=L * ratio_layer1, color='black', linestyle=':', linewidth=1.5, label='Layer Boundary')
    ax_p.set_title('Ergun Pressure Profile Along the Column', fontsize=14, fontweight='bold')
    ax_p.set_xlabel('Column Length z (m)', fontsize=12)
    ax_p.set_ylabel('Pressure (bar)', fontsize=12)
    ax_p.legend()
    ax_p.grid(True, linestyle='--', alpha=0.6)
    ax_p.set_xlim(0, L)
    fig_p.savefig(os.path.join(cycle_folder, "ads_pressure_profile.png"))

# =============================================================================
# 6. REPORTING & STATE SAVING
# =============================================================================
# FIX: Dynamically evaluate at the exact end of the simulation instead of a hardcoded 400s
eval_idx = -1
purity = purity_over_time[eval_idx]; recovery = recovery_over_time[eval_idx]

if run_type == "SCOUT":
    valid = np.where(purity_over_time >= 99.92)[0]
    breakthrough_time = t[valid[-1]] if len(valid) > 0 else 0
    config["t_op_ads"] = float(breakthrough_time * t_ads_safety_ratio)
    t_tot = config["t_op_ads"] / config["Adsorption_Ratio"]
    config["tf_des"] = config["Purge_Ratio"] * t_tot
    config["t_blowdown_end"] = config["Blowdown_Ratio"] * t_tot
    config["t_rep"] = config["Repress_Ratio"] * t_tot
    with open(config_path, "w") as f: json.dump(config, f, indent=4)

elif run_type in ["CSS", "FINAL"]: 
    net_ton_year = (moles_in_array[eval_idx] * MW[0] / t[eval_idx]) * (recovery/100) * 3600*24*330*Nsets / 1e3
    with open(os.path.join(cycle_folder, "adsorption_summary.txt"), "w") as f:
        f.write(f"Cycle: {current_cycle}\nPurity: {purity:.2f} %\nRecovery: {recovery:.1f} %\nAnnual: {net_ton_year:.1f} ton/yr\nRequired Molar In (H2): {required_molar_in_H2*Nsets*3600:.2f} mol/h\nRequired Molar In (Total): {required_molar_in_total*Nsets*3600:.2f} mol/h\n")
    
    ads_state_file = os.path.join(script_dir, 'adsorption_end_state.npz')
    np.savez(ads_state_file,
             L=L, 
             T=T, 
             R=R, 
             P_end=P_high, 
             d=d,
             C_end=sol.y[:5*N, eval_idx],  
             q_end=sol.y[5*N:, eval_idx],  
             final_mass_flow_purge=mass_flow_purge_H2[eval_idx])
    
    print(f"✅ Adsorption State Saved. Recovery: {recovery:.1f}%")

plt.close('all'); os._exit(0)
