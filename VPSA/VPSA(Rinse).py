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
# 1. SETUP & PATHS
# =============================================================================
run_type = os.environ.get("RUN_TYPE", "CSS") 
current_cycle = os.environ.get("PSA_CYCLE", "1")


script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "results")
cycle_folder = os.path.join(base_dir, f"{run_type}_cycle_{current_cycle}")
os.makedirs(cycle_folder, exist_ok=True)

print(f"\n--- Starting RINSE (Heavy Reflux): [{run_type} MODE] Cycle {current_cycle} ---")

# LOAD ADSORPTION STATE
ads_state_file = os.path.join(script_dir, 'adsorption_end_state.npz')
if not os.path.exists(ads_state_file):
    raise FileNotFoundError("Cannot find adsorption_end_state.npz! Must run Adsorption first.")

# LOAD MASTER CONFIG

config_path = os.path.join(script_dir, "master_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

L = config["L"]; T = config["T"]; R = config["R"]
P_mid = config["P_mid"]; P_low = config["P_low"]
d = config["d"]; Nsets = config["Nsets"]

N = int(config["N"])
dp = float(config.get("dp", 0.002))
mu = float(config.get("mu", 1.135086e-05))
P_atm_Pa = float(config.get("P_atm_Pa", 101325.0))
eps = float(config.get("eps", 0.35))
rho_s = float(config.get("rho_s", 2000))

# Rinse typically runs for a fraction of the adsorption time
t_tot = config.get("t_tot", 100.0) 
ratio = config.get('Rinse_Ratio', 0.1)
t_end = t_tot*ratio
t_eval = np.linspace(0, t_end, 150)

# =============================================================================
# 2. FEED SIZING (100% CO2)
# =============================================================================
A = pi*(d/2)**2 
labels = ['N2', 'CO2', 'O2']
colors = ['gray', 'blue', 'red']
MW = np.array([0.028014, 0.044009, 0.031998])

# THE RINSE FEED: 100% CO2
y_feed = np.array([0.0, 1.0, 0.0]) 

C_in_total = P_mid / (R * T)

# Update: Read superficial velocity directly from config
u_feed = float(config.get("u_feed_rinse", 0.00)) # Default to 0.05 m/s if not found

# Back-calculate molar flow for the PDE solver and reporting
feed_molar_flow = u_feed * A * C_in_total 
C_in = y_feed * C_in_total

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
# 3. PDE SOLVER (With Stability Fixes)
# =============================================================================
@njit(cache=True)
def calc_rhs(t, y, N, P_low, P_mid, u_feed, eps, rho_s, dz, MW, mu, y_feed, k_ldf, q_s, b_toth, R, T, a_ergun_arr, b_ergun_arr):
    res = np.empty(6 * N) 
    current_P = P_mid
    current_u = u_feed
    
    feed_ramp = 1.0 - np.exp(-t / 0.5)
    flux_in_0 = (current_u / eps) * (y_feed[0] * feed_ramp) * (current_P / (R * T))
    flux_in_1 = (current_u / eps) * (y_feed[1] * feed_ramp + (1 - feed_ramp)*0.0036) * (current_P / (R * T))
    flux_in_2 = (current_u / eps) * (y_feed[2] * feed_ramp) * (current_P / (R * T))
    
    mass_transfer_coef = ((1 - eps) / eps) * rho_s
    
    for j in range(N):
        raw_C_0 = y[0 * N + j]
        raw_C_1 = y[1 * N + j]
        raw_C_2 = y[2 * N + j]
        
        # Mole fraction normalization for stable pressure
        C_tot_raw = max(raw_C_0 + raw_C_1 + raw_C_2, 1e-10)
        y_frac_0 = max(raw_C_0, 0.0) / C_tot_raw
        y_frac_1 = max(raw_C_1, 0.0) / C_tot_raw
        y_frac_2 = max(raw_C_2, 0.0) / C_tot_raw
        
        P_0_kPa = (y_frac_0 * current_P) 
        P_1_kPa = (y_frac_1 * current_P) 
        P_2_kPa = (y_frac_2 * current_P) 
        
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
        
        # Velocity Update
        du = -dz * ((1 - eps) * rho_s * sum_dqdt / (current_P / (R * T)))
        next_u = current_u + du
        # Floor the velocity to prevent numerical collapse at the adsorption front.
        # Real physics: when adsorption demand exceeds feed supply, pressure builds
        # back-up to maintain flow; this code doesn't model that pressure feedback,
        # so without a floor the front cell's gas piles up to 1000+ atm.
        u_floor = 0.05 * u_feed
        if next_u < u_floor:
            next_u = u_floor
        
        # Dynamic Upwinding
        if next_u >= 0.0:
            up_C_0, up_C_1, up_C_2 = raw_C_0, raw_C_1, raw_C_2
        else:
            if j < N - 1:
                up_C_0, up_C_1, up_C_2 = y[0*N+j+1], y[1*N+j+1], y[2*N+j+1]
            else:
                up_C_0, up_C_1, up_C_2 = raw_C_0, raw_C_1, raw_C_2
                
        flux_out_0 = (next_u / eps) * up_C_0 
        flux_out_1 = (next_u / eps) * up_C_1 
        flux_out_2 = (next_u / eps) * up_C_2 
        
        res[0 * N + j] = -((flux_out_0 - flux_in_0) / dz) - mass_transfer_coef * dqdt_0
        res[1 * N + j] = -((flux_out_1 - flux_in_1) / dz) - mass_transfer_coef * dqdt_1
        res[2 * N + j] = -((flux_out_2 - flux_in_2) / dz) - mass_transfer_coef * dqdt_2
        
        flux_in_0, flux_in_1, flux_in_2 = flux_out_0, flux_out_1, flux_out_2
        
        # Pressure Update
        if j < N - 1:
            ergun_ramp = (1.0 - np.exp(-t / 2.0))
            rho_gas = (y_frac_0 * MW[0] + y_frac_1 * MW[1] + y_frac_2 * MW[2]) * (current_P / (R * T))
            dPdz = - (a_ergun_arr[j] * mu * current_u + b_ergun_arr[j] * rho_gas * current_u * abs(current_u)) * ergun_ramp
            next_P = current_P + dPdz * dz
            if next_P < P_low * 0.1: next_P = P_low * 0.1
            current_P = next_P
            
        current_u = next_u
    return res

pbar = tqdm(total=t_end, desc="Rinsing Bed", unit="s")
last_t = [0.0]
def pde_layered(t, y):
    if t > last_t[0]: pbar.update(t - last_t[0]); last_t[0] = t
    return calc_rhs(t, y, N, P_low, P_mid, u_feed, eps, rho_s, dz, MW, mu, y_feed, k_ldf, q_s, b_toth, R, T, a_ergun_arr, b_ergun_arr)


data = np.load(ads_state_file)
y0 = np.concatenate([data['C_end'].flatten(), data['q_end'].flatten()])

sol = solve_ivp(pde_layered, [0, t_end], y0, method='BDF', t_eval=t_eval, rtol=1e-3, atol=1e-4, first_step=0.01)
pbar.close()

# =============================================================================
# 4. REPORTING & STATE SAVING
# =============================================================================
# Calculate EXACT analytical integral of the feed ramp to prevent trapezoid error
tau_ramp = 0.5
co2_moles_rinse = feed_molar_flow * (t_end + tau_ramp * (np.exp(-t_end / tau_ramp) - 1.0))

# Mass-balance the rinse step so we know how much CO2 actually slipped out the
# raffinate end (vs how much stayed in the bed). This is the conserved quantity
# we need for an honest cycle recovery; the bed-inventory delta on its own has
# numerical drift that swamps the fresh feed when the rinse is large.
C_rinse_end = sol.y[:3*N, -1].reshape(3, N)
q_rinse_end = sol.y[3*N:, -1].reshape(3, N)
rinse_end_co2_inv = (np.sum(C_rinse_end[1, :]) * (A * dz * eps)
                     + np.sum(q_rinse_end[1, :]) * (A * dz * (1 - eps) * rho_s))

ads_data = np.load(ads_state_file)
ads_end_co2_inv = float(ads_data['ads_end_inventory'][1]) if 'ads_end_inventory' in ads_data.files else 0.0

# CO2 in − ΔCO2_bed = CO2 out the raffinate end
co2_moles_exhaust_rinse = max(co2_moles_rinse - (rinse_end_co2_inv - ads_end_co2_inv), 0.0)

rinse_state_file = os.path.join(script_dir, 'rinse_end_state.npz')
np.savez(rinse_state_file,
         L=L, T=T, R=R, P_end=P_mid, d=d,
         C_end=sol.y[:3*N, -1],
         q_end=sol.y[3*N:, -1],
         co2_moles_rinse=co2_moles_rinse,
         co2_moles_exhaust_rinse=float(co2_moles_exhaust_rinse))

print(f"✅ Rinse State Saved. Ready for Blowdown!")
# =============================================================================
# 5. VISUALIZATION (Separate Gas Phase Species Distributions with Time Snapshots)
# =============================================================================
print("Generating separate gas phase spatial profiles for each species with time snapshots...")

# 1. Reshape gas concentrations: (time_steps, species, nodes)
C_history = sol.y[:3*N, :].T.reshape((len(sol.t), 3, N))

# 2. Calculate gas phase molar fractions for ALL species
C_tot = np.sum(C_history, axis=1)
# Use broadcasting for efficient calculation
y_gas_all = C_history / np.maximum(C_tot[:, np.newaxis, :], 1e-10) 

# 3. Define target times (5 snapshots: 0%, 25%, 50%, 75%, 100%)
target_times = [0.0, sol.t[-1]*0.25, sol.t[-1]*0.5, sol.t[-1]*0.75, sol.t[-1]]
time_indices = [int(np.argmin(np.abs(sol.t - t_sec))) for t_sec in target_times]

# 4. Setup distinct color map bases for each species (shades)
species_cmaps = [plt.cm.Greys, plt.cm.Blues, plt.cm.Reds] # for N2, CO2, O2 respectively

# 5. Loop through each species to create separate plots
for j in range(3):
    # Setup the plot for this specific species
    fig_species, ax_species = plt.subplots(figsize=(8, 6), tight_layout=True)
    
    # Create a shade-based color gradient for the time snapshots for this species plot
    species_shade_cmap = species_cmaps[j](np.linspace(0.4, 0.9, len(time_indices))) 
    
    # Loop through the time snapshots for this species plot
    for idx, t_idx in enumerate(time_indices):
        # Plot distribution for this species and time snapshot, using the shade-based color
        ax_species.plot(z_nodes, y_gas_all[t_idx, j, :] * 100, 
                        label=f't = {sol.t[t_idx]:.1f} s', 
                        color=species_shade_cmap[idx], linewidth=2.5)

    # 6. Formatting the graph for this specific species
    ax_species.set_title(f'Gas Phase {labels[j]} Distribution (Rinse Step)', fontsize=14, fontweight='bold')
    ax_species.set_xlabel('Column Length z (m)', fontsize=12)
    ax_species.set_ylabel('Gas Molar Fraction (%)', fontsize=12)
    ax_species.grid(True, linestyle='--', alpha=0.7)
    # Legend for time snapshots within this species plot
    ax_species.legend(loc='best', title="Time Snapshots", fontsize=10) 
    ax_species.set_ylim(-5, 105) # Consistent y-axis for easy comparison
    ax_species.set_xlim(0, L)

    # 7. Save the figure with dynamic filename
    species_profile_path = os.path.join(cycle_folder, f"rinse_{labels[j].lower()}_gas_profile.png")
    fig_species.savefig(species_profile_path, dpi=150)
    print(f"✅ Gas phase {labels[j]} profile saved to {species_profile_path}")

    # Close the plot to free up memory
    plt.close(fig_species)

print("---------------------\n")