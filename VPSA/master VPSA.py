import json
import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 1. SETUP PATHS
script_dir = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# CSS TIMING GANTT CHART
# =============================================================================
def plot_css_gantt(t_ads, t_rinse, t_bd, t_rep, total_cycle, save_path):
    """
    Render a 2-row Gantt chart for one bed pair operating in cyclic steady
    state. Bed A starts adsorbing at t = 0; Bed B is offset by half a
    cycle (= total_cycle / 2) and is therefore regenerating while Bed A
    adsorbs.

    If ads_time != rinse + bd + repress an 'Idle' segment is inserted to
    keep both beds on the same cycle clock.
    """
    bed_cycle = t_ads + t_rinse + t_bd + t_rep
    half = bed_cycle / 2.0

    # ----- Bed A schedule (starts at t = 0) -----
    seq_A = [
        ('Adsorption', t_ads,   '#1f77b4'),
        ('Rinse',      t_rinse, '#2ca02c'),
        ('Blowdown',   t_bd,    '#d62728'),
        ('Repress',    t_rep,   '#ff7f0e'),
    ]
    # ----- Bed B schedule: same sequence but starts at +half cycle -----
    # Reorder so it begins with the phase that lands at t = 0 in the wrapped view.
    # For Bed B at t = 0, we are 'half' seconds into Bed A's cycle.
    # Walk Bed A's sequence and find which phase Bed B is in at t = 0.
    cumulative = 0.0
    seq_B_offset = 0.0
    seq_B_start_idx = 0
    for i, (_, dur, _) in enumerate(seq_A):
        if cumulative + dur > half:
            seq_B_start_idx = i
            seq_B_offset = half - cumulative
            break
        cumulative += dur
    # Bed B starts mid-phase at index seq_B_start_idx with seq_B_offset already elapsed
    seq_B = []
    name, dur, color = seq_A[seq_B_start_idx]
    seq_B.append((name, dur - seq_B_offset, color))
    for k in range(1, 4):
        seq_B.append(seq_A[(seq_B_start_idx + k) % 4])
    # The last phase wraps; trim it to end exactly at bed_cycle
    consumed = sum(d for _, d, _ in seq_B[:-1])
    name_last, _, color_last = seq_B[-1]
    seq_B[-1] = (name_last, bed_cycle - consumed, color_last)

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(13, 4.5), tight_layout=True)
    bed_y = {'Bed A': 11, 'Bed B': 1}
    bar_h = 8

    seen = set()

    def draw_sequence(ax, seq, y, bed_label):
        t = 0.0
        for name, dur, color in seq:
            label = name if name not in seen else None
            seen.add(name)
            ax.broken_barh([(t, dur)], (y, bar_h), facecolors=color,
                           edgecolor='black', linewidth=0.7, label=label)
            if dur > bed_cycle * 0.025:  # only label segments wide enough
                txt_color = 'white' if name in ('Adsorption', 'Blowdown') else 'black'
                ax.text(t + dur / 2, y + bar_h / 2, f'{name}\n{dur:.0f} s',
                        ha='center', va='center',
                        fontsize=8.5, color=txt_color, fontweight='bold')
            t += dur

    draw_sequence(ax, seq_A, bed_y['Bed A'], 'Bed A')
    draw_sequence(ax, seq_B, bed_y['Bed B'], 'Bed B')

    # Half-cycle marker
    ax.axvline(half, linestyle='--', linewidth=1.0, color='grey', alpha=0.7)
    ax.text(half, bed_y['Bed A'] + bar_h + 0.5,
            f'½ cycle\nt = {half:.0f} s', ha='center', va='bottom', fontsize=8,
            color='grey')

    ax.set_yticks([bed_y['Bed B'] + bar_h / 2, bed_y['Bed A'] + bar_h / 2])
    ax.set_yticklabels(['Bed B', 'Bed A'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Time within one full cycle (s)', fontsize=11)
    ax.set_xlim(0, bed_cycle)
    ax.set_ylim(-1, bed_y['Bed A'] + bar_h + 3.5)
    ax.set_title(
        f'VPSA Cyclic Steady-State Schedule — Bed Pair (one full cycle = {bed_cycle:.0f} s)',
        fontsize=12, fontweight='bold')
    ax.grid(True, axis='x', linestyle=':', alpha=0.5)

    # Legend
    legend_phases = [
        Patch(facecolor='#1f77b4', edgecolor='black', label=f'Adsorption ({t_ads:.0f} s)'),
        Patch(facecolor='#2ca02c', edgecolor='black', label=f'Rinse ({t_rinse:.0f} s)'),
        Patch(facecolor='#d62728', edgecolor='black', label=f'Blowdown ({t_bd:.0f} s)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label=f'Repress ({t_rep:.0f} s)'),
    ]
    ax.legend(handles=legend_phases, loc='upper center',
              bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=10)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


ads_script = "VPSA(ads,New).py" 
des_script = "VPSA(Depressurization).py"
rep_script = "VPSA(Repressurization).py"
rinse_script = "VPSA(Rinse).py"
rinse_path = os.path.join(script_dir, rinse_script)
ads_path = os.path.join(script_dir, ads_script)
des_path = os.path.join(script_dir, des_script)
rep_path = os.path.join(script_dir, rep_script)


def wipe_states():
    print("\n--- Cleaning up old bed states ---")
    files_to_delete = [
        "adsorption_end_state.npz",
        "desorption_end_state.npz",
        "rinse_end_state.npz",
        "previous_cycle_state.npz",
        "repressurization_end_state.npz"
    ]
    for file in files_to_delete:
        file_path = os.path.join(script_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ Deleted: {file}")
    print("✨ Bed is clean.")

# --- 1.2 DEFINE INITIAL MASTER PARAMETERS ---
Phigh = 2* 101325  
Plow = 0.03 * 101325   
feed_input= 1.456e4 #kmol/h
feed = feed_input*1000/3600 #mol/s
Nsets = 22
CSSHALF = 0.5
Rinse = 0.22
master_params = {
    # Bed & System Parametersmaster VPSA.py
    'feed_molar_flow': feed/Nsets,
    "u_feed_rinse": 0.25,
    "L": 14,
    "d":3.05, 
    "T": 35 + 273.15,
    "R": 8.314,
    "P_high": Phigh,
    'P_mid' : Phigh,   
    "P_low": Plow,
    "Nsets": Nsets,
    "N": 100,
    "dp": 0.0015,
    "mu": 1.135086e-05,
    "P_atm_Pa": 101325.0,
    
    # SINGLE MATERIAL CONSTANTS (Replaced layers)
    "eps": 0.3987,
    "rho_s": 1230,
    
    # Ratios used for auto-scaling phase times
    "Adsorption_Ratio": 0.5,
    'Rinse_Ratio':Rinse,
    "Blowdown_Ratio": CSSHALF-Rinse-0.1, 
    'Repress_Ratio': 0.1, 

    
   # Time Parameters: Adsorption (Will be optimized by Scout)
    "t_ads_start": 0,
    "t_ads_end": 5000, 
    "t_op_ads": 400,
    "t_ads_safety_ratio": 0.9,
    "tau_bd": 30.0,
}


config_path = os.path.join(script_dir, "master_config.json")
with open(config_path, "w") as f:
    json.dump(master_params, f, indent=4)

# =============================================================================
# PHASE A: THE SCOUT RUN (Cycle 0)
# =============================================================================
print("\n" + "="*50)
print(" PHASE A: SCOUT RUN (FINDING BREAKTHROUGH)")
print("="*50)
wipe_states() 

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"
env["PYTHONUNBUFFERED"] = "1"
env["PSA_CYCLE"] = "0"
env["RUN_TYPE"] = "SCOUT" # Tells the Adsorption script to optimize the config

subprocess.run(["python3", "-u", ads_path], env=env, check=True)

# Read the newly optimized times generated by the Scout run
with open(config_path, "r") as f:
    optimized_config = json.load(f)

print(f"\n---> Scout Complete. Optimized Adsorption Time: {optimized_config['t_op_ads']:.1f} s <---")

# =============================================================================
# PHASE B: THE CYCLIC STEADY STATE (CSS) LOOP
# =============================================================================
wipe_states() # Critical: Wipe the saturated Scout bed so Cycle 1 starts clean!

max_cycles = 50
convergence_tolerance = 1e-3
env["RUN_TYPE"] = "CSS" # Tells the Adsorption script to stop at t_op_ads

for cycle in range(1, max_cycles + 1):
    print(f"\n" + "="*50)
    print(f" STARTING CYCLE {cycle} (CSS LOOP)")
    print("="*50)

    env["PSA_CYCLE"] = str(cycle)
    
    subprocess.run(["python3", "-u", ads_path], env=env, check=True)
    subprocess.run(["python3", "-u", rinse_path], env=env, check=True)
    subprocess.run(["python3", "-u", des_path], env=env, check=True)
    subprocess.run(["python3", "-u", rep_path], env=env, check=True)


    state_file = os.path.join(script_dir, "repressurization_end_state.npz")
    prev_state_file = os.path.join(script_dir, "previous_cycle_state.npz")

    if os.path.exists(prev_state_file) and os.path.exists(state_file):
        current_q = np.load(state_file)['q_end']
        previous_q = np.load(prev_state_file)['q_end']
        
        residual = np.linalg.norm(current_q - previous_q) / np.linalg.norm(previous_q)
        print(f"\n---> Cycle {cycle} Convergence Residual: {residual:.4e} <---")
        
        if residual < convergence_tolerance:
            print("\n✅ CYCLIC STEADY STATE REACHED!")
            
            # --- CAPEX & OPEX TIMING DIAGNOSTICS ---
            with open(config_path, "r") as f:
                final_config = json.load(f)
                
            final_t_ads = final_config.get("t_op_ads", 0)
            final_t_bd = final_config.get("t_blowdown_end", 0)
            final_t_des = final_config.get("tf_des", 0)
            final_t_rep = final_config.get("t_rep", 60.0)
            final_t_rinse = final_config.get("t_rinse", 60.0)
            
            total_cycle_time = final_t_ads + final_t_bd + final_t_des + final_t_rep + final_t_rinse
            
            print("\n" + "="*50)
            print(" FINAL CSS TIMING METRICS (For CapEx/OpEx)")
            print("="*50)
            print(f"Adsorption Time:         {final_t_ads:>6.1f} s")
            print(f"Rinse:                   {final_t_rinse:>6.1f} s")
            print(f"Depressurization (BD):   {final_t_bd:>6.1f} s")
            print(f"Repressurization:        {final_t_rep:>6.1f} s")
            print("-" * 50)
            print(f"TOTAL CYCLE TIME:        {total_cycle_time:>6.1f} s")
            print(f"Total Cycles Simulated:  {cycle}")
            print("="*50 + "\n")
            
            with open(os.path.join(script_dir, "results", "CSS_Timing_Report.txt"), "w") as f:
                f.write("FINAL CSS TIMING METRICS\n")
                f.write(f"Adsorption:     {final_t_ads:.1f} s\n")
                f.write(f"Blowdown:       {final_t_bd:.1f} s\n")
                f.write(f"Rinse:          {final_t_rinse:.1f} s\n")
                f.write(f"Purge:          {final_t_des:.1f} s\n")
                f.write(f"Repress:        {final_t_rep:.1f} s\n")
                f.write(f"Total Cycle:    {total_cycle_time:.1f} s\n")
            # --- CSS schedule Gantt chart for one bed pair ---
            gantt_path = os.path.join(script_dir, "results", "CSS_Gantt_Schedule.png")
            plot_css_gantt(t_ads=final_t_ads,
                           t_rinse=final_t_rinse,
                           t_bd=final_t_bd,
                           t_rep=final_t_rep,
                           total_cycle=total_cycle_time,
                           save_path=gantt_path)
            print(f"📊 CSS Gantt chart saved to {gantt_path}")

            break # Exit the CSS loop
    else:
        print("\n---> Cycle 1 complete. Establishing baseline. <---")
    
    if os.path.exists(state_file):
        data = np.load(state_file)
        np.savez(prev_state_file, q_end=data['q_end'])

# =============================================================================
# PHASE C: THE FINAL DIAGNOSTIC RUN
# =============================================================================
print("\n" + "="*50)
print(" PHASE C: FINAL DIAGNOSTIC RUN (CSS BREAKTHROUGH)")
print("="*50)

env["PSA_CYCLE"] = "FINAL"
env["RUN_TYPE"] = "FINAL" # Tells Adsorption to run to 1500s on the CSS bed

subprocess.run(["python3", "-u", ads_path], env=env, check=True)
print("\n🎉 AUTOMATED WORKFLOW COMPLETE. Check the 'results' folder.")
