import json
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re

# --- 1.2 DEFINE INITIAL MASTER PARAMETERS ---
Phigh = 2* 101325  
Plow = 0.03 * 101325   
feed_input= 1.27e4 #kmol/h
feed = feed_input*1000/3600 #mol/s
Nsets = 15
CSSHALF = 0.5
Rinse = 0.22
master_params = {
    # Bed & System Parametersmaster VPSA.py
    'feed_molar_flow': feed/Nsets,
    "u_feed_rinse": 0.27,
    "L": 14,
    "d":3.6, 
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


# =============================================================================
# DESORPTION SUMMARY PARSER (drives flow_prod / flow_rinse from the model)
# =============================================================================
def parse_desorption_summary(path):
    """Parse desorption_summary.txt and return key per-bed-per-cycle CO2 amounts (mol)."""
    with open(path, "r") as f:
        text = f.read()
    def grab(label):
        m = re.search(rf"{label}:\s*([0-9.eE+\-]+)\s*mol", text)
        return float(m.group(1)) if m else None
    return {
        "fresh_co2_fed":   grab("Fresh CO2 Fed"),
        "co2_rinsed_in":   grab("Pure CO2 Rinsed In"),
        "co2_vacuumed":    grab("Total CO2 Vacuumed"),
        "net_co2_produced":grab("NET CO2 PRODUCED"),
    }


# =============================================================================
# 22-PAIR STAGGERED SUPERPOSITION (event-driven schedule, mod-time shifted)
# =============================================================================
def run_staggered_superposition(cycle_time, t_prod_start, t_prod_dur, flow_prod,
                                t_rinse_start, t_rinse_dur, flow_rinse,
                                n_pairs=Nsets, n_cycles=2, dt=1, out_dir=None,
                                extra_gantt_events=None,
                                buffer_P_Pa=1e5, buffer_T_K=308.0,
                                R_const=8.314):
 
    stagger_interval = cycle_time / n_pairs
    sim_duration = n_cycles * cycle_time
    t = np.arange(0, sim_duration + dt, dt)

    t_prod_end = t_prod_start + t_prod_dur
    t_rinse_end = t_rinse_start + t_rinse_dur

    pair_columns = {}
    pair_gross_columns = {}
    for i in range(n_pairs):
        t_local = (t - i * stagger_interval) % cycle_time
        prod_mask = (t_local >= t_prod_start) & (t_local < t_prod_end)
        rinse_mask = (t_local >= t_rinse_start) & (t_local < t_rinse_end)
        flow_net = np.where(prod_mask, flow_prod,
                            np.where(rinse_mask, flow_rinse, 0.0))
        flow_gross = np.where(prod_mask, flow_prod, 0.0)  # production only — buffer-tank sizing
        pair_columns[f"Bed_{i+1}"] = flow_net
        pair_gross_columns[f"Bed_{i+1}_Gross"] = flow_gross

    df = pd.DataFrame(pair_columns, index=pd.Index(t, name="Time_s"))
    df_gross = pd.DataFrame(pair_gross_columns, index=df.index)
    df["Total_Net_Flow"] = df.sum(axis=1)
    df["Total_Gross_Flow"] = df_gross.sum(axis=1)

    dead_time = cycle_time - t_prod_dur - t_rinse_dur
    avg_net = df["Total_Net_Flow"].mean()
    avg_gross = df["Total_Gross_Flow"].mean()
    max_flow = df["Total_Gross_Flow"].max()
    min_flow = df["Total_Gross_Flow"].min()

    # ---- Buffer-tank sizing against the NET header (rinse drawn from header) ----
    net = df["Total_Net_Flow"].values
    imbalance_net = net - avg_net           # constant demand = avg_net  (closed cycle)
    inventory_net = np.cumsum(imbalance_net) * dt
    holdup_net_mol = inventory_net.max() - inventory_net.min()
    V_net_m3 = holdup_net_mol * R_const * buffer_T_K / buffer_P_Pa
    df["Inventory_Net_mol"] = inventory_net

    # ---- Buffer-tank sizing against GROSS production (rinse from a dedicated tap) ----
    gross = df["Total_Gross_Flow"].values
    imbalance_gross = gross - avg_gross
    inventory_gross = np.cumsum(imbalance_gross) * dt
    holdup_gross_mol = inventory_gross.max() - inventory_gross.min()
    V_gross_m3 = holdup_gross_mol * R_const * buffer_T_K / buffer_P_Pa
    df["Inventory_Gross_mol"] = inventory_gross

    print("\n" + "=" * 50)
    print(f" {n_pairs}-BED STAGGERED SUPERPOSITION")
    print("=" * 50)
    print(f"Stagger Interval (offset):     {stagger_interval:>8.2f} s")
    print(f"Total Dead Time per Bed:       {dead_time:>8.2f} s")
    print(f"Average Gross Production Flow: {avg_gross:>8.3f} mol/s   (for buffer-tank sizing)")
    print(f"Average Net Header Flow:       {avg_net:>8.3f} mol/s   (gross − rinse)")
    print(f"Max Instantaneous Gross Flow:  {max_flow:>8.3f} mol/s")
    print(f"Min Instantaneous Gross Flow:  {min_flow:>8.3f} mol/s")
    print("-" * 50)
    print(" BUFFER-TANK SIZING (matching mean demand)")
    print(f"  Buffer reference: P={buffer_P_Pa/1e5:.2f} bar, T={buffer_T_K-273.15:.1f} °C")
    print(f"  Gross-side holdup:   {holdup_gross_mol:>10.1f} mol  →  V = {V_gross_m3:>7.2f} m³")
    print(f"  Net-side holdup:     {holdup_net_mol:>10.1f} mol  →  V = {V_net_m3:>7.2f} m³")
    print("=" * 50 + "\n")

    fig, (ax_top, ax_bot, ax_inv) = plt.subplots(
        3, 1, figsize=(14, 12), sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

    prod_color = "#1f77b4"
    rinse_color = "#d62728"
    bar_height = 0.8

    gantt_steps = [
        ("Production", t_prod_start, t_prod_dur, prod_color),
        ("Rinse",      t_rinse_start, t_rinse_dur, rinse_color),
    ]
    if extra_gantt_events:
        for ev in extra_gantt_events:
            gantt_steps.append((ev["name"], ev["start"], ev["dur"], ev["color"]))

    for i in range(n_pairs):
        y = i + 1
        for c in range(n_cycles):
            offset = i * stagger_interval + c * cycle_time
            for _, s, dur, color in gantt_steps:
                ax_top.broken_barh(
                    [(offset + s, dur)],
                    (y - bar_height / 2, bar_height),
                    facecolors=color,
                )

    ax_top.set_ylim(0.5, n_pairs + 0.5)
    ax_top.set_yticks(range(1, n_pairs + 1))
    ax_top.set_yticklabels([f"Bed {i}" for i in range(1, n_pairs + 1)])
    ax_top.set_ylabel("Bed")
    ax_top.set_title(f"{n_pairs}-Bed Staggered Cycle Gantt")
    ax_top.grid(axis="x", alpha=0.3)
    ax_top.legend(handles=[Patch(facecolor=c, label=name)
                           for name, _, _, c in gantt_steps],
                  loc="upper right")

    ax_bot.fill_between(t, 0, df["Total_Gross_Flow"].values,
                        color="#2ca02c", alpha=0.35)
    ax_bot.plot(t, df["Total_Gross_Flow"].values, color="#2ca02c", linewidth=1.2,
                label="Total Gross Production")
    ax_bot.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_bot.axhline(avg_gross, color="green", linestyle="--", linewidth=1.2,
                   label=f"Mean (gross) = {avg_gross:.2f} mol/s")
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("Gross Production Flow (mol/s)")
    ax_bot.set_title(f"Aggregate Gross Production into Buffer Tank ({n_pairs} beds)")
    ax_bot.grid(alpha=0.3)
    ax_bot.legend(loc="upper right")
    ax_bot.set_xlim(0, sim_duration)

    # --- Buffer inventory subplot ---
    ax_inv.plot(t, inventory_net,   color="#1f77b4", linewidth=1.4,
                label=f"Net inventory  (Δ={holdup_net_mol:.0f} mol, V={V_net_m3:.2f} m³)")
    ax_inv.plot(t, inventory_gross, color="#2ca02c", linewidth=1.0, linestyle="--",
                label=f"Gross inventory (Δ={holdup_gross_mol:.0f} mol, V={V_gross_m3:.2f} m³)")
    ax_inv.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax_inv.set_xlabel("Time (s)")
    ax_inv.set_ylabel("Buffer inventory (mol)")
    ax_inv.set_title(f"Buffer Tank Holdup vs Time  (P={buffer_P_Pa/1e5:.2f} bar, T={buffer_T_K-273.15:.0f} °C)")
    ax_inv.grid(alpha=0.3)
    ax_inv.legend(loc="upper right")

    plt.tight_layout()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, "VPSA_22pair_superposition.png")
        csv_path = os.path.join(out_dir, "VPSA_22pair_flow.csv")
        fig.savefig(fig_path, dpi=150)
        df.to_csv(csv_path)
        print(f"📊 Saved 22-pair superposition figure: {fig_path}")
        print(f"💾 Saved 22-pair flow DataFrame:       {csv_path}")
    plt.close(fig)
    return df

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

            # --- N-PAIR STAGGERED SUPERPOSITION (only valid after CSS) ---
            # Cycle order in master: ads → rinse → blowdown (des) → repress
            t_bd_start = final_t_ads + final_t_rinse
            t_rep_start = t_bd_start + final_t_bd

            # Derive flow_prod / flow_rinse from this cycle's desorption summary
            summary_path = os.path.join(
                script_dir, "results", f"CSS_cycle_{cycle}", "desorption_summary.txt"
            )
            flow_prod_derived = 11.87   # fallback if parser fails
            flow_rinse_derived = -8.00
            if os.path.exists(summary_path):
                s = parse_desorption_summary(summary_path)
                if s["net_co2_produced"] and final_t_ads > 0:
                    flow_prod_derived = s["net_co2_produced"] / final_t_ads
                if s["co2_rinsed_in"] and final_t_rinse > 0:
                    flow_rinse_derived = -s["co2_rinsed_in"] / final_t_rinse
                print(f"\n📑 Derived from {summary_path}:")
                print(f"   flow_prod  = NET CO2 produced / t_ads   = {flow_prod_derived:+.3f} mol/s")
                print(f"   flow_rinse = -CO2 rinsed in   / t_rinse = {flow_rinse_derived:+.3f} mol/s")
            else:
                print(f"\n⚠️  {summary_path} not found — using fallback flows {flow_prod_derived}/{flow_rinse_derived} mol/s")

            # Each "set" in Nsets is a bed-pair (2 beds in anti-phase).
            # For the header-flow superposition we model every individual bed,
            # so total beds = 2 * Nsets and stagger interval = T_cycle / (2*Nsets).
            n_pairs_plant = 2 * int(final_config.get("Nsets", 22))

            run_staggered_superposition(
                cycle_time=total_cycle_time,
                t_prod_start=0.0,
                t_prod_dur=final_t_ads,
                flow_prod=flow_prod_derived,
                t_rinse_start=final_t_ads,
                t_rinse_dur=final_t_rinse,
                flow_rinse=flow_rinse_derived,
                n_pairs=n_pairs_plant,
                n_cycles=2,
                dt=1,
                out_dir=os.path.join(script_dir, "results"),
                extra_gantt_events=[
                    {"name": "Blowdown", "start": t_bd_start,  "dur": final_t_bd,  "color": "#7f7f7f"},
                    {"name": "Repress",  "start": t_rep_start, "dur": final_t_rep, "color": "#ff7f0e"},
                ],
            )

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
