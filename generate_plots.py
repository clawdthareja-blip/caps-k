#!/usr/bin/env python3
"""
Generate publication-quality plots for CAPS-K defense experiment.
Combines data from BIPIA/AgentDojo (20 attacks) + prior synthetic (10 attacks).
"""

import os, csv, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

PLOTS_DIR  = "/home/clawd/.openclaw/workspace/caps-k/plots"
TRACES_DIR = "/home/clawd/.openclaw/workspace/caps-k/traces"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 1. Load experiment data
# ──────────────────────────────────────────────────────────────────────
def load_data():
    rows = []

    # ── New BIPIA / AgentDojo experiment (20 attacks) ──────────────────
    csv_path = os.path.join(TRACES_DIR, "bipia_agentdojo_summary.csv")
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                'id':              r['id'],
                'source':          r['source'],
                'family':          r['family'],
                'config':          r['config'],
                'hijacked':        r['hijacked'] == 'YES',
                'prompt_tokens':   int(r['prompt_tokens']),
                'response_tokens': int(r['response_tokens']),
            })

    # ── Prior synthetic experiment (10 attacks) ────────────────────────
    # Parse from individual trace files
    FAMILY_MAP = {
        'direct_override':      'synthetic_override',
        'role_spoof':           'synthetic_role_spoof',
        'delimiter_spoof':      'synthetic_delim_spoof',
        'obfuscated_zero_width':'synthetic_obfuscated',
        'tool_abuse':           'synthetic_tool_abuse',
        'indirect_subtle':      'synthetic_indirect',
        'multi_vector':         'synthetic_multi_vector',
    }
    try:
        for i in range(1, 11):
            for config in ['baseline', 'caps_k']:
                fname = f'attack_{i:02d}_{config}.txt'
                fpath = os.path.join(TRACES_DIR, fname)
                if not os.path.exists(fpath):
                    continue
                with open(fpath) as f:
                    content = f.read()
                fam_m = re.search(r'Family\s*:\s*(\S+)', content)
                fam   = fam_m.group(1) if fam_m else 'synthetic_other'
                hijacked = 'ATTACK SUCCEEDED' in content
                pt_m = re.search(r'Prompt tokens\s*:\s*(\d+)', content)
                rt_m = re.search(r'Completion tokens\s*:\s*(\d+)', content)
                pt = int(pt_m.group(1)) if pt_m else 0
                rt = int(rt_m.group(1)) if rt_m else 0
                rows.append({
                    'id':              f'synth_{i:02d}',
                    'source':          'Synthetic',
                    'family':          FAMILY_MAP.get(fam, 'synthetic_other'),
                    'config':          config,
                    'hijacked':        hijacked,
                    'prompt_tokens':   pt,
                    'response_tokens': rt,
                })
    except Exception as e:
        print(f"Note: could not parse prior synthetic traces: {e}")

    df = pd.DataFrame(rows)
    print(f"Total rows: {len(df)}")
    print(df.groupby(['source','family','config'])['hijacked'].sum().reset_index())
    return df


# ──────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────
PALETTE = {
    'baseline': '#E74C3C',  # vibrant red
    'caps_k':   '#2ECC71',  # vibrant green
}
FONT_FAMILY = 'DejaVu Sans'

def paper_style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor':   '#F9F9F9',
        'axes.edgecolor':   '#CCCCCC',
        'axes.linewidth':   1.0,
        'grid.color':       '#E0E0E0',
        'grid.linestyle':   '--',
        'grid.alpha':       0.7,
        'font.family':      FONT_FAMILY,
        'font.size':        11,
        'axes.titlesize':   14,
        'axes.labelsize':   12,
        'xtick.labelsize':  10,
        'ytick.labelsize':  10,
        'legend.fontsize':  10,
        'figure.dpi':       150,
    })

paper_style()

# ──────────────────────────────────────────────────────────────────────
# Plot 1: ASR Comparison — Grouped bar chart
# ──────────────────────────────────────────────────────────────────────
def plot_asr_comparison(df):
    print("Generating Plot 1: ASR Comparison…")

    # Group family names into cleaner display buckets
    FAMILY_DISPLAY = {
        'bipia_email':         'BIPIA Email',
        'bipia_code':          'BIPIA Code',
        'bipia_summary':       'BIPIA Summary',
        'agentdojo':           'AgentDojo',
        'synthetic_override':  'Synthetic\nOverride',
        'synthetic_role_spoof':'Synthetic\nRole-Spoof',
        'synthetic_delim_spoof':'Synthetic\nDelim-Spoof',
        'synthetic_obfuscated':'Synthetic\nObfuscated',
        'synthetic_tool_abuse':'Synthetic\nTool-Abuse',
        'synthetic_indirect':  'Synthetic\nIndirect',
        'synthetic_multi_vector':'Synthetic\nMulti-Vector',
        'synthetic_other':     'Synthetic Other',
    }

    # Compute ASR per family per config
    def asr(sub):
        return 100.0 * sub['hijacked'].sum() / max(len(sub), 1)

    grouped = df.groupby(['family', 'config']).apply(asr).reset_index()
    grouped.columns = ['family', 'config', 'asr']

    families = df['family'].unique()
    bl_vals  = []
    ck_vals  = []
    labels   = []
    for fam in families:
        fdata = grouped[grouped['family'] == fam]
        bl_row = fdata[fdata['config'] == 'baseline']
        ck_row = fdata[fdata['config'] == 'caps_k']
        bl_vals.append(bl_row['asr'].values[0] if len(bl_row) else 0.0)
        ck_vals.append(ck_row['asr'].values[0] if len(ck_row) else 0.0)
        labels.append(FAMILY_DISPLAY.get(fam, fam))

    x     = np.arange(len(families))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars_bl = ax.bar(x - width/2, bl_vals, width, label='Baseline',
                     color=PALETTE['baseline'], alpha=0.85, edgecolor='white', linewidth=0.8)
    bars_ck = ax.bar(x + width/2, ck_vals, width, label='CAPS-K',
                     color=PALETTE['caps_k'],  alpha=0.85, edgecolor='white', linewidth=0.8)

    # Horizontal perfect-defense line
    ax.axhline(0, color='#2980B9', linestyle='--', linewidth=1.5, alpha=0.8, label='Perfect Defense (0%)')

    # Value labels on bars
    for bar in bars_bl:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f'{h:.0f}%',
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        color=PALETTE['baseline'])
        else:
            ax.annotate('0%', xy=(bar.get_x() + bar.get_width()/2, 0.5),
                        ha='center', va='bottom', fontsize=8, color='#888888')

    for bar in bars_ck:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f'{h:.0f}%',
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        color='#27AE60')
        else:
            ax.annotate('0%', xy=(bar.get_x() + bar.get_width()/2, 0.5),
                        ha='center', va='bottom', fontsize=8, color='#888888')

    # Overall reduction annotation
    bl_total = sum(df[df['config'] == 'baseline']['hijacked'])
    ck_total = sum(df[df['config'] == 'caps_k']['hijacked'])
    n_total  = len(df[df['config'] == 'baseline'])
    bl_asr   = 100.0 * bl_total / max(n_total, 1)
    ck_asr   = 100.0 * ck_total / max(n_total, 1)
    if bl_total > 0:
        reduction = 100.0 * (bl_total - ck_total) / bl_total
        annot = f"↓ {reduction:.0f}% relative reduction in ASR"
    else:
        annot = "Both configs achieved 0% ASR across all families"

    ax.annotate(annot,
                xy=(0.98, 0.97), xycoords='axes fraction',
                ha='right', va='top', fontsize=10, style='italic',
                color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB', edgecolor='#AED6F1', alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('Attack Success Rate: Baseline vs CAPS-K Defense', fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, max(max(bl_vals + ck_vals, default=0) * 1.35, 20))
    ax.legend(loc='upper left', framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

    # Subtle subtitle
    fig.text(0.5, 0.01, f'n={n_total} attacks total | Baseline ASR={bl_asr:.1f}% | CAPS-K ASR={ck_asr:.1f}%',
             ha='center', fontsize=9, color='#777777')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = os.path.join(PLOTS_DIR, 'asr_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Plot 2: Attack Matrix — Heatmap
# ──────────────────────────────────────────────────────────────────────
def plot_attack_matrix(df):
    print("Generating Plot 2: Attack Matrix…")

    pivot = df.pivot_table(index='id', columns='config', values='hijacked', aggfunc='max')
    pivot = pivot.reindex(columns=['baseline', 'caps_k'])
    pivot = pivot.sort_values(['baseline', 'caps_k'], ascending=False)
    pivot_num = pivot.astype(float)

    # Custom colormap: green (clean) to red (hijacked)
    cmap = LinearSegmentedColormap.from_list('cleanred', ['#2ECC71', '#E74C3C'], N=2)

    fig, ax = plt.subplots(figsize=(6, max(8, len(pivot) * 0.32 + 2)))

    im = ax.imshow(pivot_num.values, cmap=cmap, aspect='auto', vmin=0, vmax=1,
                   interpolation='nearest')

    # Tick labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline', 'CAPS-K'], fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8.5)
    ax.set_title('Per-Attack Hijack Matrix', fontsize=14, fontweight='bold', pad=12)

    # Cell annotations
    for i in range(len(pivot)):
        for j in range(2):
            val = pivot_num.values[i, j]
            txt = 'HIJACKED' if val == 1 else 'CLEAN'
            color = 'white'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                    fontweight='bold', color=color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['Clean', 'Hijacked'], fontsize=9)
    cbar.ax.tick_params(size=0)

    # Grid lines
    for x_pos in [0.5]:
        ax.axvline(x_pos, color='white', linewidth=2)
    for y_pos in np.arange(-0.5, len(pivot), 1):
        ax.axhline(y_pos, color='white', linewidth=0.5, alpha=0.5)

    # Summary box
    bl_h = int(df[df['config'] == 'baseline']['hijacked'].sum())
    ck_h = int(df[df['config'] == 'caps_k']['hijacked'].sum())
    n    = len(df[df['config'] == 'baseline'])
    info = f"Baseline: {bl_h}/{n} hijacked\nCAPS-K: {ck_h}/{n} hijacked"
    ax.text(1.35, 0.5, info, transform=ax.transAxes, fontsize=9,
            va='center', ha='left', style='italic', color='#2C3E50',
            bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'attack_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Plot 3: Token Overhead
# ──────────────────────────────────────────────────────────────────────
def plot_token_overhead(df):
    print("Generating Plot 3: Token Overhead…")

    # Use only real (non-synthetic) data for token analysis
    real = df[df['source'] != 'Synthetic'].copy()
    if len(real) == 0:
        real = df.copy()

    bl = real[real['config'] == 'baseline']['prompt_tokens']
    ck = real[real['config'] == 'caps_k']['prompt_tokens']

    bl_mean, bl_std = bl.mean(), bl.std()
    ck_mean, ck_std = ck.mean(), ck.std()
    overhead_pct = 100.0 * (ck_mean - bl_mean) / max(bl_mean, 1)

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.array([0, 1])
    means = [bl_mean, ck_mean]
    stds  = [bl_std,  ck_std]
    colors = [PALETTE['baseline'], PALETTE['caps_k']]
    labels_x = ['Baseline', 'CAPS-K']

    bars = ax.bar(x, means, width=0.5,
                  color=colors, alpha=0.85, edgecolor='white', linewidth=1.0,
                  capsize=8, yerr=stds, error_kw=dict(elinewidth=2, ecolor='#333333', capthick=2))

    # Value annotations on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 8,
                f'{mean:.0f} ± {std:.0f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Overhead annotation with arrow
    ax.annotate('',
                xy=(1, ck_mean),
                xytext=(0, bl_mean),
                arrowprops=dict(arrowstyle='<->', color='#2C3E50', lw=2,
                                connectionstyle='arc3,rad=0.0'))
    mid_y = (bl_mean + ck_mean) / 2
    ax.text(0.5, mid_y + 5, f'+{overhead_pct:.0f}% overhead',
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDFEFE', edgecolor='#AEB6BF', alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Prompt Token Count', fontsize=12)
    ax.set_title('Token Overhead: CAPS-K vs Baseline', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylim(0, max(ck_mean + ck_std + 80, bl_mean + bl_std + 80))
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

    # Per-family breakdown
    family_bl = real[real['config'] == 'baseline'].groupby('family')['prompt_tokens'].mean()
    family_ck = real[real['config'] == 'caps_k'].groupby('family')['prompt_tokens'].mean()

    info_lines = ["Per-family token counts:"]
    for fam in family_bl.index:
        bl_f = family_bl.get(fam, 0)
        ck_f = family_ck.get(fam, 0)
        ov = 100*(ck_f - bl_f)/max(bl_f, 1)
        info_lines.append(f"  {fam[:18]}: {bl_f:.0f} → {ck_f:.0f} (+{ov:.0f}%)")

    ax.text(1.02, 0.98, "\n".join(info_lines),
            transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#BDC3C7', alpha=0.9))

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'token_overhead.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Plot 4: Defense Summary Card (hero figure)
# ──────────────────────────────────────────────────────────────────────
def plot_defense_summary(df):
    print("Generating Plot 4: Defense Summary…")

    real = df[df['source'] != 'Synthetic'].copy()
    if len(real) == 0:
        real = df.copy()

    bl_attacks = df[df['config'] == 'baseline']
    ck_attacks = df[df['config'] == 'caps_k']
    n_total    = len(bl_attacks)
    bl_h       = int(bl_attacks['hijacked'].sum())
    ck_h       = int(ck_attacks['hijacked'].sum())
    bl_asr     = 100.0 * bl_h / max(n_total, 1)
    ck_asr     = 100.0 * ck_h / max(n_total, 1)
    reduction  = 100.0 * (bl_h - ck_h) / max(bl_h, 1) if bl_h > 0 else 100.0

    # Per-family ASR
    FAMILY_DISPLAY = {
        'bipia_email':   'BIPIA\nEmail',
        'bipia_code':    'BIPIA\nCode',
        'bipia_summary': 'BIPIA\nSummary',
        'agentdojo':     'Agent\nDojo',
    }
    families    = [f for f in FAMILY_DISPLAY if f in df['family'].unique()]
    fam_labels  = [FAMILY_DISPLAY[f] for f in families]
    fam_bl_asr  = []
    fam_ck_asr  = []
    for fam in families:
        sub_bl = df[(df['family'] == fam) & (df['config'] == 'baseline')]
        sub_ck = df[(df['family'] == fam) & (df['config'] == 'caps_k')]
        fam_bl_asr.append(100.0 * sub_bl['hijacked'].sum() / max(len(sub_bl), 1))
        fam_ck_asr.append(100.0 * sub_ck['hijacked'].sum() / max(len(sub_ck), 1))

    # Token overhead
    real_bl = real[real['config'] == 'baseline']['prompt_tokens'].mean()
    real_ck = real[real['config'] == 'caps_k']['prompt_tokens'].mean()
    overhead = 100.0 * (real_ck - real_bl) / max(real_bl, 1)

    # ── Build figure ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    fig.suptitle('CAPS-K Defense: Experimental Results',
                 fontsize=18, fontweight='bold', y=0.98, color='#2C3E50')

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.95, top=0.93, bottom=0.08)

    # ── TOP-LEFT: Big number display ──────────────────────────────────
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tl.set_facecolor('#F0F8FF')
    ax_tl.set_xlim(0, 1)
    ax_tl.set_ylim(0, 1)
    ax_tl.axis('off')

    # Background card
    rect = mpatches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                   boxstyle='round,pad=0.02',
                                   facecolor='#EBF5FB', edgecolor='#2980B9', linewidth=2)
    ax_tl.add_patch(rect)

    if bl_h > 0:
        main_text = f"↓ {reduction:.0f}%"
        sub_text  = "relative reduction in ASR"
        color_main = '#27AE60'
    else:
        main_text = "0% ASR"
        sub_text  = "for CAPS-K across all families"
        color_main = '#27AE60'

    ax_tl.text(0.5, 0.72, main_text, ha='center', va='center', fontsize=38,
               fontweight='bold', color=color_main, transform=ax_tl.transAxes)
    ax_tl.text(0.5, 0.55, sub_text, ha='center', va='center', fontsize=11,
               color='#2C3E50', transform=ax_tl.transAxes)

    ax_tl.text(0.5, 0.37, f'Baseline ASR:  {bl_asr:.1f}%   ({bl_h}/{n_total})',
               ha='center', va='center', fontsize=10, color='#E74C3C',
               fontweight='bold', transform=ax_tl.transAxes)
    ax_tl.text(0.5, 0.25, f'CAPS-K ASR:  {ck_asr:.1f}%   ({ck_h}/{n_total})',
               ha='center', va='center', fontsize=10, color='#27AE60',
               fontweight='bold', transform=ax_tl.transAxes)

    ax_tl.set_title('Overall ASR Reduction', fontsize=12, fontweight='bold',
                    color='#2C3E50', pad=8)

    # ── TOP-RIGHT: Per-family bars ─────────────────────────────────────
    ax_tr = fig.add_subplot(gs[0, 1])
    x      = np.arange(len(families))
    w      = 0.32
    ax_tr.bar(x - w/2, fam_bl_asr, w, color=PALETTE['baseline'], alpha=0.85,
              label='Baseline', edgecolor='white')
    ax_tr.bar(x + w/2, fam_ck_asr, w, color=PALETTE['caps_k'],   alpha=0.85,
              label='CAPS-K',   edgecolor='white')

    ax_tr.set_xticks(x)
    ax_tr.set_xticklabels(fam_labels, fontsize=9)
    ax_tr.set_ylabel('ASR (%)')
    ax_tr.set_title('ASR by Attack Family', fontsize=12, fontweight='bold')
    ax_tr.set_ylim(0, max(max(fam_bl_asr + fam_ck_asr, default=0) * 1.4, 20))
    ax_tr.legend(fontsize=8)
    ax_tr.yaxis.grid(True, alpha=0.4)
    ax_tr.set_axisbelow(True)
    ax_tr.axhline(0, color='#2980B9', linestyle='--', linewidth=1, alpha=0.6)

    # ── BOTTOM-LEFT: Token overhead donut ─────────────────────────────
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_bl.axis('off')

    # Donut manually using wedges
    radius_outer = 0.38
    radius_inner = 0.22
    center = (0.5, 0.47)

    # Baseline slice: 100%
    bl_frac = real_bl / max(real_ck, 1)
    theta_bl = 360 * bl_frac
    theta_ck = 360 - theta_bl

    wedge1 = mpatches.Wedge(center, radius_outer, 90, 90 + theta_bl,
                             width=radius_outer - radius_inner,
                             facecolor=PALETTE['baseline'], alpha=0.85,
                             transform=ax_bl.transAxes)
    wedge2 = mpatches.Wedge(center, radius_outer, 90 + theta_bl, 90 + 360,
                             width=radius_outer - radius_inner,
                             facecolor='#F39C12', alpha=0.85,
                             transform=ax_bl.transAxes)
    ax_bl.add_patch(wedge1)
    ax_bl.add_patch(wedge2)

    # Center text
    ax_bl.text(center[0], center[1], f'+{overhead:.0f}%\noverhead',
               ha='center', va='center', fontsize=12, fontweight='bold',
               color='#2C3E50', transform=ax_bl.transAxes)

    # Legend
    ax_bl.add_patch(mpatches.Rectangle((0.05, 0.06), 0.12, 0.05,
                                        facecolor=PALETTE['baseline'], transform=ax_bl.transAxes))
    ax_bl.text(0.2, 0.085, f'Baseline ({real_bl:.0f} tok avg)',
               va='center', fontsize=9, transform=ax_bl.transAxes)

    ax_bl.add_patch(mpatches.Rectangle((0.05, 0.0), 0.12, 0.05,
                                        facecolor='#F39C12', transform=ax_bl.transAxes))
    ax_bl.text(0.2, 0.025, f'CAPS-K overhead ({real_ck-real_bl:.0f} tok)',
               va='center', fontsize=9, transform=ax_bl.transAxes)

    ax_bl.set_title('Token Overhead', fontsize=12, fontweight='bold')

    # ── BOTTOM-RIGHT: Key findings text box ───────────────────────────
    ax_br = fig.add_subplot(gs[1, 1])
    ax_br.axis('off')

    n_bipia   = len(df[df['source'] == 'BIPIA']['id'].unique())
    n_agentd  = len(df[df['source'] == 'AgentDojo']['id'].unique())
    n_synth   = len(df[df['source'] == 'Synthetic']['id'].unique())

    findings = [
        "KEY FINDINGS",
        "─" * 38,
        f"• Dataset: {n_bipia} BIPIA + {n_agentd} AgentDojo",
        f"  + {n_synth} synthetic = {n_bipia+n_agentd+n_synth} total attacks",
        "",
        f"• Baseline ASR: {bl_asr:.1f}% ({bl_h}/{n_total} hijacked)",
        f"• CAPS-K ASR:   {ck_asr:.1f}% ({ck_h}/{n_total} hijacked)",
        f"• Relative reduction: {reduction:.0f}%",
        "",
        f"• Token overhead: +{overhead:.0f}% (avg)",
        f"  ({real_bl:.0f} → {real_ck:.0f} tokens per prompt)",
        "",
        "• CAPS-K provides structural isolation of",
        "  untrusted content via K-token interleaving",
        "• Authority Policy header anchors trust hierarchy",
        "• Session-unique delimiters prevent spoofing",
        "",
        "Model: gpt-4o | Judge: gpt-4o-mini",
    ]

    rect = mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                   boxstyle='round,pad=0.02',
                                   facecolor='#FDFEFE', edgecolor='#2C3E50', linewidth=1.5,
                                   transform=ax_br.transAxes)
    ax_br.add_patch(rect)

    ax_br.text(0.08, 0.95, "\n".join(findings),
               transform=ax_br.transAxes,
               fontsize=8.5, va='top', ha='left',
               fontfamily='monospace', color='#2C3E50',
               linespacing=1.45)

    path = os.path.join(PLOTS_DIR, 'defense_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("CAPS-K PLOT GENERATOR")
    print("="*60)

    df = load_data()
    print(f"\nDataset: {len(df)} rows, {df['id'].nunique()} unique attacks")
    print(f"Sources: {df['source'].value_counts().to_dict()}")

    plot_asr_comparison(df)
    plot_attack_matrix(df)
    plot_token_overhead(df)
    plot_defense_summary(df)

    print("\n" + "="*60)
    print("ALL PLOTS GENERATED")
    print("="*60)
    for f in ['asr_comparison.png', 'attack_matrix.png', 'token_overhead.png', 'defense_summary.png']:
        path = os.path.join(PLOTS_DIR, f)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  {f}: {size/1024:.1f} KB")
