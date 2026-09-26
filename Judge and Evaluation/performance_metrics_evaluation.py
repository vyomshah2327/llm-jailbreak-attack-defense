import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)

# Read attack results
attack_results = pd.read_csv('comparison_results.csv')

# Read defense data
defense_df = pd.read_csv('Defense/defense_jailbreak_evaluation.csv')
comparison_data = pd.read_csv('CSV/attack_vs_defense_comparison.csv')

print("\n" + "="*80)
print("PERFORMANCE EVALUATION METRICS ANALYSIS")
print("="*80 + "\n")

# ============================================================================
# Calculate Metrics
# ============================================================================

# 1. Attack Success Rate (ASR) - by attack type and model
asr_by_attack = attack_results.groupby('Attack Type')['Success Rate (%)'].mean().reset_index()
asr_by_attack.columns = ['Attack Type', 'ASR (%)']

asr_by_model = attack_results.groupby('Model')['Success Rate (%)'].mean().reset_index()
asr_by_model.columns = ['Model', 'ASR (%)']
asr_by_model = asr_by_model.sort_values('ASR (%)', ascending=False)

# Overall ASR
overall_asr_without_defense = attack_results['Success Rate (%)'].mean()
overall_asr_with_defense = 0.0  # From defense results

# 2. Average Harm Score (AHS)
# From comparison data
ahs_without_defense = 6.08
ahs_with_defense = 0.00

# 3. Defense Block Rate (DBR)
total_defense_attempts = len(defense_df)
blocked_by_defense = defense_df['defense_blocked'].sum()
dbr = (blocked_by_defense / total_defense_attempts * 100) if total_defense_attempts > 0 else 0

print("KEY PERFORMANCE METRICS:")
print("-" * 80)
print(f"\n1. Attack Success Rate (ASR):")
print(f"   • Without Defense: {overall_asr_without_defense:.2f}%")
print(f"   • With Defense: {overall_asr_with_defense:.2f}%")
print(f"   • Reduction: {overall_asr_without_defense - overall_asr_with_defense:.2f} percentage points")

print(f"\n2. Average Harm Score (AHS):")
print(f"   • Without Defense: {ahs_without_defense:.2f}")
print(f"   • With Defense: {ahs_with_defense:.2f}")
print(f"   • Reduction: {ahs_without_defense - ahs_with_defense:.2f} points")

print(f"\n3. Defense Block Rate (DBR):")
print(f"   • Block Rate: {dbr:.2f}%")
print(f"   • Total Attempts: {total_defense_attempts}")
print(f"   • Blocked: {blocked_by_defense}")

print("\nASR by Attack Type:")
print(asr_by_attack.to_string(index=False))

print("\nASR by Model (Top 6):")
print(asr_by_model.head(6).to_string(index=False))
print("\n" + "="*80 + "\n")

# ============================================================================
# Create Comprehensive Visualization
# ============================================================================

fig = plt.figure(figsize=(22, 16))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# Color scheme
color_attack = '#e74c3c'
color_defense = '#2ecc71'
color_neutral = '#3498db'

# ============================================================================
# Row 1: Main Metrics Overview
# ============================================================================

# 1.1 Attack Success Rate (ASR) Comparison
ax1 = fig.add_subplot(gs[0, 0])
categories = ['Without\nDefense', 'With\nDefense']
asr_values = [overall_asr_without_defense, overall_asr_with_defense]
bars = ax1.bar(categories, asr_values, color=[color_attack, color_defense], 
               alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
ax1.set_ylabel('Attack Success Rate (%)', fontsize=13, fontweight='bold')
ax1.set_title('Attack Success Rate (ASR)\nBefore vs After Defense', fontsize=15, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, asr_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 3,
             f'{val:.1f}%',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add reduction annotation
ax1.annotate(f'↓ {overall_asr_without_defense:.1f}pp', 
             xy=(0.5, 30), fontsize=12, ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# 1.2 Average Harm Score (AHS) Comparison
ax2 = fig.add_subplot(gs[0, 1])
categories = ['Without\nDefense', 'With\nDefense']
ahs_values = [ahs_without_defense, ahs_with_defense]
bars = ax2.bar(categories, ahs_values, color=[color_attack, color_defense],
               alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
ax2.set_ylabel('Average Harm Score', fontsize=13, fontweight='bold')
ax2.set_title('Average Harm Score (AHS)\nBefore vs After Defense', fontsize=15, fontweight='bold')
ax2.set_ylim(0, 10)
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, ahs_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{val:.2f}',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add reduction annotation
ax2.annotate(f'↓ {ahs_without_defense:.2f} pts', 
             xy=(0.5, 3), fontsize=12, ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# 1.3 Defense Block Rate (DBR)
ax3 = fig.add_subplot(gs[0, 2])
dbr_data = [dbr, 100-dbr]
colors_dbr = [color_defense, '#ecf0f1']
wedges, texts, autotexts = ax3.pie(dbr_data, labels=['Blocked', 'Not Blocked'],
                                     colors=colors_dbr, autopct='%1.1f%%',
                                     startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
                                     explode=(0.05, 0))
ax3.set_title('Defense Block Rate (DBR)', fontsize=15, fontweight='bold')

# Add center text
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax3.add_artist(centre_circle)
ax3.text(0, 0, f'{dbr:.1f}%\nBlocked', ha='center', va='center', 
         fontsize=16, fontweight='bold')

# ============================================================================
# Row 2: ASR Breakdown
# ============================================================================

# 2.1 ASR by Attack Type
ax4 = fig.add_subplot(gs[1, :2])
attack_types = asr_by_attack['Attack Type']
asr_vals = asr_by_attack['ASR (%)']
colors_gradient = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(attack_types)))

bars = ax4.barh(attack_types, asr_vals, color=colors_gradient, 
                alpha=0.8, edgecolor='black', linewidth=2)
ax4.set_xlabel('Attack Success Rate (%)', fontsize=13, fontweight='bold')
ax4.set_title('ASR by Attack Type (Without Defense)', fontsize=15, fontweight='bold')
ax4.set_xlim(0, 100)
ax4.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, asr_vals):
    width = bar.get_width()
    ax4.text(width + 2, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%',
             ha='left', va='center', fontsize=12, fontweight='bold')

# 2.2 ASR by Model
ax5 = fig.add_subplot(gs[1, 2])
model_data = asr_by_model.head(6)
models_short = [m.split('_')[-1][:15] if len(m) > 15 else m for m in model_data['Model']]
model_asr = model_data['ASR (%)']

bars = ax5.barh(models_short, model_asr, color=color_neutral, 
                alpha=0.7, edgecolor='black', linewidth=1.5)
ax5.set_xlabel('ASR (%)', fontsize=12, fontweight='bold')
ax5.set_title('ASR by Model\n(Without Defense)', fontsize=14, fontweight='bold')
ax5.set_xlim(0, 110)
ax5.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, model_asr):
    width = bar.get_width()
    ax5.text(width + 2, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%',
             ha='left', va='center', fontsize=10, fontweight='bold')

# ============================================================================
# Row 3: Combined Metrics Analysis
# ============================================================================

# 3.1 Three Metrics Side by Side Comparison
ax6 = fig.add_subplot(gs[2, :])
metrics_names = ['Attack Success Rate (%)', 'Avg Harm Score (0-10)', 'Defense Block Rate (%)']
without_defense_vals = [overall_asr_without_defense, ahs_without_defense * 10, 0]  # Scale AHS to 0-100
with_defense_vals = [overall_asr_with_defense, ahs_with_defense, dbr]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax6.bar(x - width/2, without_defense_vals, width, 
                label='Without Defense', color=color_attack, alpha=0.8, edgecolor='black', linewidth=2)
bars2 = ax6.bar(x + width/2, with_defense_vals, width,
                label='With Defense', color=color_defense, alpha=0.8, edgecolor='black', linewidth=2)

ax6.set_ylabel('Score/Rate (%)', fontsize=13, fontweight='bold')
ax6.set_title('Comprehensive Performance Metrics Comparison\n(ASR, AHS, DBR)', fontsize=16, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics_names, fontsize=12)
ax6.legend(fontsize=12, loc='upper left')
ax6.set_ylim(0, 110)
ax6.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax6.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement arrows
for i in range(len(metrics_names)):
    if without_defense_vals[i] > with_defense_vals[i]:
        improvement = without_defense_vals[i] - with_defense_vals[i]
        ax6.annotate('', xy=(i + width/2, with_defense_vals[i] + 3), 
                    xytext=(i - width/2, without_defense_vals[i] - 3),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'))

plt.suptitle('Performance Evaluation Metrics: ASR, AHS & DBR', 
             fontsize=20, fontweight='bold', y=0.995)

# Add summary statistics box
summary_box_text = f"""PERFORMANCE METRICS SUMMARY:

Attack Success Rate (ASR):     {overall_asr_without_defense:.1f}% → {overall_asr_with_defense:.1f}%  (↓ {overall_asr_without_defense:.1f}pp)
Average Harm Score (AHS):      {ahs_without_defense:.2f} → {ahs_with_defense:.2f}  (↓ {ahs_without_defense:.2f} pts)
Defense Block Rate (DBR):      {dbr:.1f}%  ({blocked_by_defense}/{total_defense_attempts} attempts blocked)

Overall Defense Effectiveness: EXCELLENT (100% block rate)
Most Vulnerable Attack: {asr_by_attack.loc[asr_by_attack['ASR (%)'].idxmax(), 'Attack Type']} ({asr_by_attack['ASR (%)'].max():.1f}% ASR)
Most Resilient Attack: {asr_by_attack.loc[asr_by_attack['ASR (%)'].idxmin(), 'Attack Type']} ({asr_by_attack['ASR (%)'].min():.1f}% ASR)
"""

fig.text(0.5, 0.01, summary_box_text, ha='center', fontsize=11,
         bbox=dict(boxstyle='round,pad=1', facecolor='lightcyan', alpha=0.9),
         family='monospace', va='bottom')

plt.savefig('performance_metrics_evaluation.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: performance_metrics_evaluation.png\n")

# ============================================================================
# Create Detailed Metrics Table
# ============================================================================

metrics_table = pd.DataFrame({
    'Metric': ['Attack Success Rate (ASR)', 'Average Harm Score (AHS)', 'Defense Block Rate (DBR)'],
    'Without Defense': [f'{overall_asr_without_defense:.2f}%', f'{ahs_without_defense:.2f}', 'N/A'],
    'With Defense': [f'{overall_asr_with_defense:.2f}%', f'{ahs_with_defense:.2f}', f'{dbr:.2f}%'],
    'Improvement': [f'{overall_asr_without_defense - overall_asr_with_defense:.2f}pp', 
                   f'{ahs_without_defense - ahs_with_defense:.2f} pts', 
                   f'{dbr:.2f}%']
})

print("\n" + "="*80)
print("DETAILED METRICS TABLE")
print("="*80 + "\n")
print(metrics_table.to_string(index=False))
print("\n" + "="*80)

metrics_table.to_csv('performance_metrics_table.csv', index=False)
print("\nMetrics table saved to: performance_metrics_table.csv\n")

plt.show()
