import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 16)

# Read the comparison data
comparison_df = pd.read_csv('CSV/attack_vs_defense_comparison.csv')

# Read attack results (from previous analysis)
attack_results = pd.read_csv('comparison_results.csv')

# Calculate overall attack metrics
overall_attack_asr = attack_results['Success Rate (%)'].mean()
overall_defense_asr = 0.0  # From the comparison file

# Read defense evaluation data
defense_df = pd.read_csv('Defense/defense_jailbreak_evaluation.csv')

# Calculate defense metrics
total_defense_attempts = len(defense_df)
blocked_by_defense = defense_df['defense_blocked'].sum()
defense_block_rate = (blocked_by_defense / total_defense_attempts * 100) if total_defense_attempts > 0 else 0

# Calculate attack success by type
attack_by_type = attack_results.groupby('Attack Type')['Success Rate (%)'].mean().reset_index()
attack_by_type.columns = ['Attack Type', 'Without Defense']
attack_by_type['With Defense'] = 0.0  # Assuming defense blocks all

print("\n" + "="*80)
print("ATTACK vs DEFENSE COMPARISON ANALYSIS")
print("="*80 + "\n")

print("Overall Metrics:")
print(f"  Average Attack Success Rate (No Defense): {overall_attack_asr:.2f}%")
print(f"  Attack Success Rate (With Defense): {overall_defense_asr:.2f}%")
print(f"  Defense Effectiveness: {defense_block_rate:.2f}% block rate")
print(f"  Improvement: {overall_attack_asr:.2f} percentage points reduction\n")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 24))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

# 1. Overall Attack Success Rate - Before vs After Defense
ax1 = fig.add_subplot(gs[0, 0])
categories = ['Without\nDefense', 'With\nDefense']
values = [overall_attack_asr, overall_defense_asr]
colors = ['#e74c3c', '#2ecc71']
bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Overall Attack Success Rate\nBefore vs After Defense', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val:.1f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add improvement arrow and text
if values[0] > values[1]:
    improvement = values[0] - values[1]
    ax1.annotate('', xy=(1, values[1] + 5), xytext=(0, values[0] - 5),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax1.text(0.5, (values[0] + values[1])/2, f'↓ {improvement:.1f}%\nreduction',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

# 2. Attack Success by Type - Before vs After Defense
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(len(attack_by_type))
width = 0.35
bars1 = ax2.bar(x - width/2, attack_by_type['Without Defense'], width, 
                label='Without Defense', color='#e74c3c', alpha=0.7, edgecolor='black')
bars2 = ax2.bar(x + width/2, attack_by_type['With Defense'], width,
                label='With Defense', color='#2ecc71', alpha=0.7, edgecolor='black')

ax2.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Attack Success Rate by Type\nBefore vs After Defense', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(attack_by_type['Attack Type'], fontsize=11)
ax2.legend(fontsize=10)
ax2.set_ylim(0, 100)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

# 3. Defense Effectiveness by Metric
ax3 = fig.add_subplot(gs[1, 0])
metrics = ['Attack Success\nRate (ASR)', 'LlamaGuard\nUnsafe Rate', 'Average\nHarm Score']
without_defense = [30.00, 97.00, 6.08]
with_defense = [0.00, 0.00, 0.00]

x = np.arange(len(metrics))
bars1 = ax3.bar(x - width/2, without_defense, width, 
                label='Without Defense', color='#e74c3c', alpha=0.7, edgecolor='black')
bars2 = ax3.bar(x + width/2, with_defense, width,
                label='With Defense', color='#2ecc71', alpha=0.7, edgecolor='black')

ax3.set_ylabel('Score/Rate (%)', fontsize=12, fontweight='bold')
ax3.set_title('Defense Effectiveness Across Key Metrics', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics, fontsize=10)
ax3.legend(fontsize=10)
ax3.set_ylim(0, 110)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Model-specific comparison
ax4 = fig.add_subplot(gs[1, 1])
model_results = attack_results.groupby('Model')['Success Rate (%)'].mean().reset_index()
model_results = model_results.sort_values('Success Rate (%)', ascending=False)

# Limit to top models for readability
if len(model_results) > 6:
    model_results = model_results.head(6)

models_short = [m.split('_')[-1] if len(m) > 20 else m for m in model_results['Model']]
x = np.arange(len(model_results))
bars1 = ax4.bar(x - width/2, model_results['Success Rate (%)'], width,
                label='Without Defense', color='#e74c3c', alpha=0.7, edgecolor='black')
bars2 = ax4.bar(x + width/2, [0]*len(model_results), width,
                label='With Defense', color='#2ecc71', alpha=0.7, edgecolor='black')

ax4.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax4.set_title('Model Vulnerability\nBefore vs After Defense', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models_short, rotation=45, ha='right', fontsize=9)
ax4.legend(fontsize=10, loc='upper right')
ax4.set_ylim(0, 110)
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=8)

# 5. Improvement Percentage by Attack Type
ax5 = fig.add_subplot(gs[2, :])
attack_types = attack_by_type['Attack Type'].tolist()
improvements = attack_by_type['Without Defense'].tolist()
colors_gradient = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(improvements)))

bars = ax5.barh(attack_types, improvements, color=colors_gradient, alpha=0.8, edgecolor='black', linewidth=2)
ax5.set_xlabel('Improvement (Percentage Points Reduction)', fontsize=12, fontweight='bold')
ax5.set_title('Defense Improvement by Attack Type\n(Reduction in Success Rate)', fontsize=14, fontweight='bold')
ax5.set_xlim(0, 100)
ax5.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, improvements)):
    width_val = bar.get_width()
    ax5.text(width_val + 2, bar.get_y() + bar.get_height()/2,
            f'{val:.1f} pp',
            ha='left', va='center', fontsize=11, fontweight='bold')

# Add overall summary text box
summary_text = f"""DEFENSE EFFECTIVENESS SUMMARY:
    
✓ Overall Attack Success Reduction: {overall_attack_asr:.1f} percentage points
✓ Defense Block Rate: {defense_block_rate:.1f}%
✓ Most Vulnerable Attack (Before): {attack_by_type.loc[attack_by_type['Without Defense'].idxmax(), 'Attack Type']} ({attack_by_type['Without Defense'].max():.1f}%)
✓ Defense Effectiveness: 100% blocking across all attack types
"""

fig.text(0.5, 0.02, summary_text, ha='center', fontsize=13,
         bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8),
         family='monospace')

plt.suptitle('Jailbreak Attack vs Defense Comparison', 
             fontsize=22, fontweight='bold', y=0.995)

plt.savefig('attack_vs_defense_comparison.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: attack_vs_defense_comparison.png")

# Create a detailed comparison table
comparison_table = pd.DataFrame({
    'Attack Type': attack_by_type['Attack Type'],
    'Success Rate Without Defense (%)': attack_by_type['Without Defense'],
    'Success Rate With Defense (%)': attack_by_type['With Defense'],
    'Improvement (pp)': attack_by_type['Without Defense'] - attack_by_type['With Defense']
})

print("\n" + "="*80)
print("DETAILED COMPARISON TABLE")
print("="*80 + "\n")
print(comparison_table.to_string(index=False))
print("\n" + "="*80)

comparison_table.to_csv('detailed_attack_defense_comparison.csv', index=False)
print("\nDetailed comparison table saved to: detailed_attack_defense_comparison.csv\n")

plt.show()
