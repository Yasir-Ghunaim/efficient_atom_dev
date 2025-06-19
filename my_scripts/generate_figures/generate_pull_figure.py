import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(['science', 'no-latex'])  # Apply scienceplot style

# Data from the table
labels = ['rMD17 (Aspirin)', 'MD22 (Ac-Ala3-NHMe)', 'SPICE (Solvated Amino Acids)', 'QM9 ($U_0$)']
jmp_s = [6.7, 2.64, 5.71, 3.3]  # JMP-S values
ani_1x = [5.4, 2.90, 5.13, 2.9]  # ANI-1x values

x = np.array([0, 1, 0, 1]) # Set x locations for each category
width = 0.1  # Bar width to control spacing

fig, axs = plt.subplots(1, 4, figsize=(10, 2))
fig.subplots_adjust(wspace=0.2)
axs = axs.flatten()

for i, ax in enumerate(axs):
    ax.bar(x[i] - 0.6*width, jmp_s[i], width, label='JMP-S', color='#fc8d59', alpha=0.7)
    ax.bar(x[i] + 0.6*width, ani_1x[i], width, label='ANI-1x', color='#91cf60', alpha=0.7)
    
    ax.set_title(labels[i])
    ax.set_xticks([])
    ax.set_yticks([])  # Remove y-axis ticks
    if i == 0:
        ax.set_ylabel('MAE, lower is better')
    else:
        ax.set_ylabel('')
    ax.set_ylim(0, max(jmp_s[i], ani_1x[i]) + 0.25*max(jmp_s[i], ani_1x[i]))  # Add limits to the y-axis


    for bar, value in zip(ax.patches, [jmp_s[i], ani_1x[i]]):
        ax.text(bar.get_x() + bar.get_width()/2, value - 0.26*value, f'{value:.1f}', ha='center', fontsize=16)
    
    # Calculate and add relative improvement if ANI-1x improves over JMP-S
    improvement = ((jmp_s[i] - ani_1x[i]) / jmp_s[i]) * 100
    y_pos = min(jmp_s[i], ani_1x[i])
    if improvement > 0:
        ax.text(bar.get_x() + bar.get_width()/2, y_pos + 0.07*y_pos,
                f'+{int(improvement)}%',
                ha='center', fontsize=16, color='green')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, y_pos + 0.16*y_pos,
                f'{int(improvement)}%',
                ha='center', fontsize=16, color='#C70039')


fig.text(0.5, -0.2, '% indicates relative MAE improvement over SoTA', 
         ha='center', fontsize=12)

# fig.suptitle('Comparison of Pretraining Performance on Downstream Tasks', fontsize=14)
fig.legend(['SoTA (JMP-S)', 'Ours: 24x less Resources'], loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=2, fontsize=14)
plt.tight_layout(rect=[0, 0.04, 1, 1], w_pad=2.0)  # Adjust rect to make space for the legend
plt.savefig("pull_figure.png", dpi=300)
