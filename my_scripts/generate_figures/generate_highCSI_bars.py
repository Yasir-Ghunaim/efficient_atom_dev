import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(['science', 'no-latex'])  # Apply scienceplot style
# Data from the table
units = ['$(meV/\AA)$', '$(meV/\AA)$', '$(meV/\AA)$', '$(meV)$']  # Units for each task
categories = ['rMD17', 'MD22', 'SPICE', 'QM9']
ani1x_2m = [5.4, 2.90, 5.13, 2.9]
ani1x_2m_oc22_1m = [7.8, 3.21, 6.00, 3.1]
# Define colors
color_ani1x = '#B1DD8F'  # Soft green
color_ani1x_oc22 = '#FCAF8A'  # Soft orange

# Bar width and positions
bar_width = 0.35
x = np.arange(len(categories))

# Create the bar plot
fig, ax = plt.subplots(figsize=(8, 2.2))

# Plot bars with a solid pastel blue and a dashed deeper blue
ax.bar(x - bar_width/2, ani1x_2m, bar_width, label='ANI-1x (2M)', color=color_ani1x, edgecolor='black', linewidth=0.5)
ax.bar(x + bar_width/2, ani1x_2m_oc22_1m, bar_width, label='ANI-1x (2M) + OC22 (1M)', 
       color=color_ani1x_oc22, edgecolor='black', linewidth=0.5)

# Labels and title
# ax.set_xlabel('Evaluation Task', fontsize=12)
ax.set_ylabel('MAE', fontsize=14)
ax.set_ylim(0, 9)

# ax.set_title('Effect of Adding OC22 (1M) on Performance', fontsize=14)
ax.set_xticks(x)
# ax.set_xticklabels(categories, fontsize=11)
ax.set_xticklabels([f"{cat}\n{unit}" for cat, unit in zip(categories, units)], fontsize=14)  # Adding units below labels

#increase y tick font size to 13
plt.yticks(fontsize=14)


# Remove minor ticks and top ticks
plt.minorticks_off()
plt.tick_params(axis='x', which='both', top=False)
# plt.tick_params(axis='y', which='both', right=False)

# ax.legend(fontsize=12.5, loc='upper right', bbox_to_anchor=(1, 1.04))
ax.legend(
    fontsize=12.5,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.27),  # adjust vertical position
    ncol=2,
    frameon=True
)

# Save and show the figure
plt.savefig("highCSI.png", dpi=300, bbox_inches='tight')
plt.show()
