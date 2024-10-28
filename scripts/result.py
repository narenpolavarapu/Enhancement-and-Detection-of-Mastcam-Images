import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
datasets = ['First 50% of Data', 'Full Dataset']
models = ['KMeans', 'Agglomerative', 'GMM']

# Silhouette scores
scores = {
    'KMeans': [0.4055578112602234, 0.4068896770477295],
    'Agglomerative': [0.4011995196342468, 0.38432419300079346],
    'GMM': [0.35919857025146484, 0.37376266717910767]
}

# X-axis positions
x = np.arange(len(datasets))

# Bar width
width = 0.2

# Light colors for each model
colors = ['lightblue', 'lightpink', 'lightgreen']  # Using light blue, pink, and green

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars for each model
for i, model in enumerate(models):
    ax.bar(x + (i - 1) * width, scores[model], width, label=model, color=colors[i])

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Scores Comparison for Clustering Models')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(title='Models')

# Adding value labels on top of the bars
def add_value_labels(bars):
    """Add labels to the bars in a bar chart."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom')

# Adding labels for each bar
for model in models:
    add_value_labels(ax.patches)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
