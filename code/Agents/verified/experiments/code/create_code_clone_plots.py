import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# Load the enhanced issue-to-difficulty JSON file
input_file = "enhanced_issue_to_difficulty.json"

with open(input_file, "r") as file:
    data = json.load(file)

# Dictionary to store clone type counts per difficulty category
difficulty_clone_counts = defaultdict(Counter)

# Standardized colors for clone types (ensuring consistency)
clone_colors = {
    "type-1": "skyblue",
    "type-2": "lightcoral",
    "type-3": "gold",
    "type-4": "lightgreen",
    "Not Clone": "gray"
}

# Set of all possible clone types (to ensure consistency in bar charts)
all_clone_types = set(clone_colors.keys())

# Process each issue
for issue, details in data.items():
    difficulty = details.get("difficulty", "Unknown")
    
    for agent, classification in details.get("agents", {}).items():
        # Replace "N/A" with "Not clone"
        classification = classification.replace("N/A", "Not Clone")

        if classification != "Unknown" and classification != "Error in response":
            difficulty_clone_counts[difficulty][classification] += 1

# Create "plots" folder if it doesn't exist
plot_folder = "plots"
os.makedirs(plot_folder, exist_ok=True)

# Generate bar and pie charts for each difficulty category
for difficulty, clone_counts in difficulty_clone_counts.items():
    if not clone_counts:
        continue  # Skip empty categories

    # Ensure all clone types appear in the bar chart (even if count is 0)
    for clone_type in all_clone_types:
        if clone_type not in clone_counts:
            clone_counts[clone_type] = 0

    # Prepare data for bar and pie plots
    clone_types = sorted(clone_counts.keys(), key=lambda x: list(clone_colors.keys()).index(x))  # Sort by predefined order
    counts = [clone_counts[clone_type] for clone_type in clone_types]
    colors = [clone_colors[clone_type] for clone_type in clone_types]  # Assign consistent colors
    total_patches = sum(counts)  # Total patches considered

    # Bar Chart
    plt.figure(figsize=(8, 5))
    plt.bar(clone_types, counts, color=colors, edgecolor='black')
    plt.xlabel("Clone Type")
    plt.ylabel("Count")
    plt.title(f"Clone Types Distribution for {difficulty}\n(Total patches: {total_patches})")
    plt.xticks(rotation=45, ha="right")

    # Save bar plot
    bar_plot_filename = f"{plot_folder}/{difficulty.replace(' ', '_')}_bar.png"
    plt.savefig(bar_plot_filename, bbox_inches="tight")
    plt.close()  # Close the figure to prevent overlapping

    print(f"Saved bar plot: {bar_plot_filename}")

    # Filter out 0.0% entries from pie chart
    filtered_clone_types = []
    filtered_counts = []
    filtered_colors = []

    for i, count in enumerate(counts):
        percentage = (count / total_patches) * 100 if total_patches > 0 else 0
        if percentage > 0.0:
            filtered_clone_types.append(clone_types[i])
            filtered_counts.append(count)
            filtered_colors.append(colors[i])

    # Pie Chart
    plt.figure(figsize=(6, 6))
    plt.pie(filtered_counts, labels=filtered_clone_types, autopct="%1.1f%%", startangle=140, colors=filtered_colors)
    plt.title(f"Clone Type Percentage for {difficulty}\n(Total patches: {total_patches})")

    # Save pie chart
    pie_plot_filename = f"{plot_folder}/{difficulty.replace(' ', '_')}_pie.png"
    plt.savefig(pie_plot_filename, bbox_inches="tight")
    plt.close()  # Close the figure to prevent overlapping

    print(f"Saved pie chart: {pie_plot_filename}")

# Generate a total pie chart for all issues combined
total_clone_counts = Counter()
for counts in difficulty_clone_counts.values():
    total_clone_counts.update(counts)

if total_clone_counts:
    total_clone_types = sorted(total_clone_counts.keys(), key=lambda x: list(clone_colors.keys()).index(x))
    total_counts = [total_clone_counts[clone_type] for clone_type in total_clone_types]
    total_colors = [clone_colors[clone_type] for clone_type in total_clone_types]
    total_patches = sum(total_counts)  # Total patches across all difficulties

    # Filter out 0.0% entries from overall pie chart
    overall_filtered_clone_types = []
    overall_filtered_counts = []
    overall_filtered_colors = []

    for i, count in enumerate(total_counts):
        percentage = (count / total_patches) * 100 if total_patches > 0 else 0
        if percentage > 0.0:
            overall_filtered_clone_types.append(total_clone_types[i])
            overall_filtered_counts.append(count)
            overall_filtered_colors.append(total_colors[i])

    plt.figure(figsize=(7, 7))
    plt.pie(overall_filtered_counts, labels=overall_filtered_clone_types, autopct="%1.1f%%", startangle=140, colors=overall_filtered_colors)
    plt.title(f"Overall Clone Type Distribution\n(Total patches: {total_patches})")

    total_pie_filename = f"{plot_folder}/overall_clone_distribution_pie.png"
    plt.savefig(total_pie_filename, bbox_inches="tight")
    plt.close()

    print(f"Saved overall pie chart: {total_pie_filename}")

print("All plots saved in the 'plots' folder.")
