# %%
import json
import os

import matplotlib.pyplot as plt

# %%
# Define model directories
base_dir = "/Users/bcywinski/work/code/cot-tampering/experiments/results"
models = {
    "deepseek-r1": os.path.join(
        base_dir, "deepseek-r1/math500v2/random_sentence_removal_closed"
    ),
    "qwen3-235b-a22b-thinking-2507": os.path.join(
        base_dir,
        "qwen3-235b-a22b-thinking-2507/math500v2/random_sentence_removal_closed",
    ),
}


# %%
def process_model_data(results_dir, model_name):
    """Process assessment data for a single model."""
    print(f"\nProcessing {model_name}...")

    # Get all subdirectories (percentage values)
    percentage_dirs = sorted(
        [
            d
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        ],
        key=float,
    )

    print(f"Found percentage directories: {percentage_dirs}")

    percentage_data = {}

    for percentage_dir in percentage_dirs:
        assessments_file = os.path.join(
            results_dir, percentage_dir, "metrics", "assessments.json"
        )

        if not os.path.exists(assessments_file):
            print(f"Warning: {assessments_file} not found, skipping")
            continue

        with open(assessments_file, "r") as f:
            assessments_data = json.load(f)

        # Calculate metrics
        total_assessments = 0
        yes_count = 0
        total_modifieds = 0
        modified_yes_count = 0

        for problem_idx, problem_data in assessments_data.items():
            assessments = problem_data.get("assessments", [])
            modifieds = problem_data.get("modifieds", [])

            total_assessments += len(assessments)
            yes_count += sum(1 for a in assessments if a == "yes")

            total_modifieds += len(modifieds)
            modified_yes_count += sum(1 for m in modifieds if m and m.lower() == "yes")

        accuracy = yes_count / total_assessments if total_assessments > 0 else 0
        tamper_detection_rate = (
            modified_yes_count / total_modifieds if total_modifieds > 0 else 0
        )

        percentage_data[percentage_dir] = {
            "accuracy": accuracy,
            "tamper_detection_rate": tamper_detection_rate,
            "total_assessments": total_assessments,
            "total_modifieds": total_modifieds,
        }

        print(
            f"{percentage_dir}: accuracy={accuracy:.3f}, tamper_detection={tamper_detection_rate:.3f}"
        )

    return percentage_data


# Process data for all models
all_model_data = {}
for model_name, results_dir in models.items():
    all_model_data[model_name] = process_model_data(results_dir, model_name)

# %%
# Create the plot
plt.figure(figsize=(14, 8))

# Color scheme: one color per model, different styles for metrics
colors = {
    "deepseek-r1": "#1f77b4",  # Blue
    "qwen3-235b-a22b-thinking-2507": "#2ca02c",  # Green
}

# Styles: solid line for accuracy, dashed for tamper detection
linestyles = {
    "accuracy": "-",
    "tamper": "--",
}

markers = {
    "accuracy": "o",
    "tamper": "s",
}

# Plot data for each model
for model_name, percentage_data in all_model_data.items():
    percentages_sorted = sorted(percentage_data.keys(), key=float)
    percentages_sorted_scaled = [float(p) * 100 for p in percentages_sorted]
    accuracies = [percentage_data[p]["accuracy"] for p in percentages_sorted]
    tamper_rates = [
        percentage_data[p]["tamper_detection_rate"] for p in percentages_sorted
    ]

    # Short model name for labels
    short_name = "DeepSeek R1" if model_name == "deepseek-r1" else "Qwen3-235B"

    # Plot accuracy
    plt.plot(
        percentages_sorted_scaled,
        accuracies,
        marker=markers["accuracy"],
        markersize=10,
        linewidth=2.5,
        label=f"{short_name} - Accuracy",
        color=colors[model_name],
        linestyle=linestyles["accuracy"],
    )

    # Plot tamper detection rate
    plt.plot(
        percentages_sorted_scaled,
        tamper_rates,
        marker=markers["tamper"],
        linewidth=2.5,
        markersize=10,
        label=f"{short_name} - Tamper Detection",
        color=colors[model_name],
        linestyle=linestyles["tamper"],
    )

plt.xlabel("Percentage of Sentences Removed (%)", fontsize=18)
plt.ylabel("Rate", fontsize=18)
plt.title("Random Sentences Removal: Comparison (MATH-500)", fontsize=22)
plt.legend(fontsize=14, loc="best")
plt.grid(True, alpha=0.2)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

# %%
