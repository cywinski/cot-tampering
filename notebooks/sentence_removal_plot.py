# %%
import json
import os

import matplotlib.pyplot as plt

# %%
results_dir = "/Users/bcywinski/work/code/cot-tampering/experiments/results/deepseek-r1/math500v2/random_sentence_removal_closed"

# Get all subdirectories (percentage values)
percentage_dirs = sorted(
    [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))],
    key=float,
)

print(f"Found percentage directories: {percentage_dirs}")

# %%
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

# %%
# Prepare data for plotting
percentages_sorted = sorted(percentage_data.keys(), key=float)
percentages_sorted_scaled = [float(p) * 100 for p in percentages_sorted]
accuracies = [percentage_data[p]["accuracy"] for p in percentages_sorted]
tamper_rates = [percentage_data[p]["tamper_detection_rate"] for p in percentages_sorted]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(
    percentages_sorted_scaled,
    accuracies,
    marker="o",
    markersize=12,
    linewidth=3,
    label="Accuracy",
    color="#1f77b4",
)
plt.plot(
    percentages_sorted_scaled,
    tamper_rates,
    marker="s",
    linewidth=3,
    markersize=12,
    label="Tamper Detection Rate",
    color="#ff7f0e",
)

plt.xlabel("Percentage of Sentences Removed (%)", fontsize=18)
plt.title("Random Sentences Removal: DeepSeek R1 (MATH-500)", fontsize=22)
plt.legend(fontsize=18, loc="upper left")
plt.grid(True, alpha=0.2)
plt.yticks(fontsize=16)
plt.xticks(percentages_sorted_scaled, fontsize=16)
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

# %%
