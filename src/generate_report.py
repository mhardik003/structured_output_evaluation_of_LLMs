import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from rouge import Rouge
from evaluate import load
from scipy.stats import pearsonr
import sys
import os

# Increase recursion limit for deep trees
sys.setrecursionlimit(3000)

# Load BERTScore metric (adjust device if needed)
bertscore = load("bertscore")

# ------------------ Tree & Traversal Functions ------------------


class TreeNode:
    def __init__(self, content, *children: "TreeNode"):
        self.content = content
        self.children = children


def to_tree(d) -> TreeNode:
    if isinstance(d, dict):
        return TreeNode("{}", *[TreeNode(k, to_tree(v)) for k, v in d.items()])
    elif isinstance(d, list):
        return TreeNode("[]", *[to_tree(v) for v in d])
    else:
        return TreeNode(d)


def tree_to_dict(node: TreeNode):
    if node.content == "{}":
        return {child.content: tree_to_dict(child.children[0]) for child in node.children}
    elif node.content == "[]":
        return [tree_to_dict(child) for child in node.children]
    else:
        return node.content


def pre_order_traversal(node: TreeNode):
    result = [str(node.content)]
    for child in node.children:
        result.append(pre_order_traversal(child))
    return " ".join(result)


def post_order_traversal(node: TreeNode):
    result = []
    for child in node.children:
        result.append(post_order_traversal(child))
    result.append(str(node.content))
    return " ".join(result)


def in_order_traversal(node: TreeNode):
    if len(node.children) == 0:
        return str(node.content)
    elif len(node.children) == 1:
        return f"{in_order_traversal(node.children[0])} {node.content}"
    elif len(node.children) == 2:
        return f"{in_order_traversal(node.children[0])} {node.content} {in_order_traversal(node.children[1])}"
    else:
        # For non-binary trees, simply concatenate all children traversals
        return str(node.content) + " " + " ".join(in_order_traversal(child) for child in node.children)

# ------------------ Metric Evaluation Function ------------------


def evaluate(prediction: TreeNode, target: TreeNode):
    """
    Evaluate the similarity between prediction and target trees.
    Computes metrics from different traversals and returns a dictionary.
    Easily extensible: add more metrics below.
    """
    # Compute different traversals
    pred_pre = pre_order_traversal(prediction)
    target_pre = pre_order_traversal(target)
    pred_post = post_order_traversal(prediction)
    target_post = post_order_traversal(target)
    pred_in = in_order_traversal(prediction)
    target_in = in_order_traversal(target)
    pred_json = json.dumps(tree_to_dict(prediction))
    target_json = json.dumps(tree_to_dict(target))

    # --- BERTScore Metrics ---
    bert_scores = {
        "pre_order": bertscore.compute(predictions=[pred_pre], references=[target_pre], lang='en', device="cuda:0"),
        "post_order": bertscore.compute(predictions=[pred_post], references=[target_post], lang='en', device="cuda:0"),
        "in_order": bertscore.compute(predictions=[pred_in], references=[target_in], lang='en', device="cuda:0"),
        "json": bertscore.compute(predictions=[pred_json], references=[target_json], lang='en', device="cuda:0")
    }

    # --- ROUGE Metrics ---
    rouge_evaluator = Rouge()
    rouge_scores = {
        "pre_order": rouge_evaluator.get_scores(pred_pre, target_pre, avg=True),
        "post_order": rouge_evaluator.get_scores(pred_post, target_post, avg=True),
        "in_order": rouge_evaluator.get_scores(pred_in, target_in, avg=True),
        "json": rouge_evaluator.get_scores(pred_json, target_json, avg=True)
    }

    # --- Combine Metrics ---
    metrics = {}
    # Add BERTScore F1 scores for each traversal (extract scalar from list)
    for traversal in ["pre_order", "post_order", "in_order", "json"]:
        metrics[f"{traversal}_bertscore_f"] = bert_scores[traversal]["f1"][0]

    # Add ROUGE F1 scores for rouge-1, rouge-2, and rouge-l for each traversal
    for traversal in ["pre_order", "post_order", "in_order", "json"]:
        scores = rouge_scores[traversal]
        metrics[f"{traversal}_rouge_1_f"] = scores["rouge-1"]["f"]
        metrics[f"{traversal}_rouge_2_f"] = scores["rouge-2"]["f"]
        metrics[f"{traversal}_rouge_l_f"] = scores["rouge-l"]["f"]

    # Additional metrics can be added here; for example:
    # metrics["custom_metric"] = custom_metric_function(prediction, target)

    return metrics

# ------------------ Helper Functions ------------------


def compute_human_score(evaluation: dict) -> float:
    """
    Computes the human score as the average of four criteria:
    semantic_equivalence, structural_hierarchy, completeness, data_consistency.
    The score values are expected to be in evaluation["criteria_scores"] as strings.
    Applies min-max scaling on each criterion ((score - 1)/4).
    """
    criteria = ["semantic_equivalence", "structural_hierarchy",
                "completeness", "data_consistency"]
    scores = []
    for crit in criteria:
        try:
            score = (
                float(evaluation["criteria_scores"][crit]["score"]) - 1) / 4
            scores.append(score)
        except Exception:
            scores.append(np.nan)
    return np.nanmean(scores)


def min_max_scale(series: pd.Series) -> pd.Series:
    """Scales a pandas Series to the [0,1] range."""
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return series  # Avoid division by zero
    return (series - min_val) / (max_val - min_val)


def generate_statistics_report(df: pd.DataFrame, title: str = "Report") -> str:
    """
    Generates a Markdown formatted report with descriptive statistics,
    correlation with human score, and overall correlation matrix for combined data.
    """
    lines = []
    lines.append(f"# {title}\n")
    lines.append("## Descriptive Statistics\n")
    lines.append(df.describe().to_markdown())

    lines.append("\n## Correlation with Human Score (Pearson)\n")
    correlations = {}
    for col in df.columns:
        if col != "human_score":
            # Drop rows where either human_score or the metric is NaN or inf
            valid = df[['human_score', col]].replace(
                [np.inf, -np.inf], np.nan).dropna()
            if valid.shape[0] > 1:
                corr, _ = pearsonr(valid["human_score"], valid[col])
                correlations[col] = corr
            else:
                correlations[col] = np.nan
    corr_series = pd.Series(
        correlations, name="Pearson Correlation with Human Score")
    lines.append(corr_series.to_markdown())

    return "\n".join(lines)


def plot_scatter(df: pd.DataFrame, model_name: str):
    """
    For each metric (except human_score) in the DataFrame, creates a scatter plot
    of the metric versus human_score, and saves the plot in the current directory.
    """
    metrics = [col for col in df.columns if col != "human_score"]
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[metric], y=df["human_score"])
        plt.xlabel(metric)
        plt.ylabel("Human Score")
        plt.title(f"{model_name}: Human Score vs {metric}")
        filename = f"{model_name}_scatter_{metric}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def plot_overall_scatter(df: pd.DataFrame):
    """
    Creates scatter plots for each metric (except human_score) in the combined DataFrame
    versus human_score, and saves the plots in the current directory.
    """
    metrics = [col for col in df.columns if col != "human_score"]
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[metric], y=df["human_score"])
        plt.xlabel(metric)
        plt.ylabel("Human Score")
        plt.title(f"Combined: Human Score vs {metric}")
        filename = f"Combined_scatter_{metric}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

# ------------------ Main Evaluation & Report Generation ------------------


def main():
    # Read from the JSONL file (each line is a JSON object)
    data_phi = []
    data_qwen = []

    with open('ai_evaluation.jsonl', 'r') as f:
        for line in tqdm(f, desc="Processing lines"):
            if not line.strip():
                continue  # Skip empty lines
            test_json = json.loads(line)

            # Build trees for the reference and predictions
            reference_tree = to_tree(test_json['reference'])

            # --- For phi prediction ---
            phi_tree = to_tree(test_json['phi'])
            metrics_phi = evaluate(phi_tree, reference_tree)
            human_phi = compute_human_score(test_json['phi_evaluation'])
            sample_phi = {"human_score": human_phi}
            sample_phi.update(metrics_phi)
            data_phi.append(sample_phi)

            # --- For qwen prediction ---
            qwen_tree = to_tree(test_json['qwen'])
            metrics_qwen = evaluate(qwen_tree, reference_tree)
            human_qwen = compute_human_score(test_json['qwen_evaluation'])
            sample_qwen = {"human_score": human_qwen}
            sample_qwen.update(metrics_qwen)
            data_qwen.append(sample_qwen)

    # Convert results into DataFrames for easier statistics and correlation analysis
    df_phi = pd.DataFrame(data_phi)
    df_qwen = pd.DataFrame(data_qwen)

    # Generate Markdown reports for raw scores
    report_phi = generate_statistics_report(
        df_phi, title="Phi Evaluation Report (Raw Scores)")
    report_qwen = generate_statistics_report(
        df_qwen, title="Qwen Evaluation Report (Raw Scores)")

    # Generate overall combined report
    combined_df = pd.concat([df_phi, df_qwen], axis=0)
    report_combined = generate_statistics_report(
        combined_df, title="Combined Evaluation Report (Raw Scores)")

    # Append overall correlation matrix to the markdown
    overall_corr = combined_df.corr()
    overall_corr_md = overall_corr.to_markdown()
    report_combined += "\n\n## Overall Correlation Matrix\n" + overall_corr_md

    # Save the combined markdown report
    final_report = "\n\n".join([report_phi, report_qwen, report_combined])
    with open("evaluation_report.md", "w") as f:
        f.write(final_report)

    # Print reports to console (optional)
    print(report_phi)
    print("\n" + "=" * 80 + "\n")
    print(report_qwen)
    print("\n" + "=" * 80 + "\n")
    print(report_combined)

    # Apply min-max scaling on each column (scaling each metric individually)
    df_phi_scaled = df_phi.copy()
    df_qwen_scaled = df_qwen.copy()
    for col in df_phi.columns:
        df_phi_scaled[col] = min_max_scale(df_phi[col])
    for col in df_qwen.columns:
        df_qwen_scaled[col] = min_max_scale(df_qwen[col])

    # Generate and save scaled reports as well (if needed)
    report_phi_scaled = generate_statistics_report(
        df_phi_scaled, title="Phi Evaluation Report (Min-Max Scaled)")
    report_qwen_scaled = generate_statistics_report(
        df_qwen_scaled, title="Qwen Evaluation Report (Min-Max Scaled)")
    combined_df_scaled = pd.concat([df_phi_scaled, df_qwen_scaled], axis=0)
    report_combined_scaled = generate_statistics_report(
        combined_df_scaled, title="Combined Evaluation Report (Min-Max Scaled)")
    overall_corr_scaled = combined_df_scaled.corr().to_markdown()
    report_combined_scaled += "\n\n## Overall Correlation Matrix\n" + overall_corr_scaled

    with open("evaluation_report_scaled.md", "w") as f:
        f.write("\n\n".join(
            [report_phi_scaled, report_qwen_scaled, report_combined_scaled]))

    # Create scatter plots for each model and for the combined dataset, and save in current directory
    plot_scatter(df_phi, "Phi")
    plot_scatter(df_qwen, "Qwen")
    plot_overall_scatter(combined_df)


if __name__ == "__main__":
    main()
