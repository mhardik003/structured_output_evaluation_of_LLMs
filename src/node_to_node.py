from evaluate import load
import json
from tqdm import tqdm

# Load BERTScore
bertscore = load("bertscore")


class TreeNode:
    def __init__(self, content, *children: "TreeNode"):
        self.content = content
        self.children = children


def to_tree(d) -> TreeNode:
    """Convert a dictionary/list into a tree structure."""
    if isinstance(d, dict):
        return TreeNode("{}", *[TreeNode(k, to_tree(v)) for k, v in d.items()])
    elif isinstance(d, list):
        return TreeNode("[]", *[to_tree(v) for v in d])
    else:
        return TreeNode(d)


def node_similarity(pred_node: TreeNode, target_node: TreeNode):
    """Evaluate the similarity between individual nodes using BERTScore."""
    return bertscore.compute(predictions=[str(pred_node.content)], references=[str(target_node.content)], lang='en', device="cuda:0")


def collect_all_nodes(tree: TreeNode):
    """Collect all nodes from the tree."""
    nodes = [tree]
    for child in tree.children:
        nodes.extend(collect_all_nodes(child))
    return nodes


def evaluate(prediction: TreeNode, target: TreeNode):
    """Evaluate the similarity between prediction and target trees using all node-to-node BERTScore."""
    
    # Collect all nodes from both prediction and target trees
    pred_nodes = collect_all_nodes(prediction)
    target_nodes = collect_all_nodes(target)
    
    similarities = []

    # Compare every node in the prediction tree with every node in the reference tree
    for pred_node in tqdm(pred_nodes):
        for target_node in target_nodes:
            similarity = node_similarity(pred_node, target_node)
            similarities.append(similarity["f1"][0] if "f1" in similarity else 0)

    # Calculate metrics
    max_similarity = max(similarities)
    min_similarity = min(similarities)
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0

    return {"max_similarity": max_similarity, "min_similarity": min_similarity, "avg_similarity": avg_similarity}


# Load the JSON data
with open('./cleaned.json') as f:
    test_jsons = json.load(f)

scores_phi = []
scores_qwen = []

for test_json in tqdm(test_jsons):
    input_wiki = test_json['input']
    reference = test_json['reference']
    reference_tree = to_tree(reference)

    phi = test_json['phi']
    phi_tree = to_tree(phi)

    qwen = test_json['qwen']
    qwen_tree = to_tree(qwen)

    # Evaluate the phi and qwen predictions against the reference
    score_phi = evaluate(phi_tree, reference_tree)
    score_qwen = evaluate(qwen_tree, reference_tree)

    print(score_phi)
    print(score_qwen)

    scores_phi.append(score_phi)
    scores_qwen.append(score_qwen)

    # Save the results into JSON files
    with open('node_to_node_bert_scores_phi.json', 'w') as phi_file:
        json.dump(scores_phi, phi_file)

    with open('node_to_node_bert_scores_qwen.json', 'w') as qwen_file:
        json.dump(scores_qwen, qwen_file)
