from rouge import Rouge
import json
from tqdm import tqdm
from evaluate import load

bertscore = load("bertscore")


import sys
sys.setrecursionlimit(3000)  # Increase limit (use cautiously!)


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


def print_tree(node: TreeNode, level=0):
    indent = "  " * level
    print(f"{indent}{node.content}")
    for child in node.children:
        print_tree(child, level + 1)


# tree)


def pre_order_traversal(node: TreeNode):
    """Perform pre-order traversal and return as a string."""
    result = [str(node.content)]
    for child in node.children:
        result.append(pre_order_traversal(child))
    return " ".join(result)


def post_order_traversal(node: TreeNode):
    """Perform post-order traversal and return as a string."""
    result = []
    for child in node.children:
        result.append(post_order_traversal(child))
    result.append(str(node.content))
    return " ".join(result)


def in_order_traversal(node: TreeNode):
    """Perform in-order traversal and return as a string (only for binary-like trees)."""
    if len(node.children) == 0:
        return str(node.content)
    elif len(node.children) == 1:
        return f"{in_order_traversal(node.children[0])} {node.content}"
    elif len(node.children) == 2:
        return f"{in_order_traversal(node.children[0])} {node.content} {in_order_traversal(node.children[1])}"
    else:
        # For non-binary trees, we can't do a strict in-order traversal
        return str(node.content) + " " + " ".join(in_order_traversal(child) for child in node.children)


def evaluate(prediction: TreeNode, target: TreeNode):
    """Evaluate the similarity between prediction and target trees using ROUGE."""
    rouge = Rouge()

    # Perform different traversals
    pred_pre_order = pre_order_traversal(prediction)
    target_pre_order = pre_order_traversal(target)
    # print(pred_pre_order,'\n')
    # print(target_pre_order,'\n')
    # print('='*50)

    pred_post_order = post_order_traversal(prediction)
    target_post_order = post_order_traversal(target)
    # print(pred_post_order,'\n')
    # print(target_post_order, '\n')
    # print('='*50)

    pred_in_order = in_order_traversal(prediction)
    target_in_order = in_order_traversal(target)
    # print(pred_in_order,'\n')
    # print(target_in_order, '\n')
    # print('='*50)


    bert_scores={
        "pre_order" : bertscore.compute(predictions=[pred_pre_order], references=[target_pre_order], lang='en', device="cuda:0"),
        "post_order" : bertscore.compute(predictions=[pred_post_order], references=[target_post_order], lang='en', device="cuda:0"),
        "in_order" : bertscore.compute(predictions=[pred_in_order], references=[target_in_order], lang='en', device="cuda:0")
    }      

    # Compute ROUGE scores for each traversal
    # scores = {
    #     "pre_order": rouge.get_scores(pred_pre_order, target_pre_order, avg=True),
    #     "post_order": rouge.get_scores(pred_post_order, target_post_order, avg=True),
    #     "in_order": rouge.get_scores(pred_in_order, target_in_order, avg=True),
    # }

    return bert_scores


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
    
    score_phi = evaluate(phi_tree, reference_tree)
    score_qwen = evaluate(qwen_tree, reference_tree)
    
    scores_phi.append(score_phi)
    scores_qwen.append(score_qwen)
    
    # print(input_wiki)
    # print('='*50)
    # print(reference)
    # print('='*50)
    # print(phi)
    # print('='*50)
    # print(qwen)
    # print('='*50)
    
with open('bert_scores_phi.json', 'w') as phi_file: 
    json.dump(scores_phi, phi_file)
    
with open('bert_scores_qwen.json', 'w') as qwen_file:
    json.dump(scores_qwen, qwen_file)
    
    # break
    
# Convert them to trees
# prediction_tree = to_tree(json1)
# target_tree = to_tree(json2)

# print_tree(prediction_tree)
# print('-'*50)
# print_tree(target_tree)
# print('-'*50)


# # Evaluate the similarity between the trees
# scores = evaluate(prediction_tree, target_tree)

# # Print the ROUGE scores
# print("ROUGE Scores:")
# for traversal, score in scores.items():
#     print(f"{traversal}: {score}")
