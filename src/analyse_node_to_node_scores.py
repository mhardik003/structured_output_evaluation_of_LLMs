import json
import numpy as np

with open('node_to_node_bert_scores_phi.json')as phi_scores, open('node_to_node_bert_scores_qwen.json') as qwen_scores:
    phi = json.load(phi_scores)
    qwen = json.load(qwen_scores)
    
    
def print_analyse(scores, similarity='max_similarity'):
    rouge_scores = [x[similarity] for x in scores]
    
    scores_min = min(rouge_scores)
    scores_max = max(rouge_scores)
    
    min_max_scaled = [((x-scores_min)/(scores_max-scores_min)) for x in rouge_scores]
    
    avg = sum(min_max_scaled) / len(min_max_scaled)
    std_dev = np.std(min_max_scaled)
    
    return avg, std_dev
    
    
for model in ['PHI', 'QWEN']:
    print(model)
    for similarity in ['max_similarity', 'min_similarity', 'avg_similarity']:
        print('\t', similarity)
        avg, std_dev = print_analyse(phi if model == 'PHI' else qwen, similarity)
        print(f'\t {similarity} - Average : {avg} Standard Deviation : {std_dev}')



##### ROUGE Score ########

# with open('rouge_scores_phi.json')as phi_scores, open('rouge_scores_qwen.json') as qwen_scores:
#     phi = json.load(phi_scores)
#     qwen = json.load(qwen_scores)
    
    
# def print_analyse(scores, rouge='rouge-1', order='pre_order', scoring='f'):
#     rouge_scores = [x[order][rouge][scoring] for x in scores]
#     avg = sum(rouge_scores) / len(rouge_scores)
#     return avg
    
    
# for model in ['PHI', 'QWEN']:
#     print(model)
#     for order in ['pre_order', 'post_order', 'in_order']:
#         print('\t', order)
#         for rouge in ['rouge-1', 'rouge-2', 'rouge-l']:
#             print('\t\t', rouge)
#             for scoring in ['f', 'p', 'r']:
#                 avg = print_analyse(phi if model == 'PHI' else qwen, rouge, order, scoring)
#                 print(f'\t\t\t {scoring} = {avg}')



###### BERT Score ######


# import json
# import numpy as np

# with open('bert_scores_phi.json')as phi_scores, open('bert_scores_qwen.json') as qwen_scores:
#     phi = json.load(phi_scores)
#     qwen = json.load(qwen_scores)
    
    
# def print_analyse(scores, order='pre_order', scoring='f'):
#     rouge_scores = [x[order][scoring][0] for x in scores]
    
#     scores_min = min(rouge_scores)
#     scores_max = max(rouge_scores)
    
#     min_max_scaled = [((x-scores_min)/(scores_max-scores_min)) for x in rouge_scores]
    
#     avg = sum(min_max_scaled) / len(min_max_scaled)
#     std_dev = np.std(min_max_scaled)
    
    
#     return avg, std_dev
    
    
# for model in ['PHI', 'QWEN']:
#     print(model)
#     for order in ['pre_order', 'post_order', 'in_order']:
#         print('\t', order)
#         # for rouge in ['rouge-1', 'rouge-2', 'rouge-l']:
#         #     print('\t\t', rouge)
#         for scoring in ['f1', 'precision', 'recall']:
#             av, std_dev = print_analyse(phi if model == 'PHI' else qwen, order, scoring)
#             print(f'\t\t {scoring} = {avg}')
#             print(f'\t\t std - {scoring} = {std_dev}\n')

