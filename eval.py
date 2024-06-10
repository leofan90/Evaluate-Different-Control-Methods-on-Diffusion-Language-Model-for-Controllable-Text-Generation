import torch
from evaluate import load
from transformers import PreTrainedTokenizerBase
from sentence_transformers import SentenceTransformer
from nltk.util import ngrams
from collections import defaultdict
import numpy as np
import wandb
import pandas as pd
import json
import spacy
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def read_file_lines(filename):
    """
    Read lines from a text file and store them in a list.

    :param filename: The path to the text file.
    :return: A list containing the lines of the text file.
    """
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            lines.append(line.strip())  # 去除每行末尾的换行符并添加到列表中
    return lines


def compute_perplexity(all_texts_list, model_id='gpt2-large'):
    torch.cuda.empty_cache() 
    perplexity = load("/home/huangxiuyuan/evaluate-main/metrics/perplexity", module_type="metric")
    results = perplexity.compute(predictions=all_texts_list, model_id=model_id, device='cuda')
    return results['mean_perplexity']

def compute_diversity(all_texts_list):
    ngram_range = [2,3,4]

    tokenizer = spacy.load("en_core_web_sm").tokenizer
    token_list = []
    for sentence in all_texts_list:
        token_list.append([str(token) for token in tokenizer(sentence)])
    ngram_sets = {}
    ngram_counts = defaultdict(int)

    metrics = {}
    for n in ngram_range:
        ngram_sets[n] = set()
        for tokens in token_list:
            ngram_sets[n].update(ngrams(tokens, n))
            ngram_counts[n] += len(list(ngrams(tokens, n)))
        metrics[f'{n}gram_repitition'] = (1-len(ngram_sets[n])/ngram_counts[n])
    diversity = 1
    for val in metrics.values():
        diversity *= (1-val)
    metrics['diversity'] = diversity
    return metrics

def compute_mauve(all_texts_list, human_references, model_id='gpt2-large'):
    torch.cuda.empty_cache() 
    assert model_id in {'gpt2-large', 'all-mpnet-base-v2'}
    mauve = load("/home/huangxiuyuan/evaluate-main/metrics/mauve")
    assert len(all_texts_list) == len(human_references)

    if model_id == 'all-mpnet-base-v2':
        model = SentenceTransformer(model_id).cuda()
        #Sentences are encoded by calling model.encode()
        all_texts_list_embedding = model.encode(all_texts_list)
        human_references_embedding = model.encode(human_references)
        results = mauve.compute(predictions=all_texts_list, p_features=all_texts_list_embedding, references=human_references, q_features=human_references_embedding, featurize_model_name=model_id, max_text_length=256, device_id=0, mauve_scaling_factor=8,)
    elif model_id == 'gpt2-large':
        results = mauve.compute(predictions=all_texts_list, references=human_references, featurize_model_name=model_id, max_text_length=256, device_id=0)
    else:
        raise NotImplementedError
    
    return results.mauve, results.divergence_curve


def func(x):
    if x:   return 1    # True
    else:   return 0    # False

# # Refernece Text: humor-speech
# df = pd.read_csv('/home/huangxiuyuan/huggingface_cache/dataset/humor-speech/dev.csv')
# df['humor'] = df['humor'].apply(func)
# df.rename(columns={'humor': 'label'}, inplace=True)

# df_0 = df[df["label"] == 0] # False
# df_1 = df[df["label"] == 1] # True

# random_rows_0 = df_0.sample(n=500)
# random_rows_1 = df_1.sample(n=500)

# merged_df = pd.concat([random_rows_0, random_rows_1], ignore_index=True)
# merged_df.loc[:, 'idx'] = range(1, len(merged_df) + 1)
# reference_list = merged_df['text'].to_list()


# Embedding-based
# Embedding_df = pd.read_csv('/home/huangxiuyuan/controlled_text_generation/gen_data/synth_sample1000_seed42-2.csv')
# Embedding_texts_list = Embedding_df["text"].to_list()
# print(compute_perplexity(Embedding_texts_list),"\n")    # 211.41539874839782
# print(compute_diversity(Embedding_texts_list))          # {'2gram_repitition': 0.7617320335524824, '3gram_repitition': 0.7228807324450257, '4gram_repitition': 0.7054215258613015, 'diversity': 0.019450617300228464}
# print(compute_mauve(Embedding_texts_list, reference_list), "\n")    # 0.334199915272815

# # Classifier-based
# Embedding_df = pd.read_csv('/home/huangxiuyuan/controlled_text_generation/gen_data/synth_sample1000_seed42_humor_speech_classifier.csv')
# Embedding_texts_list = Embedding_df["text"].to_list()
# print(compute_perplexity(Embedding_texts_list),"\n")    # 40.39500158882141
# print(compute_diversity(Embedding_texts_list))          # {'2gram_repitition': 0.8025753283543652, '3gram_repitition': 0.6759706760792832, '4gram_repitition': 0.5932242319839218, 'diversity': 0.02602200840149496}
# print(compute_mauve(Embedding_texts_list, reference_list), "\n")    # 0.02432942517912174


# # Embedding-Classifier-both
# Embedding_df = pd.read_csv('/home/huangxiuyuan/controlled_text_generation/gen_data/synth_sample1000_seed42_humor_speech_both.csv')
# Embedding_texts_list = Embedding_df["text"].to_list()
# print(compute_perplexity(Embedding_texts_list),"\n")    # 30.553559171676635
# print(compute_diversity(Embedding_texts_list))          # {'2gram_repitition': 0.7792797142029545, '3gram_repitition': 0.614920377320215, '4gram_repitition': 0.5045935263326567, 'diversity': 0.0421070159467441}
# print(compute_mauve(Embedding_texts_list, reference_list), "\n")    # 0.049433711530963556


# Train data
# with open("/home/huangxiuyuan/huggingface_cache/dataset/humor-speech/train-500.json", 'r', encoding='utf-8') as file:
#     dict_list = json.load(file)
# all_texts_list = []
# for data in dict_list:
#     all_texts_list.append(data["text"])
# print(compute_perplexity(all_texts_list),"\n")    # 166.38666284561157
# print(compute_diversity(all_texts_list))          # {'2gram_repitition': 0.149783678949724, '3gram_repitition': 0.03304852490730292, '4gram_repitition': 0.011222163773452531, 'diversity': 0.8128919837831726}
# print(compute_mauve(all_texts_list, reference_list), "\n")  # 0.9654790374547885



# No Condition
# filename = '/home/huangxiuyuan/controlled_text_generation/gen_data/eval42-cond1_nucleus-sample-2.txt'
# lines_list = read_file_lines(filename)
# print(compute_perplexity(lines_list),"\n")    # 128.29301215171813
# print(compute_diversity(lines_list))          # {'2gram_repitition': 0.18645558487247138, '3gram_repitition': 0.0804967801287948, '4gram_repitition': 0.05255544840887172, 'diversity': 0.708742253429782}
# print(compute_mauve(lines_list, reference_list), "\n")  # 0.8547176539278489

# ========================================================================================================================

# Refernece Text: Ag news
filename = '/home/huangxiuyuan/controlled_text_generation/gen_data/eval42-cond1_nucleus-sample-2.txt'
lines_list = read_file_lines("/home/huangxiuyuan/controlled_text_generation/gen_data/ag_news-dev.jsonl")
dict_list = [json.loads(item) for item in lines_list]
df = pd.DataFrame(dict_list)

random_df = df.sample(n=1000, replace=True)
reference_list = random_df['title'].to_list()

# Ag news
filename = '/home/huangxiuyuan/controlled_text_generation/gen_data/eval42-cond1_nucleus-sample-2.txt'
lines_list = read_file_lines("/home/huangxiuyuan/controlled_text_generation/gen_data/ag_news-dev.jsonl")
dict_list = [json.loads(item) for item in lines_list]
pd_lines = pd.DataFrame(dict_list)
text_list = pd_lines["title"].to_list()
print(compute_perplexity(lines_list),"\n")    # 41.772251938177455
print(compute_diversity(lines_list))          # {'2gram_repitition': 0.5000986128459668, '3gram_repitition': 0.34150414106331817, '4gram_repitition': 0.273688318740243, 'diversity': 0.23908945331864997}
print(compute_mauve(lines_list, reference_list), "\n")  # 

