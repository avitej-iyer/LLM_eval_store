import pandas as pd
import numpy as np
import sys
from llm_analysis_utils import process_analysis
from semanticSimFunctions import getSentenceEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch 

sys.path.insert(0,'/scratch/aqi5157/pip_packages_and_cache/packages')

def GO_dataset_score_performance(
    dataset_path,
    sep,
    model_name,
    # Weights for the final dataset-level formula:
    alpha=0.2,    # Weight for average similarity on real sets
    beta=0.2,     # Weight for average rank score on real sets
    gamma=0.2,    # Weight for (1 - FPR)
    delta=0.2,    # Weight for (1 - FNR)
    epsilon=0.2   # Weight for average confidence (optional)
):
    """
    Reads a dataset of GO gene sets, contaminated sets, and LLM outputs,
    then computes a single 'LLM performance score' for the entire dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset file (CSV, TSV, etc.).
    sep : str
        Delimiter used in the dataset file (e.g., ',' or '\t').
    model_name : str
        The identifier for the LLM in the dataframe (e.g., 'gemma').
        Helps identify which columns to use.
    alpha, beta, gamma, delta, epsilon : float
        Weights for the various components in the final scoring formula.

    Returns
    -------
    float
        The LLM performance score (a single number).
    """

    # -----------------------------
    # 1. LOAD THE DATA
    # -----------------------------
    df = pd.read_csv(dataset_path, sep=sep)
    num_elements = df.shape[0]


    go_term_total_count = 11943

    sim_column = "LLM_name_GO_term_sim"            # similarity measure
    rankscore_column = "sim_rank"  # normalized rank in [0,1]
    confidence_column = f"{model_name}_assigned_score" # LLM confidence, e.g. "gemma_assigned_score"

    # false positive and false negative calulation

    fpr, fnr = 0,0

    false_negative_count = df[f"{model_name}_assigned_name"].eq("System of unrelated proteins").sum() + df[f"{model_name}_assigned_name"].eq("system of unrelated proteins").sum()
    
    false_positive_count = num_elements - (df[f"{model_name}_assigned_name_100_perc"].eq("System of unrelated proteins").sum() + df[f"{model_name}_assigned_name_100_perc"].eq("system of unrelated proteins").sum())

    fnr = false_negative_count/num_elements
    fpr = false_positive_count/num_elements


    count = 0    
    # Apply the formula to compute RankScore for each row and accumulate it in count
    for rank_i in df['sim_rank']:
        rank_score = 1 - ((rank_i - 1) / (go_term_total_count - 1))
        count += rank_score

    print("Count is : ", count)
    
    avg_rankscore_real = count/num_elements

    avg_conf_all = df[f"{model_name}_assigned_score"].mean()
    avg_sim_real = df["LLM_name_GO_term_sim"].mean()



    llm_performance = (
        alpha  * avg_sim_real +
        beta   * avg_rankscore_real +
        gamma  * (1 - fpr) +
        delta  * (1 - fnr) +
        epsilon * avg_conf_all
    )

    print("Average rank score = ", avg_rankscore_real)
    print("Average sim real = ", avg_sim_real)
    print("False positive rate = ", fpr)
    print("False negative rate = ", fnr)

    return llm_performance

def GO_single_score_performance(query, GO_name, GO_term_embeddings):
    llm_name, llm_score, llm_analysis = process_analysis(query)

    tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

    assigned_name_emb = getSentenceEmbedding(llm_name, tokenizer, model)
    
    go_term_emb = GO_term_embeddings.get(GO_name, None)
    if go_term_emb is None:
        raise ValueError(f"No embedding found for the GO term: '{actual_go_term}'")

    if isinstance(assigned_name_emb, torch.Tensor):
        assigned_name_emb = assigned_name_emb.detach().cpu().numpy()
    if isinstance(go_term_emb, torch.Tensor):
        go_term_emb = go_term_emb.detach().cpu().numpy()

    # Reshape if needed (to 2D) for cosine_similarity
    assigned_name_emb = assigned_name_emb.reshape(1, -1)
    go_term_emb = go_term_emb.reshape(1, -1)

    # compute cosine sim
    similarity = float(cosine_similarity(assigned_name_emb, go_term_emb)[0][0])

    # Rank the assigned name by comparing similarity to *all* GO terms

    rank = 1  # We'll start from 1 => the best if no other term beats it
    for other_go_term, emb in GO_term_embeddings.items():
        # Skip the actual GO term if you prefer, or keep it if you want to measure
        # how often the assigned name is closer to other terms
        if other_go_term == GO_name:
            continue

        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        emb = emb.reshape(1, -1)

        # Compare similarity with this "other" GO term
        other_sim = float(cosine_similarity(assigned_name_emb, emb)[0][0])
        if other_sim > similarity:
            rank += 1

    # normalize rank => rankscore in [0,1]

    N = len(GO_term_embeddings)
    # rank_score = 1 - (rank - 1) / (N - 1)
    if N > 1:
        rank_score = 1 - (rank - 1) / (N - 1)
    else:
        rank_score = 1.0  # trivial case if only 1 term

    final_score = (similarity + rank_score) / 2

    return [final_score, similarity, rank_score]


