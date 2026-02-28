import torch
from collections import Counter
import math

def calculate_entropy(probabilities):
    """
    Calculates the Shannon entropy of a probability distribution.
    Lower entropy means the model is more confident (spiky distribution).
    Higher entropy means the model is less confident (flat distribution).
    """
    epsilon = 1e-10
    prob_tensor = probabilities.clone().detach() # ensure it's a tensor
    prob_tensor = torch.clamp(prob_tensor, min=epsilon)
    
    entropy_val = -torch.sum(prob_tensor * torch.log2(prob_tensor), dim=-1)
    return entropy_val.item()

def token_diversity_score(generated_tokens):
    """
    Calculates the ratio of unique tokens to total tokens.
    1.0 means every token is unique.
    Lower values indicate more repetition.
    """
    if not generated_tokens:
        return 0.0
    
    unique_tokens = set(generated_tokens)
    return len(unique_tokens) / len(generated_tokens)

def repetition_score(generated_tokens, n=3):
    """
    Calculates how often n-grams repeat in the generated text.
    Returns a score from 0.0 (no repetition) to 1.0 (highly repetitive).
    """
    if len(generated_tokens) < n:
        return 0.0
        
    ngrams = [tuple(generated_tokens[i:i+n]) for i in range(len(generated_tokens)-n+1)]
    ngram_counts = Counter(ngrams)
    
    total_ngrams = len(ngrams)
    unique_ngrams = len(ngram_counts)
    
    score = 1.0 - (unique_ngrams / total_ngrams)
    return score

def track_probability_mass(probabilities, top_n=10):
    """
    Returns the cumulative probability mass captured by the top N tokens.
    This helps visualize how "spiky" vs "flat" the distribution is.
    """
    top_probs, _ = torch.topk(probabilities, k=top_n, dim=-1)
    cumulative_mass = torch.sum(top_probs, dim=-1).item()
    return cumulative_mass
