import mlx.core as mx
from mlx_lm import load
import numpy as np

model_id = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
model, tokenizer = load(model_id)
original_weight = mx.array(model.lm_head.weight)

def apply_surgery(shift_amount):
    model.lm_head.weight = original_weight
    corrupted_weight = (original_weight.astype(mx.int64) - shift_amount).astype(mx.uint32)
    model.lm_head.weight = corrupted_weight

def top_p_sampling(logits, p=0.9):
    probs = mx.softmax(logits, axis=-1)
    sorted_indices = mx.argsort(probs, axis=-1)[..., ::-1]
    sorted_probs = probs[..., sorted_indices]
    sorted_probs_np = np.array(sorted_probs)
    cumulative_probs = np.cumsum(sorted_probs_np, axis=-1)
    mask_np = cumulative_probs > p
    mask_np[..., 1:] = mask_np[..., :-1]
    mask_np[..., 0] = False
    sorted_logits = logits[..., sorted_indices]
    filtered_sorted_logits = mx.where(mx.array(mask_np), mx.array(-float("inf")), sorted_logits)
    reverse_indices = mx.argsort(sorted_indices, axis=-1)
    return filtered_sorted_logits[..., reverse_indices]

def min_p_sampling(logits, min_p=0.1):
    probs = mx.softmax(logits, axis=-1)
    p_max = mx.max(probs)
    threshold = p_max * min_p
    logits = mx.where(probs >= threshold, logits, mx.array(-float("inf")))
    return logits

def generate_text(prompt, sampler, max_tokens=25, **kwargs):
    tokens = tokenizer.encode(prompt)
    for _ in range(max_tokens):
        input_ids = mx.array(tokens)[None]
        logits = model(input_ids)[:, -1, :]
        logits = sampler(logits, **kwargs)
        probs = mx.softmax(logits, axis=-1)
        next_token = mx.random.categorical(probs, num_samples=1)
        tokens.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id: break
    return tokenizer.decode(tokens)

# 10 Iterations Sweep
experiments = [
    {"p": "What is 2+2?", "min_p": 0.01, "shift": 1000001},
    {"p": "What is 2+2?", "min_p": 0.05, "shift": 1000001},
    {"p": "What is 2+2?", "min_p": 0.1,  "shift": 1000001},
    {"p": "The capital of France is", "min_p": 0.05, "shift": 1000003},
    {"p": "The capital of France is", "min_p": 0.15, "shift": 1000003},
    {"p": "Once upon a time", "min_p": 0.02, "shift": 0}, # Healthy Model
    {"p": "Once upon a time", "min_p": 0.1,  "shift": 0}, # Healthy Model
    {"p": "The robot thought that humans", "min_p": 0.05, "shift": 1000001},
    {"p": "The robot thought that humans", "min_p": 0.1, "shift": 1000001},
    {"p": "A beautiful sunset over", "min_p": 0.05, "shift": 1000003}
]

print(f"{'Prompt':<30} | {'Min-P':<6} | {'Shift':<8} | {'Result'}")
print("-" * 100)

for ex in experiments:
    apply_surgery(ex["shift"])
    res = generate_text(ex["p"], min_p_sampling, min_p=ex["min_p"])
    # Clean up newline for display
    display_res = res.replace('\n', ' ').strip()
    print(f"{ex['p']:<30} | {ex['min_p']:<6} | {ex['shift']:<8} | {display_res}")

model.lm_head.weight = original_weight
