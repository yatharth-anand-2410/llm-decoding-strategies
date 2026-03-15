# Gemini Context: LLM Decoding Strategies

## Completed Experiments

### 1. Stochastic vs. Deterministic Decoding
- **Exploration:** Greedy Search vs. Temperature, Top-K, and Top-P (Nucleus) sampling.
- **Finding:** Dynamic pool sizing in Top-P is critical for adapting to varying context confidence.

### 2. Advanced & Semantic Decoding
- **Exploration:** Beam Search, Repetition Penalties, and Contrastive Search.
- **Finding:** Contrastive Search is superior for creative coherence because it penalizes semantic redundancy in the hidden state, not just exact token repetition.

### 3. Inference Engineering (Core Discovery)
- **Experiment:** Using decoding strategies to "rescue" a model with corrupted weights (from the MLX Research Lab experiment).
- **Key Finding:** Top-K is the most robust filter for bit-level noise. Contrastive Search can navigate "wobbly" weights better than Greedy Search because its hidden states remain uncorrupted by the `lm_head` perturbations.

## Summary of Findings
- **Decoding as a Shield:** The choice of decoding algorithm is not just a stylistic preference; it is a vital layer of structural integrity that can filter out noise and improve model reliability.

## Advanced Research Tracks (Pending)

### 1. Speculative Decoding Experiments
- **Scope:** Implement and evaluate Speculative Decoding in MLX to accelerate inference. This involves using a fast "draft" model to generate candidate tokens, which are then verified in parallel by the larger "target" model.
- **Track A: Standard Speculative Decoding:**
  - **Goal:** Pair the Llama 3 8B target model with a smaller/faster draft model (e.g., Llama 3 1B or a heavily quantized variant).
  - **Metrics:** Measure the acceleration factor (tokens/sec), acceptance rate, and memory overhead of dual KV-cache management.
- **Track B: "Noisy" Drafts (Cross-Project Integration):**
  - **Goal:** Use a weight-corrupted version of Llama 3 8B (at the 1,000,001 shift mark) as a "Draft Model" for a clean target version.
  - **Insight:** Measure the acceptance rate of "noisy" tokens to find the threshold where structural weight corruption destroys the draft model's utility.

### 2. Entropy-Triggered "Self-Healing"
- **Goal:** Monitor token entropy in real-time during generation.
- **Experiment:** Automatically switch from Greedy to Contrastive/Beam search if entropy spikes (indicating confusion).
- **Insight:** Can a model "notice" it's about to hallucinate and change its own strategy?

### 3. Topological Mapping of the "Hallucination Well"
- **Goal:** Use high-temperature sampling on a corrupted model to generate 100+ responses.
- **Insight:** Map whether errors are random noise or if the model falls into specific "attractors" (repetitive loops or specific multilingual garbage).
