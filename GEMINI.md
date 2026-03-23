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

## Advanced Research Tracks (Pending / In Progress)

### 1. Speculative Decoding Experiments
- **Status:** Completed.
- **Scope:** Evaluated Speculative Decoding with corrupted draft models.

### 2. Min-P Sampling
- **Goal:** Implement Min-P dynamic thresholding to test robustness against the Top-P "garbage magnet" effect at high temperatures and bit-level corruption.

### 3. DoLa (Decoding by Contrasting Layers)
- **Goal:** Contrast late "mature" layers with early "premature" layers to surface deeper factual recall and reduce hallucination.

### 4. Entropy-Triggered "Self-Healing"
- **Goal:** Monitor token entropy in real-time during generation.
- **Experiment:** Automatically switch from Greedy to Contrastive/Beam search if entropy spikes or drops to near-zero (indicating a loop).
- **Insight:** Can a model "notice" it's failing and change its own strategy?

### 5. Lookahead Decoding
- **Goal:** Break the autoregressive bottleneck via Jacobi Iteration without relying on a separate draft model. Measure tokens/sec vs. Speculative Decoding.

### 6. Topological Mapping of the "Hallucination Well"
- **Goal:** Use high-temperature sampling on a corrupted model to generate 100+ responses.
- **Insight:** Map whether errors are random noise or if the model falls into specific "attractors" (repetitive loops or specific multilingual garbage).
