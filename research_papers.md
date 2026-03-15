# 📚 Research Papers: LLM Decoding Strategies

These papers cover the mathematical and algorithmic techniques for word selection (decoding) and how these methods influence model coherence, creativity, and robustness.

### Core Papers

*   **[The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)** (Holtzman et al., 2019)
    *   **Relevance:** The primary paper for **Top-P (Nucleus) Sampling**. It explains why standard decoding methods (greedy, top-K) often lead to repetitive or incoherent text and how the "nucleus" of a probability distribution can be sampled for better results.
    *   **Key Concept:** Dynamic pool sizing and the "unreliability" of the long-tail of token probabilities.

*   **[A Contrastive Framework for Neural Text Generation](https://arxiv.org/abs/2202.06417)** (Su et al., 2022)
    *   **Relevance:** The origin for **Contrastive Search**. It describes how hidden-state similarity can be used as a penalty term to force semantic divergence and increase text diversity without sacrificing coherence.
    *   **Key Concept:** Balancing model confidence against a "Degeneration Penalty" calculated from internal neural activations.

*   **[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)** (GPT-3 Paper - Brown et al., 2020)
    *   **Relevance:** Important for understanding the context-sensitivity and few-shot capabilities of LLMs, which were leveraged in the "Decoding vs. Weight Surgery" experiments.
    *   **Key Concept:** In-context learning (ICL) and its impact on the residual stream's stability.


*   **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)** (Leviathan et al., 2022) / **[Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)** (Chen et al., 2023)
    *   **Relevance:** The foundational papers introducing **Speculative Decoding**. They demonstrate how drafting tokens with a smaller model and verifying them in parallel with a larger target model can significantly speed up inference without altering the output distribution.
    *   **Key Concept:** Rejection sampling, probability verification $p(x)/q(x)$, and the mathematical guarantee that the output remains identical to the target model.

---
*Generated for the Decoding Strategies context.*