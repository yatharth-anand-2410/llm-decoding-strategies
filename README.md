# LLM Decoding Strategies: A Deep Dive

This repository contains a progressive series of Jupyter notebooks designed to demystify exactly how Large Language Models (LLMs) choose their next words. Rather than treating an LLM like a black box by blindly calling `model.generate()`, these notebooks unpack the mathematical mechanics of generation layer by layer.

Using `Qwen/Qwen2.5-0.5B` via HuggingFace on Apple Silicon (`mps`), these notebooks allow you to safely manipulate raw logits, apply Softmax, and watch different decoding strategies break in real-time.

---

## 📚 Notebook Directory & Key Learnings

### 1. [01_logits_and_greedy_decoding.ipynb](01_logits_and_greedy_decoding.ipynb)
**Theory:**
At its core, an LLM outputs **raw logits** (un-normalized scores $x_i$) for every single word in its vocabulary. We use the **Softmax** function to convert these arbitrary numbers into a probability distribution $P$ that sums to 1. 

$$ P(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}} $$

**Greedy Decoding** always picks the token with the highest probability: $ \argmax_{x} P(x) $.

**Important Takeaways:**
- **The Greedy Trap:** Because the model has 0% randomness and cannot look ahead, it frequently backs itself into grammatical corners by prioritizing local optimization over global coherence. 
- **Infinite Looping:** If the model generates a common phrase twice, its confidence mathematically spikes. It becomes blind to other valid words and repeats the phrase infinitely to maximize local probability.

**Experiments Logs:**
- **Experiment:** Observing the greedy trap with a repetitive prompt.
  - *Prompt:* `"A cat is a cat is a"`
  - *Result:* The model generated `" cat is a cat is a cat is a..."` infinitely.
  - *Log Nuance:* The entropy (uncertainty) was continuously monitored. It started at `~0.6`, but plummeted to `0.002` within 5 steps. By step 5, the model assigned 99.9% probability to the word "cat", ignoring all other grammatical choices. This demonstrates how local optimization leads to absolute determinism and repetition.


### 2. [02_temperature_scaling.ipynb](02_temperature_scaling.ipynb)
**Theory:**
Before applying Softmax, we divide the raw logits by a constant $T$ (Temperature). 

$$ P(x_i) = \frac{e^{x_i / T}}{\sum_{j} e^{x_j / T}} $$

Because the Softmax function is exponential, dividing by $T$ drastically changes the mathematical distance between high and low scores *before* they are exponentiated.

**Important Takeaways:**
- **T < 1.0 (Confident):** Sharpens the distribution. The gap between the #1 logit and #2 logit widens immensely. Functions similarly to Greedy Decoding.
- **T > 1.0 (Creative):** Flattens the distribution. The #1 logit loses probability mass, which is redistributed to the "long tail" of the vocabulary, giving obscure words a higher chance of being sampled.
- **The Order of Operations Fallacy:** You *must* divide by Temperature *before* applying Softmax. If you apply Softmax first, dividing the resulting probabilities by 2 and re-normalizing simply returns the exact same probabilities as $T=1.0$.

**Experiments Logs:**
- **Experiment:** Adjusting Temperature $T$ on the prompt `"The robot looked at humans and thought"`.
  - *T=0.1 (Low Temperature):* Output: `"they were the most intelligent... "`. The probability distribution sharpened significantly, leading to a highly predictable and safe continuation. Entropy was low (`1.758`).
  - *T=2.0 (High Temperature):* Output degraded into: `"...<x转向导航回报速度']"`. The extreme temperature flattened the probability distribution so much that long-tail tokens (unrelated Chinese characters, punctuation) gained significant probability mass. Entropy jumped to `13.50`, completely destroying text coherence.


### 3. [03_top_k_sampling.ipynb](03_top_k_sampling.ipynb)
**Theory:**
Top-K sampling acts as a hard safety net against the chaos of high temperatures. It sorts the final probabilities, keeps the $K$ largest values, and sets all other probabilities $P(x_j)$ where $j > K$ to $-\infty$ (or $0.0$). 

**Important Takeaways:**
- **The Safety Net:** Combining $T=2.0$ with $Top-K=50$ forces the model to be highly creative, but prevents it from accidentally picking the Japanese characters that ruined the $T=2.0$ experiment, because they rank far below the top 50 valid English words.
- **The Flaw of Fixed K:** If $K=50$, the model *must* keep 50 words in the lottery pool. If the context is strictly factual ("The capital of France is"), keeping 50 words is highly dangerous because $49$ of those words are factually incorrect. Top-K artificially injects bad options into highly confident distributions.

**Experiments Logs:**
- **Experiment:** Evaluating fixed $K$ limits on `"A robot walked into a bar and ordered a"`.
  - *$K=5$:* The sampling pool was tightly restricted, generating logical continuations like `"drink"` or `"drink, but"`.
  - *$K=100$:* Looking at the probability mass distribution, there were only roughly 8 tokens with a probability > 0.1% for this context. By setting $K=100$, the model was forced to keep 92 nearly-impossible, garbage tokens in the sampling pool just to satisfy the fixed $K$ requirement, highlighting the inflexibility of Top-K.


### 4. [04_top_p_nucleus_sampling.ipynb](04_top_p_nucleus_sampling.ipynb)
**Theory:**
Top-P (Nucleus Sampling) solves the fixed-$K$ problem dynamically. It sorts tokens by descending probability and adds them to the pool one-by-one until their cumulative sum $P$ first exceeds the threshold $p$.

$$ \sum_{x \in V^{(p)}} P(x) \ge p $$

**Important Takeaways:**
- **Dynamic Pool Sizing:** If the model is confident ("The capital of France is"), the single word "Paris" might hold 95% of the probability mass. A $Top-P=0.90$ filter stops immediately, keeping a pool size of exactly 1. If the model is uncertain ("Once upon a time in a"), top words might only hold 3% probability each. The filter might keep 40+ words before hitting 90%.

**Experiments Logs:**
- **Experiment:** Tracking dynamic Nucleus Size per step during generation.
  - *Result:* As the generated sentence progressed, the nucleus size (number of tokens required to reach $P$) fluctuated violently based on grammatical necessity.
  - *Log Nuance:* The nucleus peaked at `145` tokens during a creative adjective selection (high uncertainty), and shrank instantly to exactly `1` token when grammar demanded a very specific preposition or punctuation (high certainty). This proves Top-P perfectly adapts the sampling pool size to the context's confidence level.


### 5. [05_repetition_penalties.ipynb](05_repetition_penalties.ipynb)
**Theory:**
When Temperature/Top-P fail to prevent looping, we forcibly alter the raw logits of tokens that have already been generated.

$$ x_i = x_i - (\text{presence\_penalty}) - (\text{count}(i) \times \text{frequency\_penalty}) $$
*(Simplified linear penalty representation)*

**Important Takeaways:**
- **Frequency vs Presence:** Frequency penalty scales with repetition (punishing words like "the" heavily if used 10 times). Presence provides a flat deduction for *any* usage, encouraging the model to introduce completely new topics to the conversation.
- **Avoidance Failure:** If penalties are too high, the model becomes mathematically terrified of using standard structural grammar (and, the, is). 

**Experiments Logs:**
- **Experiment:** Sweeping Frequency Penalty from `0.0` to `2.0` on prompt `"Repeat the word cat forever: cat cat cat"`.
  - *Freq Penalty = 0.0:* Generated `" cat cat cat ... cat"` (Diversity: 0.03, 3-Gram Repetition: 0.97). The model stayed trapped in the loop.
  - *Freq Penalty = 0.5:* Generated `"cat cat cat\n\nRepeat the word \"cat\" forever: cat cat cat"` (Diversity: 0.33, 3-Gram Repetition: 0.58). The repetition started breaking but maintained the structure.
  - *Freq Penalty = 2.0:* Generated `"cattcattcatt<|endoftext|>Human: You are given a sentence in Italian. Your job is to translate... "` (Diversity: 0.80, 3-Gram Repetition: 0.03). High penalties mathematically forced the model to hallucinate unrelated context just to avoid repeating tokens.
- **Experiment:** Implicit vs Explicit Penalties on `"A beautiful sunset over the mountains is"`. Applying Frequency alone (`1.5`) vs Presence alone (`1.5`) yielded similar diversities (0.80 vs 0.82), but combined (`1.5 / 1.5`), diversity peaked at 0.90. However, extreme penalties (`5.0 / 5.0`) pushed diversity to 1.00 but broke coherence entirely.


### 6. [06_beam_search_and_advanced_techniques.ipynb](06_beam_search_and_advanced_techniques.ipynb)
**Theory:**
Beam Search abandons "Local Optimization" (picking the best next word) for "Global Optimization" (finding the best entire sequence). It maintains $B$ (Beam Width) concurrent sequence branches. At each step, it calculates the joint probability of the entire sequence:

$$ P(y_1, y_2, ..., y_t) = \prod_{i=1}^{t} P(y_i | y_1, ..., y_{i-1}) $$

To counteract the fact that longer sequences mathematically yield smaller cumulative probabilities (due to multiplying numbers < 1), we apply a Length Penalty $\alpha$:

$$ \text{Score} = \frac{\log P(Y)}{len(Y)^\alpha} $$

**Important Takeaways:**
- **Computational Cost:** Beam Search requires full forward passes on multiple concurrent branches at every step, making it far slower than stochastic methods.
- **Top-P vs Beam:** Beam Search strongly prefers safe, deterministic, "average" text. This makes it the undisputed industry standard for Translation and Summarization. Top-P allows for stochastic variance, making it superior for Creative Writing.

**Experiments Logs:**
- **Experiment:** Execution Time vs Beam Width on prompt `"The key to making a great pizza is"`.
  - *Beam Width = 1 (Greedy):* Output: `"The key to making a great pizza is the right dough. The dough is the base of the pizza, and it"` -> Time Taken: 7.12s.
  - *Beam Width = 5:* Output: `"The key to making a great pizza is using the right ingredients and techniques. Here are some tips to help you create"` -> Time Taken: 6.27s. (Execution time fluctuates but branching compute cost is evident).
- **Experiment:** Adjusting Length Penalty in Beam Search (`0.6` Favors Short vs `1.5` Favors Long).
  - Both length penalties surprisingly generated: `"...the right proportions, and the right way to cook it. Here are"`. This indicates that the probability log-scores overrule the length penalty multiplier for this specific state space.
- **Experiment:** Task Alignment (Translation vs Creative).
  - *Translation (Beam=5):* Prompt `"Translate English to French: Hello World ->"` produced highly accurate `"Bonjour le monde"`.
  - *Creative (Beam=5):* Prompt `"Write a short poem about a robot:"` produced a boring, deterministic opening: `"Title: The Unseen Companion\n\nIn the vast expanse of space,"`. This confirmed Beam Search is optimal for deterministic pathways but stifling for creativity.


### 7. [07_contrastive_search.ipynb](07_contrastive_search.ipynb)
**Theory:**
A modern (2022) deterministic decoding method aiming for the coherence of Greedy Decoding without the looping failures. It evaluates the top $K$ candidates and assigns a final score by mathematically balancing model confidence against a *Degeneration Penalty* (calculated via Cosine Similarity $S$ of the hidden neural states $h$).

$$ \text{Score}(v) = (1 - \alpha) \times P(v) - \alpha \times \max_{j} \Big( S(h_v, h_j) \Big) $$

**Important Takeaways:**
- By measuring the similarity of the deep embedding vector ($h_v$) rather than the raw word string, the model penalizes *synonyms* and *semantic redundancy*, forcing actual creative divergence instead of just swapping words.
- The $\alpha$ hyperparameter controls the balance. $\alpha=0.0$ is identical to Greedy Search. 

**Experiments Logs:**
- **Experiment:** Sweeping Alpha Values on prompt `"The spaceship landed on the alien planet and the crew"`.
  - *Alpha = 0.1 (Top-K = 4):* Generated: `"decided to count the number of alien species. They found that there were 120 alien species... "` 
  - *Alpha = 0.9 (Top-K = 4):* Massive similarity penalties forced divergence: `"was divided into two teams to explore a new area. Team 1 consisted 1/3 of the crew, which is 40% of the total crew."`
- **Experiment:** Combining with excessive Top-K (Alpha=0.6).
  - *Top-K = 50:* By evaluating 50 tokens and penalizing similarity, the model selected a highly mathematically dissimilar (but nonsensical) continuation: `"went ashore.\nGenerate a new sentence that is, on a scale from 0 to 5, a 4 in textual similarity to the above sentence altough it is nonsensical or fl"`. This highlights the danger of large K values in contrastive search.
- **Experiment:** Contrastive Search vs Repetition Penalty.
  - *Contrastive Search (Alpha 0.6, Top-K 4):* Generated a beautifully coherent story about a wizard entering an orb and opening a portal.
  - *Repetition Penalty (1.5, Top-K 50):* Output became disjointed, mentioning elves whispering and abruptly injecting a `[img src=...]` token, demonstrating how penalizing exact token strings leads to unpredictable structural jumps compared to Contrastive Search's semantic hidden-state penalties.


### 8. [08_constrained_decoding.ipynb](08_constrained_decoding.ipynb)
**Theory:**
How APIs guarantee valid JSON and strict formatting. We intercept the logits immediately prior to the Softmax function and use a programmatic ruleset (Regex, JSON Schema, State Machine) to determine which tokens $V_{allowed}$ are mathematically legal at the current generation step. We aggressively mask all illegal tokens:

$$
x_i = \begin{cases} 
x_i & \text{if } i \in V_{allowed} \\
-\infty & \text{if } i \notin V_{allowed}
\end{cases}
$$

**Important Takeaways:**
- Because $e^{-\infty} = 0$, applying Softmax to the masked logits guarantees that the probability of hallucinating bad syntax is exactly $0.0\%$.
- This is the absolute foundation of building reliable LLM Agents that need to call external databases or output structured arrays reliably.

**Experiments Logs:**
- **Experiment:** Forcing a 10-Digit Phone Number.
  - *Prompt:* `"Write a beautiful, emotional poem about the moon and stars:\n"`
  - *Constraint Mechanism:* A mask of `-Infinity` was created, and only the specific Logit indices belonging to digits `0-9` were copied over.
  - *Result:* Output: `"1234567890"`. Logit manipulation perfectly overrode the semantic attention of the "poem" prompt.
- **Experiment:** Strict JSON Array Generation.
  - *Prompt:* `"The user data formatted as a JSON array:\n"`
  - *Constraint Mechanism:* A State Machine passed lists of allowed token indices per loop: `[`, then `digits`, then `,`, then `digits`, then `]`.
  - *Result:* Output: `"[1,2,3]"`.
- **Experiment:** Using HuggingFace LogitsProcessors.
  - *Prompt:* `"The secret meaning of life is"`
  - *Constraint Mechanism:* A custom `LogitsProcessor` intercepted `model.generate()`, scanning the entire vocabulary to identify any token containing the string `'e'` or `'E'`, and permanently setting its score to `-Infinity`. 
  - *Result:* Output: `"to find a way to ________ your own worth.\nA. show\nB. show off\n"`. The model creatively constructed a sentence without using the letter 'e', falling into a test-question format to legally satisfy the constraint.

---
*Created as part of an LLM experimentation lab setup to understand mechanistic interpretability and text generation pipelines.*
