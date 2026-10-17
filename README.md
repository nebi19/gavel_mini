# gavel_mini

A small, from-scratch re-implementation of **GAVEL** (rule-based activation monitoring for LLM safety) on a 1.5B-parameter open model, built to understand the method end to end and to test it on new questions, including Amharic.

> Based on: Rozenfeld et al., *GAVEL: Towards Rule-Based Safety through Activation Monitoring*, ICLR 2026 ([arXiv:2601.19768](https://arxiv.org/abs/2601.19768)) from the [Offensive AI Lab](https://offensive-ai-lab.com/). This is an independent learning project, not the official code. The official implementation is at [Offensive-AI-Lab/gavel](https://github.com/Offensive-AI-Lab/gavel).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nebi19/gavel_mini/blob/main/Mini_GAVEL_Lab.ipynb)

## The idea in one paragraph

Instead of training one detector per harm ("is this a tax scam?"), GAVEL detects small, interpretable **Cognitive Elements (CEs)** in the model's internal activations, such as *threaten*, *taxation* or *payment tools*, and fires human-written **rules** over them, such as `stop if threaten AND taxation`. Because a topic alone never completes a rule, an honest tax question should pass while a tax scam is caught. Rules can be changed without retraining.

## What the notebook does

`Mini_GAVEL_Lab.ipynb` runs on a free Colab T4 GPU in about 20 minutes.

| Step | What | Paper |
|---|---|---|
| 0–1 | Load **Qwen2.5-1.5B-Instruct** (28 layers, d = 1536) | Paper used Mistral-7B |
| 2 | Inspect hidden states, shape `(batch, tokens, d)` | §2 |
| 3 | Forward hooks on the **attention output** of layers 11–21 (the paper's 40–80% depth), concatenated into one 16,896-dim vector per token | §3.0.1 |
| 4 | Excitation data: 5 CEs × 20 sentences (`threaten`, `taxation`, `payment_tools`, `personal_info`, `build_trust`) + 20 neutral sentences | §3.0.1 (300 per CE) |
| 5 | **ERI** prompt, *"Think about ⟨CE⟩ while revising the following: ⟨sentence⟩"*, keeping only generated-token activations | §3.0.1, Fig. 2 |
| 6 | Multi-label MLP detector (sigmoid outputs, BCE, Adam 3e-4), split by sentence | §3.0.2 (paper: 3-layer GRU) |
| 7 | A tiny rule compiler: `stop if threaten AND taxation` → Boolean function | §3.1 |
| 8 | Live monitor: per-token CE scores, time window, `min_hits = 2`, first token where each rule fires | §3.1, Fig. 4 |
| 9 | Experiments: window size, threshold, Amharic, no neutral data, attention vs MLP | §4.2, App. B |
| 10–12 | Per-CE threshold calibration (Youden's J), a test set with near-misses, TPR / FPR / bACC / F1 / AUC per rule | §4, App. D, G |
| 13 | Baseline: the same model as a text-only yes/no judge | App. G.2 (GPT-4 judge) |
| 14 | Misdirection stress test: harmful text wrapped in benign filler | Table 11 |

## Results (first run)

### The CE detector learns its training data perfectly

On held-out excitation sentences (threshold 0.5):

| CE | TPR | FPR |
|---|---|---|
| threaten | 1.00 | 0.00 |
| taxation | 1.00 | 0.00 |
| payment_tools | 0.97 | 0.00 |
| personal_info | 1.00 | 0.00 |
| build_trust | 0.99 | 0.00 |

The CE signatures are clearly distinct inside the model (cosine similarity of CE centroids):

![CE similarity heatmap](assets/ce_similarity_heatmap.png)

### …but it does not yet transfer to real conversations

On a 9-conversation test set (4 harmful, 5 near-miss), with calibrated thresholds:

| Rule | TPR | FPR | bACC | F1 | AUC |
|---|---|---|---|---|---|
| Tax authority scam | 1.00 | **1.00** | 0.50 | 0.36 | 1.00 |
| Payment info grab | 1.00 | 0.29 | 0.86 | 0.67 | 0.93 |
| *LLM judge, tax scam* | 1.00 | 0.00 | 1.00 | 1.00 | – |
| *LLM judge, payment grab* | 1.00 | 0.00 | 1.00 | 1.00 | – |

Every harmful conversation was caught, but every near-miss also fired the tax scam rule, and the text-only judge did better. Misdirection did not change either system's results on this small set.

### Diagnosis: the hidden system prompt triggers *threaten*

![Honest tax question](assets/monitor_honest_tax_question.png)

On the honest question *"How do I pay my income tax online?"*, **`threaten` rises above 0.5 at token positions ~8–21, before the user's message starts (~token 25)**. These positions most likely belong to the chat template's default system prompt (*"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."*), which the model inserts automatically and which never appeared in training. Once `threaten` is ticked by the template, any later mention of taxes completes the tax-scam rule.

For comparison, the scam itself (both CEs come from the assistant's words):

![Tax scam](assets/monitor_tax_scam.png)

Two other factors likely contribute:

1. **Calibration on the wrong data.** Thresholds were tuned on held-out *excitation* tokens, which are perfectly separable, so the selected cut-offs are very low (0.05–0.45) and fire easily on real dialogue. The paper calibrates on realistic multi-turn conversations.
2. **Tiny, uniform training data.** 20 short sentences per CE, generated through one prompt style, are easy to separate but don't cover real conversational text.

### Amharic (cross-lingual) observation

| | Words | Tokens | Tokens per word |
|---|---|---|---|
| English tax scam | 42 | 47 | 1.1 |
| Amharic tax scam | 18 | 119 | **6.6** |

The English-only detector **did fire** on the Amharic tax scam, but it also fired on the honest Amharic tax question, so the same false-alarm problem applies. Amharic needs about 6× more tokens per word than English for this model, which may spread a concept's signal thin. This is the question the project aims to study properly.

## Next steps

- [ ] Score only the tokens of real `user` / `assistant` messages (mask the system prompt and template tokens), then re-run Steps 8–14
- [ ] Calibrate thresholds on multi-turn conversations, as the paper does
- [ ] Grow each CE to 100–300 varied examples, and add template-style text to the neutral set
- [ ] Replace the MLP with the paper's GRU over 5-token segments
- [ ] Grow the test set to 20+ harmful and 20+ near-miss conversations per rule
- [ ] Build a small Amharic / Tigrinya test set and compare English-only vs. multilingual CE data
- [ ] Add East African mobile-money scam CEs (e.g. a mistaken-transfer claim)

## Run it yourself

1. Click **Open in Colab** above (or upload the notebook to [colab.research.google.com](https://colab.research.google.com)).
2. **Runtime → Change runtime type → T4 GPU**.
3. Run the cells top to bottom. Step 1 downloads the model (~3 GB); Step 5 takes the longest.
4. If you run out of memory, set `MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"` in Step 1.

## Author

Nebiat Hailemariam. Feedback and issues are welcome.
