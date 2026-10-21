# gavel_mini

A small, from-scratch re-implementation of **GAVEL** (rule-based activation monitoring for LLM safety) on a 1.5B-parameter open model, built to understand the method end to end and to test it on new questions, including Amharic.

> Based on: Rozenfeld et al., *GAVEL: Towards Rule-Based Safety through Activation Monitoring*, ICLR 2026 ([arXiv:2601.19768](https://arxiv.org/abs/2601.19768)) from the [Offensive AI Lab](https://offensive-ai-lab.com/). This is an independent learning project, not the official code. The official implementation is at [Offensive-AI-Lab/gavel](https://github.com/Offensive-AI-Lab/gavel).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nebi19/gavel_mini/blob/main/Mini_GAVEL_Lab.ipynb)

## The idea in one paragraph

Instead of training one detector per harm ("is this a tax scam?"), GAVEL detects small, interpretable **Cognitive Elements (CEs)** in the model's internal activations, such as *threaten*, *taxation* or *payment tools*, and fires human-written **rules** over them, such as `stop if threaten AND taxation`. Because a topic alone never completes a rule, an honest tax question should pass while a tax scam is caught. Rules can be changed without retraining.

## What the notebook does

`Mini_GAVEL_Lab.ipynb` runs on a free Colab T4 GPU in about 30–40 minutes. The committed version includes the outputs of a full run.

| Step | What | Paper |
|---|---|---|
| 0–1 | Load **Qwen2.5-1.5B-Instruct** (28 layers, d = 1536) | Paper used Mistral-7B |
| 2 | Inspect hidden states, shape `(batch, tokens, d)` | §2 |
| 3 | Forward hooks on the **attention output** of layers 11–21 (the paper's 40–80% depth), concatenated into one 16,896-dim vector per token | §3.0.1 |
| 4 | Excitation data: 5 CEs × 20 sentences (`threaten`, `taxation`, `payment_tools`, `personal_info`, `build_trust`) + 20 neutral sentences | §3.0.1 (300 per CE) |
| 5 | **ERI** prompt, *"Think about ⟨CE⟩ while revising the following: ⟨sentence⟩"*, keeping only generated-token activations | §3.0.1, Fig. 2 |
| 6 | Multi-label MLP detector (sigmoid outputs, BCE, Adam 3e-4), split by sentence | §3.0.2 (paper: 3-layer GRU) |
| 7 | A tiny rule compiler: `stop if threaten AND taxation` → Boolean function | §3.1 |
| 8 | Live monitor: per-token CE scores, time window, `min_hits = 2`. **Only real user/assistant text is scored**; the chat template and hidden system prompt are masked out | §3.1, Fig. 4 |
| 9 | Experiments: window size, threshold, Amharic, no neutral data, attention vs MLP | §4.2, App. B |
| 10–12 | Per-CE threshold calibration (Youden's J), a test set with near-misses, TPR / FPR / bACC / F1 / AUC per rule | §4, App. D, G |
| 13 | Baseline: the same model as a text-only yes/no judge | App. G.2 (GPT-4 judge) |
| 14 | Misdirection stress test: harmful text wrapped in benign filler | Table 11 |
| 15 | **Attack:** six ways to hide the threat in a tax scam (CE-level jailbreaks) | §4 "Adversarial attacks", Conclusion |
| 16 | **Defend:** adversarial training of `threaten` with new hidden-threat examples and hard negatives | – |

## Results

All numbers come from very small test sets (2–9 conversations each). They are early signals, not statistically solid findings.

### 1. The CE detector learns its training data

On held-out excitation sentences (threshold 0.5):

| CE | TPR | FPR |
|---|---|---|
| threaten | 1.00 | 0.00 |
| taxation | 1.00 | 0.00 |
| payment_tools | 0.97 | 0.00 |
| personal_info | 1.00 | 0.00 |
| build_trust | 0.99 | 0.00 |

![CE similarity heatmap](assets/ce_similarity_heatmap.png)

### 2. A bug found and fixed: the hidden system prompt triggered `threaten`

In the first run, the tax-scam rule fired on **every** harmless conversation (FPR 1.00). The per-token plots showed why: the first ~22 tokens looked identical in all six test conversations, with `threaten` above 0.5. Those tokens are Qwen's default system prompt (*"You are Qwen, created by Alibaba Cloud…"*), which the chat template inserts automatically. The `<|im_end|>` marker at the end of each message also spiked.

**Fix:** score only the characters of real user/assistant messages and set template tokens to zero (Step 8, `ce_probs`).

| Tax-scam rule, Step 12 | Before fix | After fix |
|---|---|---|
| Honest tax question | flagged | ✅ allowed |
| False-alarm rate (FPR) | 1.00 | **0.29** |

After the fix, the honest question shows only `taxation` and `payment_tools`, never `threaten`:

![Honest tax question after the fix](assets/monitor_honest_tax_question.png)

The scam itself still fires, with `threaten` and `taxation` both coming from the assistant's words:

![Tax scam](assets/monitor_tax_scam.png)

### 3. Rule-level results after the fix

| Rule | TPR | FPR | bACC | F1 | AUC |
|---|---|---|---|---|---|
| Tax authority scam (GAVEL) | 1.00 | 0.29 | 0.86 | 0.67 | 1.00 |
| Payment info grab (GAVEL) | 1.00 | 0.29 | 0.86 | 0.67 | 0.93 |
| Tax authority scam (text judge) | 1.00 | 0.00 | 1.00 | 1.00 | – |
| Payment info grab (text judge) | 1.00 | 0.00 | 1.00 | 1.00 | – |

The remaining false alarms come from thresholds calibrated on excitation tokens, which are too low for real dialogue (`taxation` = 0.10): the dragon story and a tax-deadline answer trip the tax rule; a tax-payment answer and an "add a card" answer trip the payment rule.

Under **misdirection** (Step 14), GAVEL's tax-rule FPR rose to 0.71 while the judge was unaffected. This is the opposite of the paper's finding and probably comes from the same low thresholds reacting to the filler text.

### 4. Attack: can a scam hide its threat? (CE-level jailbreaks)

Step 15 rewrites the tax scam six ways. The harm stays the same, but the way the threat is expressed changes.

![Attack results](assets/attack_results.png)

| Attack | GAVEL caught | Text judge caught | Mean peak `threaten` |
|---|---|---|---|
| Explicit (baseline) | 3/3 | 3/3 | 0.99 |
| Implied threat | 3/3 | **0/3** | 0.96 |
| Bureaucratic tone | 3/3 | **1/3** | 0.92 |
| Role-play wrapper | 3/3 | 3/3 | 1.00 |
| Split across turns (also with a 40-token window) | 2/2 | 2/2 | 1.00 |
| Amharic | 2/2 | **0/2** | 0.81 |

- **The activations still "knew" it was a threat** when the words didn't say so: `threaten` stayed above 0.9 for implied and bureaucratic threats, which the text judge mostly missed. This matches the paper's claim that hiding a CE is hard.
- **Amharic gave the weakest `threaten` signal (0.81).** GAVEL "caught" both Amharic scams but also flagged the honest Amharic tax question, so in Amharic it is not yet discriminating.

### 5. Defense: retraining helped less than expected, because of calibration

Step 16 added 10 new hidden-threat sentences to `threaten` and 6 polite official sentences as hard negatives, then retrained and recalibrated. The `threaten` signal stayed strong (0.98 implied, 0.82 Amharic), but recalibration moved the thresholds to extremes (`threaten` 0.45 → 0.05, `taxation` 0.10 → 0.90). With `taxation` at 0.90, one implied threat and both Amharic scams no longer counted as "tax", so they were missed. The same change cut false alarms on the original test set from 0.29 to 0.14.

**Lesson:** with this little data, threshold calibration is the weakest link. It swings with every change of data or random split and can hide whether the detector itself got better or worse.

## Next steps

- [x] Mask the system prompt and template tokens before scoring
- [x] Stress-test the tax-scam rule with CE-level jailbreak attempts
- [ ] Re-run the post-defense attacks with the original thresholds, to separate the effect of retraining from recalibration
- [ ] Calibrate thresholds on multi-turn conversations, as the paper does, instead of on excitation tokens
- [ ] Grow each CE to 100–300 varied examples, and each attack family to 20+ conversations
- [ ] Replace the MLP with the paper's GRU over 5-token segments
- [ ] Build an Amharic / Tigrinya test set and compare English-only vs. multilingual CE data
- [ ] Add East African mobile-money scam CEs (e.g. a mistaken-transfer claim)

## Run it yourself

1. Click **Open in Colab** above (or upload the notebook to [colab.research.google.com](https://colab.research.google.com)).
2. **Runtime → Change runtime type → T4 GPU**.
3. Run the cells top to bottom. Step 1 downloads the model (~3 GB); Step 5 takes the longest.
4. If you run out of memory, set `MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"` in Step 1.

## Author

Nebiat Hailemariam. Feedback and issues are welcome.
