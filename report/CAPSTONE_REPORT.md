# Sentiment Classification of Kinyarwanda Social Media Text

**Author:** *[Your name]*  
**Course:** *[NLP capstone]*  
**Date:** October 2026

---

**Project links**

- **GitHub:** *[repository URL]*
- **Demo video:** *[7–10 min video URL]*
- **Live system:** *[Hugging Face Spaces URL, or “run locally via Streamlit”]*

---

## Abstract

We build a three-class sentiment classifier for Kinyarwanda short text using the AfriSenti Twitter corpus (Muhammad et al., 2023). We compare a majority baseline, TF-IDF with logistic regression, a bidirectional LSTM, and fine-tuned AfriBERTa on the official `kin` train/validation/test splits (~5.1k tweets). Macro-F1 on the held-out test set is reported for each experiment; error analysis highlights neutral–negative confusion and informal spelling. A Streamlit application exposes the best model for interactive prediction.

## 1. Introduction

Kinyarwanda is spoken by over 12 million people in Rwanda and neighboring regions. Social platforms carry growing volumes of user-generated Kinyarwanda text, but automated sentiment tools remain less available than for high-resource languages. Sentiment classification supports public-health campaigns, civic feedback, and market research when used responsibly.

**Objective:** Train and evaluate ML models that assign each Kinyarwanda tweet to *negative*, *neutral*, or *positive*, compare baselines and neural approaches, analyze errors, and deploy an interactive demo.

## 2. Related Work

AfriSenti provides 110k+ tweets in 14 African languages with native-speaker labels and was used in SemEval-2023 Task 12 (Muhammad et al., 2023). Prior work on Kinyarwanda NLP includes multilingual Transformers such as AfriBERTa (Ogueji et al., 2021) pretrained on African languages. Our project fine-tunes AfriBERTa on the Kinyarwanda subset only and compares it to classical and RNN baselines on the same splits.

## 3. Dataset and Data Preparation

**Source:** AfriSenti `kin` configuration ([Hugging Face](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment), [GitHub splits](https://github.com/afrisenti-semeval/afrisent-semeval-2023/tree/main/data/kin)).

| Split | Examples |
|-------|----------|
| Train | 3,302 |
| Validation | 827 |
| Test | 1,034 |

**Fields:** `tweet`, `label` ∈ {negative, neutral, positive}.

**Preprocessing:** Strip whitespace; replace URLs with `<URL>`; preserve `@user` placeholders as in the corpus. We do not machine-translate or synthetically generate training tweets.

**Limitations:** Twitter domain; possible label noise; class imbalance; code-switching and informal spelling.

## 4. Methodology

### 4.1 Baselines

- **E0 — Majority class:** Predict the most frequent training label.
- **E1 — TF-IDF (word, 1–2 grams) + balanced logistic regression.**
- **E2 — TF-IDF (character, 3–5 grams) + logistic regression** (morphology/spelling variations).

### 4.2 BiLSTM (E3)

Trainable embeddings (128-d), single-layer bidirectional LSTM (128 hidden), max-pooling, dropout 0.3, class-weighted cross-entropy, max sequence length 64.

### 4.3 Transformer (E4, E5)

Fine-tune `castorini/afriberta_base` with AdamW (lr 2e-5), max length 128, 3 epochs; **E5** adds balanced class weights in the loss.

**Metrics:** Accuracy, macro-averaged precision/recall/F1, per-class F1, confusion matrices. Macro-F1 is primary because classes are imbalanced.

## 5. Experiments and Results

*Fill this table after running training (from `results/experiments.csv`):*

| ID | Model | Key change | Macro-F1 (val) | Macro-F1 (test) |
|----|--------|------------|----------------|-----------------|
| E0 | Majority | always `neutral` | 0.184 | 0.185 |
| E1 | TF-IDF + LR | word 1–2 grams | 0.585 | 0.568 |
| E2 | TF-IDF + LR | char 3–5 grams | 0.629 | 0.597 |
| E3 | BiLSTM | embed/hidden 128 | 0.537 | 0.527 |
| E4 | AfriBERTa | lr=2e-5, 2 epochs (CPU) | 0.640 | 0.627 |
| E5 | AfriBERTa + class weights | *run locally* | — | — |

**Takeaways:** Character n-grams beat word n-grams on noisy Twitter Kinyarwanda. The BiLSTM underperformed TF-IDF, likely due to limited data and a small vocabulary. Fine-tuned `castorini/afriberta_base` gave the best macro-F1 on the official test split; re-run with 3+ epochs and optional class weights (E5) on GPU for your final submission numbers.

## 6. Error Analysis and Discussion

See `results/error_analysis.md`. Common patterns to discuss:

- Neutral vs negative boundary (understated criticism).
- Positive phrases with negation.
- Rare tokens / spelling variants (char TF-IDF helps).
- Short tweets with little lexical signal.

Include 2–3 success and 2–3 failure examples from the demo in your video.

## 7. Deployment

Streamlit app (`app/streamlit_app.py`) loads `models/afriberta-best` via `src/predict.py`. Users enter Kinyarwanda text and receive a label with class probabilities.

## 8. Limitations and Future Work

- Not for medical, legal, or electoral decision-making.
- Domain limited to Twitter; drift on news or WhatsApp text.
- More data, Kinyarwanda-specific pretraining, or human-in-the-loop review could improve rare classes.

## 9. Conclusion

We implemented a full ML pipeline for Kinyarwanda sentiment: official AfriSenti splits, baselines, RNN and Transformer models, quantitative evaluation, error analysis, and a web demo. On our test run, fine-tuned AfriBERTa reached **macro-F1 ≈ 0.63**, improving over the strongest TF-IDF baseline (≈ 0.60) and majority class (≈ 0.18).

## 10. References

- Muhammad, S. H., et al. (2023). AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages. *EMNLP 2023*.
- SemEval-2023 Task 12: AfriSenti-SemEval. https://github.com/afrisenti-semeval/afrisent-semeval-2023
- Ogueji, J., et al. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. https://huggingface.co/castorini/afriberta_base
- Pedregosa, F., et al. (2011). Scikit-learn. *JMLR*.
- Wolf, T., et al. (2020). Transformers. *EMNLP System Demonstrations*.
