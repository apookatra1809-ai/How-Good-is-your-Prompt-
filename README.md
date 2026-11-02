# How Good is your Prompt?

### A multidimensional framework for evaluating prompt driven LLM response quality

MSc Human Centred Artificial Intelligence thesis, Technological University Dublin (TU Dublin)
Industry partner: CeADAR | Supervisor: Dr Keith Nolan


## Overview

Large Language Models are highly sensitive to how a prompt is written, yet "prompt quality" is rarely defined or measured in a way that a person can interpret. Most evaluations collapse quality into one overall score, which hides cases where a response improves on one property while getting worse on another.

This project introduces the **Rubric Quality Matrix (RQM)**, a seven dimension framework that scores every response on distinct, interpretable properties using fixed and published computational metrics. No LLM is used as a judge, so every score can be inspected, reproduced and explained.

The framework is applied to **1,024 real user prompts** from WildBench v2. Each prompt is paired with a deliberately degraded version, and both versions are answered by three open weight models from different families. The result is a controlled study of how prompt quality shapes response quality, and of whether that relationship holds across tasks and models.

## Key findings

1. **Response quality is multidimensional.** The seven dimensions are related but not redundant. No pair reached the redundancy threshold (strongest Spearman correlation: 0.550), and PCA showed no dominant first component.
2. **Original prompts produced stronger responses on three dimensions:** Format Compliance, Clarity and Specificity, and Response Richness. This held across all three models.
3. **Vague prompts produced higher Coherence and Readability.** Response length is a candidate explanation, although no mediation analysis was run to confirm it.
4. **The effect of prompt quality depends on the model.** Task Alignment showed the sharpest divergence: Phi favoured original prompts strongly, while Mistral and Qwen favoured vague ones. Five of seven dimensions showed a significant condition by model interaction.
5. **There is no universal rule for "better prompts".** The effect depends on which aspect of quality matters, which model is used and, in some cases, which task category is involved. A prompting strategy validated on one model should not be assumed to transfer to another.

## The Rubric Quality Matrix

| # | Dimension | Question it answers | Metric |
|---|-----------|---------------------|--------|
| 1 | Task Alignment | Did the response answer what was asked? | BERTScore F1 |
| 2 | Response Richness | How detailed and informative is it? | ROUGE L |
| 3 | Readability | How easy is it to read? | Flesch Reading Ease |
| 4 | Lexical Diversity | How varied is the vocabulary? | Type Token Ratio |
| 5 | Format Compliance | Did it follow the requested format? | Rule based pattern detection |
| 6 | Clarity and Specificity | Is it concrete rather than generic? | Lexical Density |
| 7 | Coherence | Is it logically organised and connected? | Discourse based coherence measure |

Raw metric values sit on different scales, so each dimension is converted to a common **1 to 5 score** using pooled quintile based scoring across all valid responses.

## Experimental design

```
WildBench v2 (1,024 prompts, 11 task categories)
        |
        v
Zephyr 7B rewrites each prompt to be shorter and less specific
        |
        v
2,048 prompt instances (1,024 original + 1,024 degraded, paired)
        |
        v
Qwen, Mistral and Phi generate responses (temperature 0.7, 512 token limit)
        |
        v
6,144 expected responses, 6,118 valid after removing 26 empty outputs
        |
        v
RQM scoring on seven dimensions, standardised to a 1 to 5 scale
        |
        v
Statistical analysis (correlation, PCA, Wilcoxon, Kruskal Wallis, OLS, interactions)
```

**Dataset summary**

| Property | Value |
|----------|-------|
| Original prompts | 1,024 |
| Degraded prompts | 1,024 |
| Total prompt instances | 2,048 |
| Task categories | 11 |
| Response generation models | 3 (Qwen, Mistral, Phi) |
| Expected responses | 6,144 |
| Empty responses removed | 26 (0.4%) |
| Final valid responses | 6,118 |

**Task categories:** Advice Seeking, Brainstorming, Coding and Debugging, Creative Writing, Data Analysis, Editing, Information Seeking, Math, Planning, Reasoning and Role Playing.

**Design decisions worth noting**

* Zephyr 7B was used only to degrade prompts and played no part in response generation, keeping the transformation automated and free of researcher bias.
* All models were used as pretrained systems with no fine tuning and one fixed generation configuration, so prompt quality is the main variable under study.
* Original and degraded prompts are kept as paired observations, which isolates the effect of prompt quality from the effect of the task.
* Effect sizes, not p values, are the primary basis for interpretation, because a sample of this size makes very small effects statistically significant.

## Results at a glance

Paired Wilcoxon signed rank tests comparing Original and Vague prompts, pooled across models. All seven comparisons were statistically significant, but effect sizes ranged from negligible to moderate.

| RQM dimension | Original (mean) | Vague (mean) | Effect size (r) | Favours |
|---------------|-----------------|--------------|-----------------|---------|
| Format Compliance | 3.721 | 3.632 | 0.352 | Original |
| Clarity and Specificity | 3.196 | 2.793 | 0.299 | Original |
| Response Richness | 3.210 | 2.808 | 0.285 | Original |
| Coherence | 2.840 | 3.159 | 0.218 | Vague |
| Task Alignment | 3.059 | 2.941 | 0.092 | Original |
| Readability | 2.956 | 3.044 | 0.075 | Vague |
| Lexical Diversity | 3.033 | 2.950 | 0.070 | Original |

**Model dependence of Task Alignment** (Spearman correlation between condition and score):

| Model | Correlation |
|-------|-------------|
| Phi | 0.748 (favours Original) |
| Mistral | 0.205 (favours Vague) |
| Qwen | 0.108 (favours Vague) |

**Other results**

* Task category had significant but generally small effects. Lexical Diversity was the exception (epsilon squared 0.2213).
* Prompt length was not a broad driver of response quality.
* Prompts with an explicit output format scored lower on Format Compliance (mean 2.868) than prompts without one (mean 4.000). This gap is likely inflated by how the metric is scored, since a prompt with no format requirement has nothing to fail.
* After controlling for model, moving from Vague to Original prompts was associated with a regression coefficient of 0.404 for Clarity and Specificity, 0.403 for Response Richness and a negative coefficient of 0.319 for Coherence.

## Research questions

1. **RQ1:** How do the selected response quality dimensions relate to each other when evaluating LLM generated responses?
2. **RQ2:** How does prompt degradation affect each response quality dimension, and does this effect vary across task categories?
3. **RQ3:** To what extent does the effect of prompt quality vary across different LLMs and model families?

## Statistical methods

* Spearman rank correlation for relationships between dimensions and prompt characteristics
* Kaiser Meyer Olkin measure and Bartlett's test of sphericity, followed by Principal Component Analysis
* Wilcoxon signed rank test for paired Original and Vague comparisons
* Kruskal Wallis test for differences across task categories
* Mann Whitney U test for Format Compliance with and without explicit format requirements
* OLS regression with a condition by model interaction term
* Effect sizes reported alongside significance throughout

## Repository structure

Adjust this section to match the folders in your repository.

```
.
├── data/          Prompts, degraded prompts and generated responses
├── notebooks/     Exploration, scoring and statistical analysis
├── src/           Degradation, response generation and RQM scoring code
├── results/       Scored datasets, tables and figures
├── thesis/        Final thesis document
└── README.md
```

## Reproducing the study

1. Obtain the WildBench v2 prompts from the allenai/WildBench dataset on Hugging Face.
2. Generate a degraded version of each prompt with Zephyr 7B, instructing it to make the prompt shorter and less specific while keeping the task unchanged.
3. Generate responses from Qwen, Mistral and Phi using the same configuration (temperature 0.7, maximum 512 tokens) for every prompt.
4. Remove empty responses, since BERTScore based metrics are undefined for them.
5. Compute the seven RQM metrics for every response and convert each to a 1 to 5 score using pooled quintiles.
6. Run the statistical analysis in the notebooks to reproduce the tables and figures.

## Limitations

* Prompt quality is inferred from responses under one specific degradation style (Zephyr 7B), and degraded prompts were not manually checked for preserved task intent.
* BERTScore and ROUGE L were designed for reference based comparison, so their use for prompt to response comparison is a reinterpretation of their original purpose.
* No correction for multiple comparisons was applied. Effect sizes were prioritised to reduce the impact of this.
* Only three open weight models were tested, so results may not generalise to larger or closed source systems.
* The RQM does not measure factual accuracy or hallucination. A response can score well on every dimension and still be wrong.

## Future work

* Add closed source and frontier models to test whether the model dependent patterns generalise.
* Replace the binary Original and Vague contrast with a continuous measure of prompt quality.
* Test other degradation strategies, such as ambiguity, missing context, conflicting instructions and excessive constraint.
* Validate the RQM dimensions against human judgement on a subsample.
* Test alternative metrics for Task Alignment and Format Compliance to separate real model behaviour from measurement effects.
* Add a hallucination or factual accuracy dimension.

## Human centred contribution

The seven dimensions were chosen to reflect what a person reading an AI response actually notices: whether it answered the question, whether it was understandable, whether it was detailed enough and whether it followed the requested format. Reporting these separately makes it clear where a response succeeded or fell short, and makes the evaluation explainable to users as well as researchers.

## Citation

```
Katra, A. (2026). How Good is your Prompt? MSc Thesis, Human Centred AI,
Technological University Dublin.
```

## Author

**Aparna Katra**
MSc Human Centred Artificial Intelligence, TU Dublin

* GitHub: github.com/apookatra1809-ai
* Portfolio: aparnakatra.lovable.app

## Acknowledgements

Thanks to Dr Keith Nolan for supervision and guidance, to CeADAR as industry partner, and to TU Dublin and the MSc Human Centred AI programme.

