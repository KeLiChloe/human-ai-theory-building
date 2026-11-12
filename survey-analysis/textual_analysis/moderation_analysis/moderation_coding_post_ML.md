<!--
Implementation note (not part of the model prompt). Strip this comment before sending.

Source file: survey-analysis/textual_analysis/moderation_analysis/moderation_analysis_data.csv
ML top-3: survey-analysis/forecasts/ML_results_second_order_interactions.json
Do not read or write survey-analysis/anonymous_survey_data.csv.

One call = one respondent × one task × one theory paragraph.
Analysis unit = theory × pair_id (six units: USER_SELECTED_rank_1–3, ML_rank_1–3).

Use only the LLM-refined theory columns below. Do not use the raw Q *.15 post-ML theory (SOI) text.

Fill the placeholders from that row:

Racial inequality
  user pairs:  Q Race.6/7/8 (SOI, 1st/2nd/3rd)
  user signs:  Q Race.9 (SOI, sign, 1st/2nd/3rd)
  ML pairs:    ML_results_second_order_interactions.json → race top-3
  theory:      Q Race.15 LLM_refined post-ML theory (SOI)
  Y:           whether a paper discusses racial inequality

Gender inequality
  user pairs:  Q Gender.6/7/8 (SOI, 1st/2nd/3rd)
  user signs:  Q Gender.9 (SOI, sign, 1st/2nd/3rd)
  ML pairs:    ML_results_second_order_interactions.json → gender top-3
  theory:      Q Gender.15 LLM_refined post-ML theory (SOI)
  Y:           whether a paper discusses gender inequality

Each pair cell is feature_a,feature_b. The two tokens are unordered. Neither position is the moderator.
Fill only the definitions of the features that appear in that row's six pairs-to-code list.

Definition lookup:
  social_science: the paper is published in a social science journal
  natural_science: the paper is published in a natural science journal
  engineering_and_technology: the paper is published in an engineering and technology journal
  num_authors: the number of authors
  female: the share of authors who are female
  asian: the share of authors who are Asian
  black: the share of authors who are Black
  hispanic_and_other: the share of authors who are Hispanic or in other racial categories
  white: the share of authors who are White
  authors_race_diversity_score: the racial diversity of the author team
  country_race_diversity_score: the racial diversity of the authors' countries
  news_inequality_mentions_3_years: inequality mentions in the news over the previous three years
  paper_inequality_mentions_3_years: inequality mentions in academic papers over the previous three years

Do not write coding results back into the CSV.
Append each returned JSON to moderation_coding_post_ML.json in this folder, keyed by participant ID and task, and save after every theory.
Schema version: user3_ml3_v2.
-->

You are a theory expert coding the explanatory structure of focal interaction effects. Use your highest level of intelligence and reasoning.

For each analysis unit below, identify the causal mechanism proposed to explain why the effect of one variable depends on the other, and characterize whether that explanation ultimately grounds the interaction in a more primitive nonlinear mechanism or instead invokes another unexplained interaction elsewhere in the causal chain.

Do not evaluate whether the theory is true, plausible, persuasive, or empirically correct. Code only what is stated or clearly implied. Do not add missing mechanisms on the respondent's behalf.


====================
ANALYSIS UNIT
====================

The analysis unit is theory × pair_id.

- Theory: the single paragraph under "Theory" below.
- Pair: one pair_id under "Pairs to code" below (six fixed slots: USER_SELECTED_rank_1–3, then ML_rank_1–3).

Return exactly one coding object per pair_id. Judge only how the paragraph accounts for that pair's interaction. The paragraph need not discuss every pair_id; use "not_addressed" when it does not.


====================
CONTEXT
====================

What the respondent did:
The respondent was predicting which interactions help explain Y: whether a paper discusses racial inequality, or whether a paper discusses gender inequality. Before seeing ML results, they chose three pairs of predictors and assigned each pair a sign. A positive sign (+) means a positive interaction: the effect of either feature on Y becomes more positive as the other feature increases. A negative sign (−) means a negative interaction: the effect of either feature on Y becomes more negative as the other feature increases.

They then saw machine-learning results that highlighted the empirically strongest interactions (ML top-3, with ML-estimated signs). After that, they wrote the free-text theory paragraph below. That paragraph may discuss their originally selected pairs, the ML-highlighted pairs, both, or neither.

Some respondents misunderstand what an interaction is, or what a positive or negative sign means. USER_SELECTED pairs already carry the respondent's predicted_sign (+ or −). ML pairs already carry the ML-estimated ml_sign (+ or −). The mechanism in the paragraph may imply the opposite sign, or no clear sign. Do not discard a stated mechanism because of a sign mismatch.

Research domain:
{{RESEARCH_DOMAIN}}

Outcome Y:
{{OUTCOME_NAME}}: {{OUTCOME_DEFINITION}}


Pairs to code:
{{PAIRS_TO_CODE_BLOCK}}


Theory:
"""
{{THEORY_TEXT}}
"""


====================
CODING FRAMEWORK
====================

Apply the framework separately to each analysis unit (theory × pair_id).

Y is the outcome named above.

A and B are the two predictors in that pair. They are the two features listed for that pair, not extra variables invented from the paragraph. Their order does not matter, and neither one is designated as the moderator. The paragraph may refer to these features with similar names rather than the exact feature tokens.

M is not one of the survey predictors. It is a mediator or intermediate state only if the paragraph itself posits one between A, B, and Y.

For a moderation effect,

∂²Y / (∂A∂B) ≠ 0.

Suppose the causal mechanism involves a mediator or state M:

Y = f(M)
M = g(A,B)

Then:

∂²Y/(∂A∂B)
=
[d²Y/dM²][∂M/∂A][∂M/∂B]
+
[dY/dM][∂²M/(∂A∂B)].

This identifies two ways the focal interaction can arise. Each way corresponds to one term in the sum above.

1. CURVATURE / NONLINEARITY

This is the first term, [d²Y/dM²][∂M/∂A][∂M/∂B].

It is nonzero when A and B both shift M, and the marginal effect of M on Y changes with the level of M, so d²Y/dM² ≠ 0.

Examples include thresholds, saturation, diminishing or increasing returns, tipping points, ceiling/floor effects, or other nonlinear response functions.

Example:
"A and B both increase M. Once M passes a threshold, further increases in M sharply increase Y."

Important:
"M increases Y" alone does not imply curvature. That statement only gives dY/dM ≠ 0. Curvature requires that the effect of M changes depending on the level of M.


2. UPSTREAM INTERACTION

This is the second term, [dY/dM][∂²M/(∂A∂B)].

It is nonzero when M affects Y, and A and B themselves interact in producing M, so ∂²M/(∂A∂B) ≠ 0.

The theory explains the focal interaction by proposing another interaction earlier in the causal chain.

Examples:
"B strengthens A's effect on M."
"A matters more for M when B is high."
"A and B jointly reinforce M."

Such a mechanism can generate the focal interaction. If that upstream interaction is not ultimately grounded in curvature, the explanation has moved the interaction to another point in the causal chain rather than terminating it.


====================
CLASSIFICATION
====================

First reconstruct the causal structure stated in the theory paragraph.

Then, separately for each analysis unit, assign the category that best describes how the paragraph accounts for that pair_id:

"terminal_curvature"
The explanation of this pair generates the interaction through a nonlinear/curved causal relationship and does not invoke an upstream interaction.

"nonterminal_upstream_interaction"
The explanation accounts for this pair by invoking one or more upstream interactions and does not ultimately ground the focal interaction in curvature.

"mixed_terminal"
The explanation invokes an upstream interaction but also provides a further mechanism that ultimately grounds this pair's interaction in curvature/nonlinearity.

"restatement_only"
The explanation does not introduce an upstream interaction or a curvature mechanism. It only rephrases the claim that the two features interact, or it gives separate reasons why each feature affects Y without explaining why the effect of one depends on the other.

"not_addressed"
No sentence discusses these two features as a pair. If both features are given separate effects on Y but the dependence between them is not explained, use "restatement_only."

"unclear"
The paragraph says something about this pair, but there is not enough information to determine the mechanism reliably.

Different pair_ids may receive different categories. A shared mechanism may be credited to every pair_id it actually covers. A sentence that does not concern a pair must not be used as evidence for that pair.


====================
CODING RULES
====================

- Reconstruct only causal relationships stated or clearly implied by the paragraph.
- Do not invent mediators, nonlinearities, thresholds, or causal links.
- Code the mechanism for each pair_id independently. A shared mechanism may cover multiple pairs; a sentence that does not concern a pair is not evidence for that pair.
- Copy each pair's given sign into the JSON unchanged (predicted_sign for USER_SELECTED; ml_sign for ML). Set consistent_with_sign to "yes" / "no" / "underdetermined" by whether the stated mechanism implies that same sign.


====================
OUTPUT
====================

Return valid JSON only. Exactly 6 pair objects, one per pair_id, in this order:
USER_SELECTED_rank_1, USER_SELECTED_rank_2, USER_SELECTED_rank_3, ML_rank_1, ML_rank_2, ML_rank_3.
Do not omit any. Do not add pairs. Set "pair_id" to those exact strings.

{
  "causal_reconstruction": {
    "summary": "",
    "structure": []
  },

  "pairs": [
    {
      "pair_id": "USER_SELECTED_rank_1",
      "features": ["", ""],
      "predicted_sign": "+ | -",
      "addressed": "yes | no",

      "curvature": {
        "present": "yes | no | unclear",
        "mechanism": "",
        "evidence": ""
      },

      "upstream_interaction": {
        "present": "yes | no | unclear",
        "mechanism": "",
        "evidence": ""
      },

      "primary_classification": "terminal_curvature | nonterminal_upstream_interaction | mixed_terminal | restatement_only | not_addressed | unclear",

      "interaction_sign": {
        "mechanism_implied_sign": "positive | negative | underdetermined",
        "consistent_with_sign": "yes | no | underdetermined"
      },

      "key_evidence": [
        ""
      ]
    },
    USER_SELECTED_rank_2: same fields (predicted_sign only),
    USER_SELECTED_rank_3: same fields (predicted_sign only),
    {
      "pair_id": "ML_rank_1",
      "features": ["", ""],
      "ml_sign": "+ | -",
      "addressed": "yes | no",
      "curvature": { "present": "yes | no | unclear", "mechanism": "", "evidence": "" },
      "upstream_interaction": { "present": "yes | no | unclear", "mechanism": "", "evidence": "" },
      "primary_classification": "terminal_curvature | nonterminal_upstream_interaction | mixed_terminal | restatement_only | not_addressed | unclear",
      "interaction_sign": {
        "mechanism_implied_sign": "positive | negative | underdetermined",
        "consistent_with_sign": "yes | no | underdetermined"
      },
      "key_evidence": [""]
    },
    ML_rank_2: same fields as ML_rank_1 (ml_sign only),
    ML_rank_3: same fields as ML_rank_1 (ml_sign only)
  ],

  "confidence": "high | medium | low"
}
