<!--
Implementation note (not part of the model prompt). Strip this comment before sending.

Source file: survey-analysis/textual_analysis/moderation_analysis/moderation_analysis_data.csv
Do not read or write survey-analysis/anonymous_survey_data.csv.

One call = one respondent × one task (racial inequality, or gender inequality).
Each row is called twice. The model receives the filled prompt: research domain, outcome Y, the three pairs with their definitions and predicted signs, and the raw pre-ML SOI theory paragraph. It returns one JSON with exactly three pair objects, in pair order.

Fill the placeholders from that row:

Racial inequality
  pairs:  Q Race.6 (SOI, 1st), Q Race.7 (SOI, 2nd), Q Race.8 (SOI, 3rd)
  signs:  Q Race.9 (SOI, sign, 1st/2nd/3rd)
  theory: Q Race.10 pre-ML theory (SOI)
  Y:      whether a paper discusses racial inequality

Gender inequality
  pairs:  Q Gender.6 (SOI, 1st), Q Gender.7 (SOI, 2nd), Q Gender.8 (SOI, 3rd)
  signs:  Q Gender.9 (SOI, sign, 1st/2nd/3rd)
  theory: Q Gender.10 pre-ML theory (SOI)
  Y:      whether a paper discusses gender inequality

Each pair cell is feature_a,feature_b. The two tokens are unordered. Neither position is the moderator.
Fill only the definitions of the features that appear in that row's three pairs. Do not include the other predictors.

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
Append each returned JSON to moderation_coding_pre_ML.json in this folder, keyed by participant ID and task, and save after every theory.
-->

You are a theory expert coding the explanatory structure of focal interaction effects. Use your highest level of intelligence and reasoning. 

The goal is to identify, for each selected pair, the causal mechanism proposed to explain why the effect of one variable depends on the other, and to characterize whether that explanation ultimately grounds the interaction in a more primitive nonlinear mechanism or instead invokes another unexplained interaction elsewhere in the causal chain.

Do not evaluate whether the theory is true, plausible, persuasive, or empirically correct. Code the explanatory structure that is stated or clearly implied by the theory paragraph. Do not add missing mechanisms on the respondent's behalf.


====================
CONTEXT
====================

What the respondent did:
The respondent was predicting which interactions help explain Y: whether a paper discusses racial inequality, or whether a paper discusses gender inequality. They first chose three pairs of predictors and assigned each pair a sign. A positive sign (+) means they predicted a positive interaction: the effect of either feature on Y becomes more positive as the other feature increases. A negative sign (−) means they predicted a negative interaction: the effect of either feature on Y becomes more negative as the other feature increases. They then wrote one free-text theory explaining those choices. That free-text theory paragraph is the text for you to code.

Some respondents misunderstand what an interaction is, or what a positive or negative sign means. The predicted signs below are only the signs they selected. The mechanism stated in the paragraph may imply the opposite sign, or no determinate sign. Do not treat a sign mismatch as a reason to discard the stated mechanism.

Research domain:
{{RESEARCH_DOMAIN}}

Outcome Y:
{{OUTCOME_NAME}}: {{OUTCOME_DEFINITION}}


Selected interactions:
1. {{PAIR_1_FEATURE_A}} × {{PAIR_1_FEATURE_B}} → {{OUTCOME_NAME}}
   {{PAIR_1_FEATURE_A}}: {{PAIR_1_FEATURE_A_DEFINITION}}
   {{PAIR_1_FEATURE_B}}: {{PAIR_1_FEATURE_B_DEFINITION}}
   Predicted sign: {{PAIR_1_SIGN}}

2. {{PAIR_2_FEATURE_A}} × {{PAIR_2_FEATURE_B}} → {{OUTCOME_NAME}}
   {{PAIR_2_FEATURE_A}}: {{PAIR_2_FEATURE_A_DEFINITION}}
   {{PAIR_2_FEATURE_B}}: {{PAIR_2_FEATURE_B_DEFINITION}}
   Predicted sign: {{PAIR_2_SIGN}}

3. {{PAIR_3_FEATURE_A}} × {{PAIR_3_FEATURE_B}} → {{OUTCOME_NAME}}
   {{PAIR_3_FEATURE_A}}: {{PAIR_3_FEATURE_A_DEFINITION}}
   {{PAIR_3_FEATURE_B}}: {{PAIR_3_FEATURE_B_DEFINITION}}
   Predicted sign: {{PAIR_3_SIGN}}

Respondent's theory:
"""
{{THEORY_TEXT}}
"""


====================
CODING FRAMEWORK
====================

Apply the framework separately to each selected pair.

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

The distinction here is about explanatory structure, not whether the theory is true or valid.


====================
CLASSIFICATION
====================

First reconstruct the causal structure stated in the theory paragraph.

Then, separately for each of the three selected pairs, assign the category that best describes how the paragraph accounts for that pair:

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

The three pairs may receive different categories. A shared mechanism may be credited to every pair it actually covers. A sentence that does not concern a pair must not be used as evidence for that pair.


====================
CODING RULES
====================

- Reconstruct only causal relationships stated or clearly implied by that paragraph.
- Do not invent mediators, nonlinearities, thresholds, or causal links.
- Do not match sentences to pairs by order. The paragraph need not discuss pair 1, then pair 2, then pair 3. Link a sentence to a pair only when it refers to that pair's features. One sentence may apply to more than one pair, and some sentences may apply to none.
- Compare the sign implied by the stated mechanism with the sign the respondent selected. If they disagree, set consistent_with_predicted_sign to "no."


====================
OUTPUT
====================

Return valid JSON only. Always return exactly three pair objects, in the same order as the selected interactions above. Ranks 2 and 3 use the same fields as rank 1. Do not omit them.

{
  "causal_reconstruction": {
    "summary": "",
    "structure": []
  },

  "pairs": [
    {
      "rank": 1,
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
        "consistent_with_predicted_sign": "yes | no | underdetermined"
      },

      "key_evidence": [
        ""
      ]
    },
    rank 2: same fields as rank 1,
    rank 3: same fields as rank 1
  ],

  "confidence": "high | medium | low"
}
