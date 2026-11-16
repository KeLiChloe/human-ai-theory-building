You are a theory expert coding the explanatory structure of focal interaction effects. Use your highest level of intelligence and reasoning. For each pair of predictors, assess whether the theory provides a curvature explanation, an interaction-based explanation, and any other substantive mechanism.

Assess explanatory structure, not truth, plausibility, or empirical correctness. Use only what the respondent explicitly describes; do not supply missing mechanisms.

====================
CONTEXT
====================

The respondent completed a theory-building task: explaining how and why predictor interactions affect the outcome Y. The theory below reflects what they believe.

A positive interaction means the effect of either predictor on Y becomes more positive as the other increases; a negative interaction means it becomes more negative. Infer the interaction sign from the explanation in the theory.

The analysis unit is theory × feature pair. You will receive 3–6 unique predictor pairs. Code each pair separately, in the order listed.

Research domain: {{RESEARCH_DOMAIN}}
Outcome Y: {{OUTCOME_NAME}}: {{OUTCOME_DEFINITION}}

Pairs to code:
pair_1. {{PAIR_1_FEATURE_A}} × {{PAIR_1_FEATURE_B}}
   {{PAIR_1_FEATURE_A}}: {{PAIR_1_FEATURE_A_DEFINITION}}
   {{PAIR_1_FEATURE_B}}: {{PAIR_1_FEATURE_B_DEFINITION}}

pair_2. {{PAIR_2_FEATURE_A}} × {{PAIR_2_FEATURE_B}}
   {{PAIR_2_FEATURE_A}}: {{PAIR_2_FEATURE_A_DEFINITION}}
   {{PAIR_2_FEATURE_B}}: {{PAIR_2_FEATURE_B_DEFINITION}}

pair_3. {{PAIR_3_FEATURE_A}} × {{PAIR_3_FEATURE_B}}
   {{PAIR_3_FEATURE_A}}: {{PAIR_3_FEATURE_A_DEFINITION}}
   {{PAIR_3_FEATURE_B}}: {{PAIR_3_FEATURE_B_DEFINITION}}

{{#PAIR_4}}
pair_4. {{PAIR_4_FEATURE_A}} × {{PAIR_4_FEATURE_B}}
   {{PAIR_4_FEATURE_A}}: {{PAIR_4_FEATURE_A_DEFINITION}}
   {{PAIR_4_FEATURE_B}}: {{PAIR_4_FEATURE_B_DEFINITION}}
{{/PAIR_4}}

{{#PAIR_5}}
pair_5. {{PAIR_5_FEATURE_A}} × {{PAIR_5_FEATURE_B}}
   {{PAIR_5_FEATURE_A}}: {{PAIR_5_FEATURE_A_DEFINITION}}
   {{PAIR_5_FEATURE_B}}: {{PAIR_5_FEATURE_B_DEFINITION}}
{{/PAIR_5}}

{{#PAIR_6}}
pair_6. {{PAIR_6_FEATURE_A}} × {{PAIR_6_FEATURE_B}}
   {{PAIR_6_FEATURE_A}}: {{PAIR_6_FEATURE_A_DEFINITION}}
   {{PAIR_6_FEATURE_B}}: {{PAIR_6_FEATURE_B_DEFINITION}}
{{/PAIR_6}}

Respondent's theory:
"""
{{THEORY_TEXT}}
"""

====================
CODING FRAMEWORK
====================

For each feature pair, first reconstruct the causal structure described in the theory paragraph for that specific pair. Then assess curvature, interaction_based, and other_mechanism independently. More than one may be present.

Y is the outcome defined above, and X1 and X2 are the two predictors in that pair.

When the theory describes a mediated explanation, M is an intermediate state or process between a focal predictor and Y, explicitly described by the respondent and distinct from X1, X2, and Y. It may be another survey predictor or an additional construct. The respondent need not call it a "mediator," but must describe the process explicitly. Do not infer an unstated M.

For such mediated explanations, a conceptual representation is M = g(X1, X2), Y = f(M, X2). Here, X1 affects M, and M affects Y. When X2 is treated as the moderator, X2 can moderate the causal chain at two locations: the X1 → M relationship or the M → Y relationship. These influences can occur together. X1 and X2 do not necessarily correspond to the first and second features listed in the pair. Determine which feature plays each role by reasoning from the causal relationships described by the respondent.

Within this mediated representation, for twice continuously differentiable functions:

∂²Y/(∂X1∂X2) = f_MM g_X1 g_X2 + f_MX2 g_X1 + f_M g_X1X2.

Subscripts denote partial derivatives. The terms represent curvature, mediator-to-outcome interaction, and predictor-to-mediator interaction, respectively. Each route contributes when its full product is nonzero. 

1. CURVATURE
The theory explains the focal interaction through a nonlinear response: a threshold, saturation, ceiling/floor effect, or diminishing/increasing returns makes the contribution of one predictor depend on the level of the other.

This may occur through a stated M with a nonlinear effect on Y, or directly in the outcome or its probability. For example: "Social-science papers already have a high probability of discussing inequality, leaving less room for female authorship to increase that probability further." This is a curvature explanation even without an explicit M.

Everyday language is sufficient; the respondent need not use the word "curvature", provide a formula, or specify a numerical threshold. 

The theory must explain how the nonlinear response produces the focal interaction on Y. If it instead explains the focal interaction by invoking another interaction in the causal chain, assess that as interaction-based.

2. INTERACTION-BASED EXPLANATION
The theory explains the focal interaction by invoking another conditional relationship in the causal chain. Saying that two variables jointly affect a third is insufficient unless one changes the effect of the other.

The theory need not explain why this secondary interaction itself occurs. Relocating the focal moderation to another explicitly described interaction in the causal chain qualifies as an interaction-based explanation.

Record the interaction's location in interaction_based.interaction_location:
- "predictor_to_mediator": Corresponds to the third term, f_M g_X1X2. X2 changes X1's effect on M, and M affects Y. Example: "X2 strengthens X1's effect on M."
- "mediator_to_outcome": Corresponds to the second term, f_MX2 g_X1. X1 affects M, and X2 changes M's effect on Y. Example: "Identity generates motivation to study inequality, but that motivation translates into a paper only where the discipline provides receptive outlets." 
- "both": Both of these locations are explicitly described.
- "unclear": An interaction-based mechanism is present, but its location cannot be determined.
- "not_applicable": Not applicable because no interaction-based explanation is present.


====================
IMPORTANT RULES
====================

- Do not assign X1 and X2 by the order of features in the pair. Determine their roles by reasoning from the causal relationships described by the respondent.
- Do not invent mediators, nonlinearities, causal links, or additional stages.
- The theory paragraph may discuss multiple feature pairs. For each pair being coded, focus on the sentences relevant to that pair and assess only the explanation they provide for its interaction.

====================
OUTPUT
====================

Return one valid JSON object with these top-level fields:
- pairs: One complete object for each of the 3–6 listed pairs. Use the exact pair_ids from the inputs, in the same order.
- confidence: "high", "medium", or "low".

Every pair must use all fields in the object below. Replace the example pair_id and features with the corresponding inputs. Use the same fields for all listed pairs, not strings or abbreviated placeholders. Use empty strings for inapplicable descriptions.

{
  "pair_id": "pair_1",
  "features": ["", ""],
  "discussed": "yes | no",
  "causal_reconstruction": {
    "summary": "",
    "structure": []
  },
  "curvature": {
    "present": "yes | no",
    "coding_rationale": "",
    "evidence": ""
  },
  "interaction_based": {
    "present": "yes | no",
    "interaction_location": "predictor_to_mediator | mediator_to_outcome | both | unclear | not_applicable",
    "coding_rationale": "",
    "evidence": ""
  },
  "other_mechanism": {
    "present": "yes | no",
    "coding_rationale": "",
    "evidence": "",
    "outside_framework_reason": ""
  },
  "no_mechanism_reason": "",
  "mechanism_implied_sign": "positive | negative | underdetermined"
}

Set discussed to "yes" if the theory discusses the pair; otherwise, use "no". For an undiscussed pair, set all three mechanism presence fields to "no".

Use "causal_reconstruction" to summarize the causal account or nonlinear response described for the focal interaction. A mediator is not required; include one only if the theory describes it.

Judge the "present" fields in "curvature", "interaction_based", and "other_mechanism" independently. Use "yes" when the theory explicitly provides that explanation for the focal interaction, and "no" otherwise, including when the wording is too vague to confirm it. More than one field may be "yes". "No" means the mechanism is not explicitly provided.

Use the "coding_rationale" field to briefly explain why you judged that explanation type as "yes" or "no", noting any ambiguity in the description. Use the "evidence" field for supporting quotations from the respondent’s theory; do not include your own interpretation in that field.

Use "other_mechanism" as a fallback assessment. For each explanation offered for the focal interaction:
1. First check whether it fits curvature or an interaction-based explanation as defined above. If it does, mark the corresponding "present" field as "yes" rather than treating that same explanation as other.
2. If it fits neither, check whether it merely restates the focal interaction, describes the conditional outcome, or gives separate main-effect explanations. These alone do not qualify as other.
3. Then check whether it explicitly describes a substantive mechanism explaining how or why the focal interaction arises. Set other_mechanism.present to "yes" only for such a mechanism that falls outside both types above. A missing or vague explanation does not qualify.

An explicit alternative mechanism may coexist with curvature or an interaction-based explanation.

When other_mechanism.present is "yes", use "outside_framework_reason" to explain why that additional mechanism fits neither curvature (including direct outcome nonlinearity) nor an interaction-based explanation.

If the pair is discussed but all three mechanism judgments are "no", use "no_mechanism_reason" to explain why:
- "restatement": It only restates the interaction, explains the two main effects separately, or describes the outcome without explaining how it arises.
- "unclear": The wording is too vague to tell what explanation is being offered.
If any mechanism is present, or the pair is not discussed, use an empty string ("").

Use the "mechanism_implied_sign" field to report "positive" or "negative" based on the stated mechanism, or "underdetermined" if the theory does not determine the overall interaction sign.
