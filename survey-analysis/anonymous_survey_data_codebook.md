# Codebook: `anonymous_survey_data.csv`

The survey has two parallel tasks—**racial inequality** (`Q Race.`) and **gender inequality** *(*`Q Gender.`)—with the same structure. Question IDs in this file (e.g. `Q Race.1`, `Q Gender.12`) match the items in the paper appendix Survey contents.

One row = one survey respondent (human or GenAI system).  
**98 rows × 118 columns.** Identifying fields have been removed; human names are replaced by anonymous `Participant ID` codes.


| Source                   | Meaning                                                                          |
| ------------------------ | -------------------------------------------------------------------------------- |
| **Raw**                  | Native survey response.                                                          |
| **Coded by author team** | Assigned by the author team (e.g. topic expertise; anonymized `Participant ID`). |
| **Coded by LLM**         | Produced by LLM post-processing.                                                 |


---



## 1. Identifiers and group codes


| Column                         | Meaning                                                                                                                                                | Source               |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------- |
| `student_0, senior_1, genAI_2` | Respondent type. `0` = PhD student; `1` = senior scientist; `2` = GenAI model.                                                                         | Raw                  |
| `Participant ID`               | Anonymous ID. Humans: `phd_contributor_<code>` or `senior_contributor_<code>`. GenAI: model display name (e.g. GPT / Claude / Theorista style labels). | Coded by author team |
| `gender`                       | Coded self-reported gender. `Female` / `Male` / `Other` (non-binary or prefer not to disclose) / `-1` = N/A (GenAI). Used for moderator analyses.     | Coded by author team |
| `topic_expert`                 | Human topic expertise flag. `1` = topic expert; `0` = human non-expert; `-1` = N/A (GenAI).                                                            | Coded by author team |


---



## 2. Shared feature vocabulary (predictors)

Respondents choose among the same 13 paper-level predictors used in the ML analyses. Feature tokens appear in selection, rank, and sign columns:

`social_science`, `natural_science`, `engineering_and_technology`, `num_authors`, `female`, `asian`, `black`, `hispanic_and_other`, `white`, `authors_race_diversity_score`, `country_race_diversity_score`, `news_inequality_mentions_3_years`, `paper_inequality_mentions_3_years`

Empty cells mean the respondent did not select that feature for that question.

---



## 3. Racial inequality block (`Q Race.*`)



### 3.1 Main-effect forecasts (before ML)


| Column                        | Meaning                                                                                                                                           | Source |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| `Q Race.1`                    | Comma-separated list of up to five predictors chosen as most important for whether a paper discusses **racial** inequality.                       | Raw    |
| `Q Race.2 (rank) - <feature>` | Rank of that feature among the respondent’s Race.1 selections (`1` = most important). Blank if not selected. One column per feature (13 columns). | Raw    |
| `Q Race.3 (sign) - <feature>` | Expected direction of the main effect: `+` or `-`. Blank if not selected. One column per feature (13 columns).                                    | Raw    |




### 3.2 Pre-ML theory and diagram (main effects)


| Column                                   | Meaning                                                                                   | Source       |
| ---------------------------------------- | ----------------------------------------------------------------------------------------- | ------------ |
| `Q Race.4 pre-ML theory (main effects)`  | Free-text theoretical explanation for the main-effect choices (before seeing ML results). | Raw          |
| `Q Race.5 pre-ML diagram (main effects)` | Free-text causal diagram / path description for the main-effect theory.                   | Raw          |
| `Q Race.5 Number of paths`               | Count of causal paths in the pre-ML Race diagram.                                         | Coded by LLM |
| `Q Race.5 Maximum path length`           | Length of the longest path in that diagram.                                               | Coded by LLM |
| `Q Race.5 Number of latent variables`    | Count of latent (unobserved) constructs in that diagram.                                  | Coded by LLM |
| `Q Race.5 Coding reasoning`              | Brief justification of the three diagram codes above.                                     | Coded by LLM |




### 3.3 Second-order interactions (SOI), before ML


| Column                          | Meaning                                                          | Source |
| ------------------------------- | ---------------------------------------------------------------- | ------ |
| `Q Race.6 (SOI, 1st)`           | First chosen interaction: two features as `feature_a,feature_b`. | Raw    |
| `Q Race.7 (SOI, 2nd)`           | Second chosen interaction (same format).                         | Raw    |
| `Q Race.8 (SOI, 3rd)`           | Third chosen interaction (same format).                          | Raw    |
| `Q Race.9 (SOI, sign, 1st)`     | Expected sign of the 1st interaction (`+` / `-`).                | Raw    |
| `Q Race.9 (SOI, sign, 2nd)`     | Expected sign of the 2nd interaction.                            | Raw    |
| `Q Race.9 (SOI, sign, 3rd)`     | Expected sign of the 3rd interaction.                            | Raw    |
| `Q Race.10 pre-ML theory (SOI)` | Free-text theory for the three Race SOI choices (before ML).     | Raw    |




### 3.4 After viewing ML results (main effects)


| Column                                                           | Meaning                                                                                                                                                                                                                                                                                                   | Source       |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| `Q Race.11 reaction after viewing the ML results (main effects)` | Free-text reaction to the ML main-effect results.                                                                                                                                                                                                                                                         | Raw          |
| `Q Race.12 post-ML theory (main effects)`                        | Respondent’s revised main-effect theory text after seeing ML.                                                                                                                                                                                                                                             | Raw          |
| `Q Race.12 LLM_refined post-ML theory (main effects)`            | Self-contained post-ML theory text for analysis (embeddings / quality ratings). Many respondents’ raw post-ML answers only say what to change relative to their pre-ML theory and are not complete on their own; an LLM therefore combines the pre- and post-ML responses into one self-contained theory. | Coded by LLM |
| `Q Race.12 LLM_status post-ML theory (main effects)`             | How the refined text was produced: `already_complete` (post text usable as-is); `merged_pre_post` (pre + post combined); `no_change` (kept pre-ML theory).                                                                                                                                                | Coded by LLM |
| `Q Race.12 LLM_uncertainty_note post-ML theory (main effects)`   | Optional note when merging/refining was uncertain or incomplete.                                                                                                                                                                                                                                          | Coded by LLM |
| `Q Race.13 post-ML diagram (main effects)`                       | Revised causal diagram after seeing ML main-effect results.                                                                                                                                                                                                                                               | Raw          |
| `Q Race.13 Number of paths`                                      | Path count for the post-ML Race diagram.                                                                                                                                                                                                                                                                  | Coded by LLM |
| `Q Race.13 Maximum path length`                                  | Max path length for that diagram.                                                                                                                                                                                                                                                                         | Coded by LLM |
| `Q Race.13 Number of latent variables`                           | Latent-variable count for that diagram.                                                                                                                                                                                                                                                                   | Coded by LLM |
| `Q Race.13 Coding reasoning`                                     | Justification of the post-ML Race diagram codes.                                                                                                                                                                                                                                                          | Coded by LLM |




### 3.5 After viewing ML results (SOI) and empirical-test prompts


| Column                                                                                                                                                               | Meaning                                                                                                                                     | Source       |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| `Q Race.14 reaction after viewing the ML results (SOI)`                                                                                                              | Free-text reaction to the ML interaction results.                                                                                           | Raw          |
| `Q Race.15 post-ML theory (SOI)`                                                                                                                                     | Revised SOI theory after seeing ML.                                                                                                         | Raw          |
| `Q Race.15 LLM_refined post-ML theory (SOI)`                                                                                                                         | Same as Race.12 LLM_refined, for SOI: self-contained post-ML theory when the raw post answer only describes edits to the pre-ML SOI theory. | Coded by LLM |
| `Q Race.15 LLM_status post-ML theory (SOI)`                                                                                                                          | Same codes as Race.12 LLM_status.                                                                                                           | Coded by LLM |
| `Q Race.15 LLM_uncertainty_note post-ML theory (SOI)`                                                                                                                | Optional uncertainty note for SOI refine/merge.                                                                                             | Coded by LLM |
| `Q Race.16 How could your revised theoretical reasoning on racial inequality be empirically tested using the available data?`                                        | Open response: tests using predictors already in the project.                                                                               | Raw          |
| `Q Race.17 How could your theoretical reasoning be further tested using additional variables that you consider relevant but are NOT currently included in our list?` | Open response: additional variables **not** in the current feature list, how to use them, and expected results.                             | Raw          |


---



## 4. Gender inequality block (`Q Gender.*`)

Same structure and Source labels as Race, for predicting whether a paper discusses gender inequality.

---



## 5. Survey flow (for reading the columns in order)

For the full wording of each item, see the paper appendix **Survey contents**.

```text
IDs → Race main-effect select/rank/sign → Race pre-ML main-effect theory/diagram
    → Race SOI select/sign → Race pre-ML SOI theory
    → Race post-ML main-effect reaction/theory/diagram
    → Race post-ML SOI reaction/theory
    → Race proposed empirical tests
→ same sequence for Gender
```

---

