# human-ai-theory-building

Code accompanying the paper *AI Theory Building: Artificial intelligences and human scientists exhibit complementary strengths in theory building*.

- **survey-analysis** — Analysis of the collected survey data.
- **machine-learning-analysis** — Random Forest and logistic regression code for predicting inequality mentions in academic discourses.

This code was developed and tested on macOS. For any questions about the code, please contact Ke Li ([ke.li@insead.edu](mailto:ke.li@insead.edu)).

## Setup

1. Install [Python 3](https://www.python.org/downloads/) if you do not have it yet.
2. Clone this repository and enter the project folder:

```bash
git clone https://github.com/KeLiChloe/human-ai-theory-building.git
cd human-ai-theory-building
```

1. Install the packages:

```bash
pip install -r requirements.txt
```

We recommend using an isolated virtual environment so these packages do not interfere with other Python projects on your computer. On macOS/Linux, if `pip` is not found, try `pip3` instead.

## survey-analysis

This folder analyzes the survey responses from humans and GenAI systems on predicting racial and gender inequality in academic discourses. Column definitions for the survey file: [`anonymous_survey_data_codebook.md`](survey-analysis/anonymous_survey_data_codebook.md).

To reproduce **Fig. 1**, run `survey-analysis/forecasts/fig_sorted_individuals.py`.

To reproduce **Fig. 2**, run `survey-analysis/forecasts/fig_equal_size_aggregation.py`.

To reproduce **Fig. 3**, run `survey-analysis/textual_analysis/embedding_analysis/fig_semantic_space_map_PCA.py`.

To reproduce **Fig. 4**, run `survey-analysis/textual_analysis/embedding_analysis/fig_within_group_dispersion.py`.

To reproduce **Fig. 5**, run `survey-analysis/textual_analysis/embedding_analysis/fig_theory_revision.py`.

To reproduce **Extended Data Table 1**, run `survey-analysis/forecasts/fig_accuracy_tables_all.py`.

To reproduce **Extended Data Table 2**, run `survey-analysis/forecasts/fig_forecast_diversity.py`.

To reproduce **Extended Data Table 3**, run `survey-analysis/moderator_analysis/moderator_regression.py`.

To reproduce **Extended Data Table 4**, run `survey-analysis/diagram/fig_theory_complexity.py`.

To reproduce **Extended Data Fig. 4**, run `survey-analysis/forecasts/fig_feature_selection_frequency.py`.

To reproduce **Extended Data Fig. 5**, run `survey-analysis/textual_analysis/embedding_analysis/fig_core_tail_structure.py`.

To reproduce **Extended Data Fig. 6**, run `survey-analysis/textual_analysis/theory_quality_rating/fig_theory_rating_quality.py`.

## machine-learning-analysis

To reproduce **Extended Data Fig. 2** and **Extended Data Fig. 3** in the paper, run `machine-learning-analysis/code/plot_ML_results_by_domain.py`.

Precomputed random forest and logistic regression results are already stored under `machine-learning-analysis/models/`. The plotting script reads those files and regenerates the figures—no re-training needed.

If you want to re-train the models yourself:

1. Download the paper-level datasets from [Zenodo](https://zenodo.org/records/22647761?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjczN2FlY2Q4LTlhMmItNGM0Yy1iMDliLWE2MTQyMGVhOWRjYiIsImRhdGEiOnt9LCJyYW5kb20iOiI0ZTljNDdlN2FiNDM3MGJjNDBkMjY2MzAzYjM0ZjI0MSJ9.TLXZh_F8dwkNPlKWkalo_w9YvTAwz2JJwiVVtEJRRJCAT8raAr_Fdd7SBkt_qA6nf58C8gMjlKs6XpxoNINJrA) (`gender_ineq_dataset.csv` and `race_ineq_dataset.csv`) and place them in `machine-learning-analysis/data/`.
2. Run the following scripts, then run the plotting script again:
  - `machine-learning-analysis/run_gender.sh`
  - `machine-learning-analysis/run_race.sh`

