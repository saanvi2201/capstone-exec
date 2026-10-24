# Evidence-Gated Heart Disease Risk Assessment

This project explores evidence-gated heart disease risk assessment for older adults in India using data from the Longitudinal Ageing Study in India (LASI). It develops models for coronary heart disease (CHD) and ischemic heart disease (IHD), then compares them with WHO-based risk benchmarks.

## Analysis workflow

The notebooks in `notebooks/lasi/` are intended to run in order:

1. `nb1_lasi_feature_engineering.ipynb` prepares the modelling features.
2. `nb2_lasi_train_test_split.ipynb` creates training and test sets.
3. `nb3_lasi_model_training.ipynb` trains the models.
4. `nb4_lasi_model_evaluation.ipynb` evaluates discrimination, calibration, and decision-curve performance, with subgroup and geographic checks.
5. `nb5_who_benchmark.ipynb` compares model results with WHO-based benchmarks.

The analysis includes logistic regression and gradient boosting models, along with age and WHO-based baseline comparisons. The notebooks save summary tables under `outputs/tables/` and plots under `outputs/figures/`.

## Data

LASI source and processed datasets are excluded from Git and are not included in this repository. Obtain and use the data according to its access requirements, then place the source file at `data/lasi/raw/lasi_merged_clear.csv` before running the notebooks. The first notebook creates processed files under `data/lasi/processed/` for subsequent steps.

## Run

1. Set up Python with Jupyter and the packages imported by the notebooks.
2. Add the LASI source file at the path above.
3. Open `notebooks/lasi/` and run notebooks `nb1` through `nb5` in order.
