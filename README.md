<h2 align="center">
  PCOS Classification
</h2>

<p align="center">
  <img src="https://github.com/ardhikaptr11/pcos-classification/actions/workflows/train.yml/badge.svg" alt="Train">
  <img src="https://github.com/ardhikaptr11/pcos-classification/actions/workflows/preprocess.yml/badge.svg" alt="Preprocess">
  <img src="https://github.com/ardhikaptr11/pcos-classification/actions/workflows/deploy.yml/badge.svg" alt="Deploy">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  <a href="https://colab.research.google.com/drive/1SmrvvH-ixLNi0K7eE1vo2a6Ly8gG2mYP#scrollTo=LDLf2l8fCGe8">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</p>

## Overview

**🔗 Live demo: [PCOS Pal](https://pcos-pal-liart.vercel.app)**

This project shows an end-to-end ML system for PCOS risk classification to predicts whether a patient has PCOS from clinical and physical examination data. For **positive** predictions, it returns the model's confidence score; for **negative** predictions, it performs risk stratification, categorizing the patient into **Low**, **Moderate**, or **High** risk. It covers the full ML lifecycle — data preprocessing, model training with hyperparameter tuning, experiment tracking, and deployment as a REST API — built as a personal portfolio project to demonstrate an end-to-end, production-style ML workflow.

> **Disclaimer:** This project is for portfolio and demonstration purposes only. It is trained on a small dataset (541 records) and is **not intended for clinical or diagnostic use**.

## Dataset

- **Source:** [Polycystic Ovary Syndrome (PCOS) — Kaggle](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos/data)
- **Size:** 541 rows, 45 columns (before preprocessing)
- **Target:** Binary classification — PCOS Positive / Negative
- **Class distribution:** 177 positive, 364 negative (~33% positive; moderately imbalanced)
- **Features:** All available clinical, hormonal, and physical examination features are used

## Tech Stack

**Data & Modeling**
- pandas, numpy, openpyxl
- scikit-learn, XGBoost

**Hyperparameter Tuning**
- Optuna, OptunaHub

**Experiment Tracking**
- MLflow
- DagsHub (remote tracking backend)

**Serving & Monitoring**
- FastAPI, Uvicorn
- Prometheus FastAPI Instrumentator

**Testing & Code Quality**
- pytest
- black, flake8, mypy

**Environment & Packaging**
- uv
- Docker

**CI/CD**
- GitHub Actions

## Repository Structure

```
.
├── .github/workflows/         # CI pipelines (preprocess, train, deploy)
├── common/                    # Shared utilities (logging, artifact download/upload)
├── config/                    # Pipeline configuration (config.yaml)
├── data/                      # Sample request payloads for API testing
├── dataset/
│   ├── raw/                   # Original Kaggle dataset files
│   └── processed/             # Cleaned & transformed dataset
├── deployment/                # FastAPI app, CLI handler/parser, predictor logic
├── notebooks/                 # Exploratory model experimentation
├── preprocessing/             # Data cleaning & transformation pipeline
│   └── utils/                 # Cleaner, loader, schemas, transformers
├── training/                  # Baseline & tuned training pipelines
│   └── utils/                 # Artifacts, evaluation, metrics, loader
├── tests/                     # Automated tests
├── Dockerfile
├── MLProject                  # MLflow project definition (baseline/tuning entry points)
├── conda.yaml                 # MLflow project environment
├── main.py                    # CLI entry point (preprocess / train / deploy)
├── pyproject.toml
└── uv.lock
```

## Methodology

### Preprocessing

- **Cleaning:** fix typos, adjust whitespace in column names, drop unneeded features, fill missing values, handle extreme values
- **Transforming:** convert object columns to numeric, cast float-typed categorical columns to integer, engineer new features, standardize column names to snake_case, reorder columns

Output is written to `dataset/processed/pcos_data_preprocessed.csv`.

### Training

- **Model:** XGBoost classifier
- **Baseline:** trained with default/fixed hyperparameters (`training/baseline.py`)
- **Tuning:** hyperparameter search via Optuna/OptunaHub — 30 trials, optimizing ROC-AUC (`training/tuning.py`)
- Both paths are orchestrated as MLflow Project entry points (`MLProject`) and driven by `config/config.yaml`
- Experiment tracking is toggled with `--watch`: `remote` logs to [DagsHub](https://dagshub.com/putucrisna11/pcos-classification/experiments), `local` logs to a local MLflow server

`config.yaml` controls the path to the preprocessed dataset, the MLflow experiment name, the baseline model's fixed hyperparameters, and the search space/sampler settings used during tuning. It's already included in the repo at `config/config.yaml` — use it as-is or edit it to your own values:

**Example `config.yaml`:**

```yaml
data_path: dataset/processed/pcos_data_preprocessed.csv
experiment_name: PCOS Classification

baseline:
  run_name: XGBoost_Baseline
  params:
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1
    random_state: 42

tuning:
  run_name: XGBoost_Tuning_Optuna_ROC_AUC
  optimization_target: roc_auc
  eval_metric: logloss
  seed: 42

  sampler:
    use_optunahub: true
    package: samplers/tpe_union_multivariate
    class_name: TPEUnionMultivariateSampler
    kwargs:
      n_startup_trials: 5
      seed: 42

  search_space:
    n_estimators:
      type: int
      low: 50
      high: 200
    max_depth:
      type: int
      low: 3
      high: 10
    learning_rate:
      type: float
      low: 0.01
      high: 0.3
      log: true
    subsample:
      type: float
      low: 0.5
      high: 1.0
    colsample_bytree:
      type: float
      low: 0.5
      high: 1.0
    gamma:
      type: float
      low: 0.0
      high: 5.0
    min_child_weight:
      type: int
      low: 1
      high: 10
    scale_pos_weight:
      type: dynamic_scale
      low: 1.0
      multiplier: 2.0
```

### Deployment
 
- A FastAPI app (`deployment/app.py`) serves predictions, loading the trained model from an MLflow model artifact (`MODEL_PATH=model_artifacts/hub/model/MLmodel`)
- The `deploy` CLI command exposes three subcommands: `download-model <dagshub|gdrive>` (pull model artifacts from the specified source), `serve` (run the FastAPI server), and `predict` (run one-off inference locally)
- `deploy predict` runs inference directly in the terminal for quick local debugging without starting a server, while the `/predict` API endpoint (served via `deploy serve`) accepts structured JSON patient data from external clients — such as the [PCOS Pal](https://pcos-pal-liart.vercel.app) frontend — and returns real-time predictions
- Containerized with Docker, exposed on port `8000`
- Sample request payloads for manual testing are provided in `data/sample_positive.json` and `data/sample_negative.json`

## Results

| Metric | Baseline | Tuned |
| --- | --- | --- |
| Accuracy | 0.9266 | 0.9174 |
| Precision | 0.9375 | 0.9091 |
| Recall | 0.8333 | 0.8333 |
| F1 Score | 0.8824 | 0.8696 |
| ROC-AUC | 0.9429 | 0.9528 |
| PR-AUC | 0.9322 | 0.9445 |
| Log Loss | 0.2635 | 0.2295 |

Tuning optimizes for ROC-AUC evaluated via K-Fold cross-validation, rather than accuracy on a single split. The baseline's higher raw accuracy reflects some overfitting to that split, while the tuned model generalizes better and separates positive/negative cases more reliably across thresholds — reflected in its improved ROC-AUC, PR-AUC, and Log Loss, at the cost of slightly lower accuracy, precision, and F1 on this particular split. This is expected behavior, not a regression.

## Usage

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

Core runtime libraries (`fastapi`, `pydantic`, `xgboost`, `pandas`, `numpy`, `scikit-learn`, etc.) and their exact versions are pinned in `pyproject.toml` / `uv.lock` — `uv sync` sets up the full environment automatically.

### Environment Variables

Copy `.env.example` to `.env` and fill in your own credentials before running any pipeline step:

```bash
cp .env.example .env
```

```dotenv
MODEL_NAME=test-model

MLFLOW_TRACKING_URI=https://dagshub.com/<USERNAME>/<REPOSITORY>.mlflow/
MLFLOW_TRACKING_URI_LOCAL=http://127.0.0.1:<PORT>

DAGSHUB_REPO_OWNER=<YOUR USERNAME>
DAGSHUB_USER_TOKEN=<YOUR USER TOKEN>

GDRIVE_API_KEY=<YOUR API KEY>
GDRIVE_CREDENTIALS={"token": "", "refresh_token": "", "token_uri": "", "client_id": "", "client_secret": "", "scopes": ["https://www.googleapis.com/auth/drive"], "universe_domain": "googleapis.com", "account": "", "expiry": ""}
GDRIVE_FOLDER_ID=<YOUR FOLDER ID>

DOCKER_HUB_USERNAME=<YOUR USERNAME>
DOCKER_HUB_ACCESS_TOKEN=<YOUR ACCESS TOKEN>
DOCKER_HUB_REPO_NAME=<YOUR REPOSITORY NAME>
```

> `.env.example` is not yet included in the repository; it will be added.

### Installation

```bash
git clone https://github.com/ardhikaptr11/pcos-classification.git
cd pcos-classification
uv sync
```

### Running the pipeline

> The preprocessed dataset (`dataset/processed/pcos_data_preprocessed.csv`) is already included in the repo, so you can skip straight to training/evaluation on a fresh clone. Re-run preprocessing only if you introduce new raw data or change the preprocessing logic.

```bash
# Preprocess the raw dataset (see `--help` for which file goes in `primary` vs. `secondary`)
uv run python main.py preprocess [-o OUTPUT] primary [secondary]
uv run python main.py preprocess --help

# primary    Path to the primary dataset file (.csv)
# secondary  Path to the secondary dataset file (.xlsx/.csv) [optional]

# Train a baseline model, tracked locally
uv run python main.py train --baseline --watch local --config config/config.yaml

# Train with hyperparameter tuning, tracked remotely on DagsHub, custom trial count
uv run python main.py train --tuning --watch remote --trials 30 --config config/config.yaml

# Download model artifacts from DagsHub or Google Drive
uv run python main.py deploy download-model dagshub

# Serve the trained model as a REST API
uv run python main.py deploy serve

# Run one-off inference from the CLI
uv run python main.py deploy predict
```

**Full CLI usage:**

```
usage: uv run python main.py train [-h] (--baseline | --tuning) [-W {local,remote}] [--trials TRIALS] [-C CONFIG]

options:
  -h, --help            show this help message and exit
  --baseline            Train a baseline model
  --tuning              Train a model with hyperparameter tuning
  -W {local,remote}, --watch {local,remote}
                        Watch the experiment
  --trials TRIALS       Number of trials for hyperparameter tuning
  -C CONFIG, --config CONFIG
                        Path to the configuration file (.yaml)
```

```
usage: uv run python main.py deploy [-h] {download-model,serve,predict} ...

positional arguments:
  {download-model,serve,predict}
    download-model      Download model artifacts
    serve               Run FastAPI server
    predict             Run inference
```

```
usage: uv run python main.py deploy download-model [-h] {dagshub,gdrive}

positional arguments:
  {dagshub,gdrive}      Source to download the model artifacts from
```

Training can also be run as an MLflow Project:

```bash
uv run mlflow run . -e baseline --env-manager=local
uv run mlflow run . -e tuning --env-manager=local
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Root — service info |
| `/health` | GET | Service health status |
| `/predict` | POST | Run inference on a patient record |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @data/sample_positive.json
```

### Running with Docker

```bash
docker build -t pcos-classification .
docker run -p 8000:8000 pcos-classification
```

## Testing

```bash
uv run pytest
```

> Unit and integration tests are currently combined in `tests/test_deployment.py` for simplicity during early development. Splitting these into dedicated test modules is planned.

## Roadmap

- [x] Next.js frontend for interactive predictions
- [ ] Prometheus + Grafana monitoring integration
- [ ] Separate unit and integration test suites

No live demo is available yet since the frontend is still planned. In the meantime, run the API locally with `uv run python main.py deploy serve` and test it with `uv run python main.py deploy predict` or the `curl` example above.

## Limitations

- Trained on a small dataset (541 records); results should be interpreted cautiously
- Not validated for clinical or diagnostic use — built for portfolio demonstration only

## Contributing

This project is still in its early stages and is planned to be developed further — contributions are welcome. Feel free to open an issue or submit a pull request with improvements, bug fixes, or new features.

## License

This project is licensed under the [MIT License](LICENSE).