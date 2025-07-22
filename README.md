# TabAdjust
Adjuster This! TabPFN for Solar Forecast Error Adjustment

A modular pipeline for evaluating and improving photovoltaic (PV) power forecast accuracy using TabPFN and XGBoost. This project allows comparison of machine-learned adjusters against rule-based baselines (OCF-style adjustments).

---

## 🔧 Features

- Tabular foundation model support (TabPFN, XGB) for forecasting adjuster error
- Baseline comparison with Open Climate Fix-style adjusters
- Modular structure for preprocessing, modeling, and evaluation
- CLI-based runner with config YAML support

---

## 📁 Project Structure
<pre>
TabAdjust/
├── data/
│   ├── load.py                # Function to load input CSV
│   └── preprocess.py          # Basic preprocessing and lag feature generation
│
├── models/
│   └── tab_adjust.py          # Contains BaseModel, TabPFNModel, and XGBModel
│
├── core/
│   ├── preprocessing.py       # Model-specific data preparation
│   ├── feature_selection.py   # Permutation-based feature selection
│   ├── splits.py              # Train/test split based on dates
│
├── evaluation/
│   ├── metrics.py             # MAE/RMSE evaluations for model & OCF
│   └── runner.py              # Main rolling evaluation logic
│
├── utils/
│   └── diagnostics.py         # Optional styled display for notebooks
│
├── run.py                     # Entry point with argparse + YAML config
├── pyproject.toml             # Project metadata & dependencies
</pre>



---

## 🚀 Usage

### 1. Install dependencies using pyproject.toml

```bash
pip install -e .
```

### 2. Running the pipeline

```bash
python run.py --input_csv adjuster_dataset_gsoc_v1.csv --start_date 2024-08-01 --model_type xgboost --output_prefix results_xgb
```
