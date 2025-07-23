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
│   ├── splits.py              # Train/test split based on dates (Train: Data from the past week, Test: Data from the current date)
│
├── evaluation/
│   ├── metrics.py             # MAE/RMSE evaluations for model & OCF
│   └── runner.py              # Main rolling evaluation logic
│
├── utils/
│   └── utils.py               # Optional functions: Currently has styled display for notebooks or command line
│
├── run.py                     # Entry point with argparse
├── requirements.txt           # Project dependencies
</pre>

---

## 🚀 Usage

### 1. Setup Virtual Environment

```bash
python3 -m venv adjuster_env
source adjuster_env/bin/activate
```

### 2. Install dependencies using requirements.txt

```bash
pip install -r requirements.txt
```

### 3. Running the pipeline

```bash
python run.py --input_csv adjuster_dataset_gsoc_v1.csv --start_date 2024-08-01 --model_type xgboost 
```

Currently has optional parameters like --add_lagged_features
<!-- Can be expanded to use only certain features -->

### 4. Interpreting Results
The results are saved in a structured folder format under results/ for easy navigation and comparison across models and dates:
<pre>
results/                                        
├── model/                                     # TabPFN, XGBoost
│   └── start_date/                            # Starting Date for Evaluation (Following Results are averaged between start date and max possible end date)
│       ├── avg_errors_all_dates.csv           # Summary of MAE/RMSE for TabPFN as model compared to OCF Adjuster across all the dates
│       ├── avg_errors_per_date.csv            # MAE/RMSE for each forecast day across every horizon and hour
│       ├── avg_errors_per_horizon.csv         # Errors aggregated per forecast horizon only for all the dates
│       ├── avg_errors_per_hour.csv            # Errors aggregated per hour of the day only for all the dates
│       └── avg_errors_per_horizon_hour.csv    # Detailed breakdown by both hour and horizon
</pre>
<!-- ### 5.  Plotting ? -->
