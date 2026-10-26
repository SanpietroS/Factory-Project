# Factory Machine Failure Prediction

## Project Overview

This project uses machine sensor data to identify machine failures. The goal is to detect failures while limiting unnecessary maintenance alerts.

Missed failures could lead to downtime, repair costs, or safety problems. Incorrect failure alerts could cause unnecessary inspections and maintenance.

The project compares Logistic Regression, an MVP Random Forest, and an improved Random Forest.

The dataset is synthetic. Its labels describe failure at the recorded observation, so this project does not establish how far ahead a failure can be predicted.

## Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository.

- **Observations:** 10,000
- **Failure observations:** 339
- **Nonfailure observations:** 9,661
- **Failure rate:** 3.39%
- **Target:** Machine failure
- **Inputs:** Product type, air temperature, process temperature, rotational speed, torque, and tool wear

Failures are uncommon, so accuracy alone is not enough to evaluate the models.

**Source:** [AI4I 2020 Predictive Maintenance Dataset](https://doi.org/10.24432/C5HS5C)

**License:** [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)

## Data Cleaning

I checked the dataset for missing values, duplicate source rows, and incorrect data types.

The source contains no missing values or duplicate rows, so no missing-value replacement or row removal was needed.

The preparation steps were:

- Remove extra whitespace from column names.
- Remove UDI and Product ID because they are identifiers.
- Exclude TWF, HDF, PWF, OSF, and RNF from the model inputs because they describe failure outcomes and could reveal the answer.
- Convert product Type into numeric indicator columns.
- Keep valid extreme sensor readings because unusual conditions may help identify failures.

The notebook uses a local CSV or downloads the public dataset automatically. It does not depend on a personal Downloads path.

## Model Development

| Stage | Model | Purpose |
|---|---|---|
| Reference | Logistic Regression | Provide a simple linear comparison |
| MVP baseline | Random Forest with 100 trees | Capture nonlinear relationships using the original inputs |
| Improvement | Random Forest with added features and tuned settings | Improve the balance between failure detection and false alerts |

### Reference Model

Logistic Regression uses scaled inputs and balanced class weights. The scaler is fitted only on the training data.

### MVP Baseline

The original Random Forest uses 100 trees, balanced class weights, and a random seed of 42.

It uses five sensor inputs and two product-type indicators.

### Improved Model

The improved Random Forest adds two features:

| Feature | Calculation | Reason |
|---|---|---|
| Temperature difference | Process temperature minus air temperature | Represent the temperature gap related to heat dissipation |
| Power proxy | Torque multiplied by rotational speed | Combine information about load and speed |

The power proxy uses torque multiplied by rpm and is not measured in watts.

An earlier high-tool-wear flag was removed because its cutoff used the full dataset. The retained features use formulas that do not learn statistics from test observations.

The improved model tests eight configurations using three stratified cross-validation folds within the training data. F1 is used to select the settings.

The selected model uses 300 trees, unlimited maximum depth, and a minimum leaf size of 1. Its mean training cross-validation F1 is 0.806.

Feature engineering and tuning were changed together, so their separate contributions have not been measured.

## Evaluation Method

The data is divided into 80% training and 20% testing using stratification and a random seed of 42.

- **Training observations:** 8,000
- **Test observations:** 2,000
- **Test failures:** 68

All models use the same test observations and the default 0.5 probability threshold.

The evaluation includes:

- **Accuracy:** Share of all predictions that are correct.
- **Precision:** Share of failure alerts that are actual failures.
- **Recall:** Share of actual failures detected.
- **F1:** Balance between precision and recall.
- **ROC-AUC:** Ability to rank failures above nonfailures across thresholds.

F1 is the main comparison measure because failures are rare. Precision and recall are also examined to understand the maintenance tradeoff.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.825 | 0.142 | 0.824 | 0.242 | 0.907 |
| MVP Random Forest | 0.981 | 0.734 | 0.691 | 0.712 | 0.960 |
| Improved Random Forest | 0.991 | 0.917 | 0.809 | 0.859 | 0.970 |

### Comparing the Models

Logistic Regression detects 56 of the 68 failures, but it produces 339 false alerts. Its high recall comes with low precision, which could create a large inspection workload.

The MVP Random Forest reduces false alerts to 17, but it misses 21 failures. Its F1 of 0.712 shows a better balance than Logistic Regression, although its lower recall remains a concern.

The improved Random Forest detects 55 failures, misses 13, and produces only 5 false alerts. Compared with the MVP, it detects 8 additional failures while producing 12 fewer false alerts.

The improved model has the highest F1 at 0.859. It also detects nearly as many failures as Logistic Regression while producing 334 fewer false alerts.

These results make the improved Random Forest the strongest model for this project.

### Maintenance Meaning

The improved model still misses 13 failures. Those missed cases could lead to downtime, repair costs, or safety problems.

Its five false alerts could cause unnecessary inspections. Choosing an alert threshold would therefore require information about maintenance costs and inspection capacity.

These are possible consequences. The dataset does not measure actual maintenance costs or savings.

### Feature Importance

Tool wear, rotational speed, and torque have the highest importance in the improved model. The added power proxy ranks fourth, followed by temperature difference.

These rankings describe how the fitted model uses its inputs. They do not establish cause and effect or prove which change caused the improvement.

## Notebook Charts and Findings

The charts are available in the notebook:

- [Machine failure counts](Factory_Project_Updated.ipynb#machine-failure-counts): Shows why class imbalance makes accuracy alone misleading.
- [Model comparison](Factory_Project_Updated.ipynb#model-comparison): Compares the models' accuracy, precision, recall, and F1.
- [Refined confusion matrix](Factory_Project_Updated.ipynb#refined-confusion-matrix): Shows the improved model's 13 missed failures and 5 false alerts.
- [Refined feature importance](Factory_Project_Updated.ipynb#refined-feature-importance): Shows the relative importance of all nine improved-model inputs.

## How to Run

1. Download `Factory_Project_Updated.ipynb`.
2. Install Python 3.12 and the required packages:

   ```bash
   python -m pip install pandas numpy matplotlib seaborn scikit-learn notebook ipykernel
   ```

3. Start Jupyter:

   ```bash
   python -m jupyter notebook
   ```

4. Open the notebook.
5. Select **Restart Kernel and Run All**.

The first dataset download requires internet access. The notebook can also use an unchanged `ai4i2020.csv` already in its working folder.

The notebook saves its charts and model-comparison results when it runs.

## Limitations

- The dataset is synthetic and may not represent real factory conditions.
- Only 68 failures appear in the test set.
- The data includes time-related patterns, so a random split may place similar observations in both groups.
- The test set has already been examined during development.
- Feature engineering and tuning were changed together.
- No real maintenance costs or future prediction horizon are established.

The reported scores are development results rather than proof that the model is ready for operational use.

## Future Work for Unit 10

- Compare time-based validation with the random split.
- Evaluate the model on fresh data.
- Test feature engineering and tuning separately.
- Compare alert thresholds using training or validation data.
- Measure uncertainty in the results.
- Seek real machine histories and maintenance-cost information.

## Improvements Made After Feedback

I clarified the reference model, MVP baseline, and improved model. I also explained the feature changes, corrected the personal data path, and documented the evaluation method.

This README now includes cleaning decisions, methodology, results, limitations, and future work. Links to the notebook charts connect the visualizations to specific findings.
