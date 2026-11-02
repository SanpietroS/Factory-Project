# Factory Machine Failure Prediction

## Problem Statement

This project identifies observed machine failures from sensor readings. The goal is to detect failures while limiting unnecessary maintenance alerts. Missed failures may lead to downtime, repairs, and safety problems; false alerts may cause unnecessary inspections.

The analysis compares Logistic Regression, an MVP Random Forest, and an improved Random Forest. The dataset labels describe failure at the recorded observation, so this project does not establish how far ahead failures can be predicted.

## Data Description

The [AI4I 2020 Predictive Maintenance Dataset](https://doi.org/10.24432/C5HS5C) is a synthetic dataset from the UCI Machine Learning Repository, shared under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

- 10,000 observations, including 339 failures and 9,661 nonfailures.
- Failure rate: 3.39%.
- Inputs: product Type, air temperature, process temperature, rotational speed, torque, and tool wear.
- Target: `Machine failure`.

### Cleaning Decisions

The source has no missing values or duplicate rows, so no imputation or row removal is needed. Column whitespace is trimmed, and duplicate source rows are checked before removing identifiers.

UDI and Product ID are excluded because they identify records. TWF, HDF, PWF, OSF, and RNF are excluded from model inputs because they describe failure outcomes and could reveal the answer. Valid extreme sensor readings are kept because unusual values may help identify failures. Product Type is encoded with H as the reference and indicators for L and M.

## Methodology

| Stage | Model | Purpose |
|---|---|---|
| Reference | Logistic Regression with training-fitted scaling and balanced class weights | Provide a simple linear comparison |
| MVP baseline | Random Forest with 100 trees and balanced class weights | Capture nonlinear patterns using the original seven inputs |
| Improved model | Random Forest with two added features and training-only tuning | Improve the precision/recall balance |

The improvement adds temperature difference (process minus air temperature) and a power proxy (torque multiplied by rpm, not watts). The earlier high-tool-wear flag was removed because its cutoff used the full dataset.

All models use the same 80/20 stratified split with seed 42: 8,000 training observations and 2,000 test observations, including 68 test failures. Scaling and tuning use training data only. The prediction threshold is 0.5 for every model.

The improved forest tests eight configurations across three shuffled stratified training folds. F1 selects the number of trees, maximum depth, and minimum leaf size. The verified run selected 300 trees, unlimited depth, and a minimum leaf size of 1, with mean training cross-validation F1 of 0.806. Feature engineering and tuning change together, so their separate effects have not been established.

Accuracy is reported alongside precision, recall, F1, and ROC-AUC. Failures are rare: always predicting no failure would achieve 96.61% accuracy and detect none of the failures. F1, precision, and recall therefore matter more than accuracy alone.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Missed failures | False alerts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.825 | 0.142 | 0.824 | 0.242 | 0.907 | 12 | 339 |
| MVP Random Forest | 0.981 | 0.734 | 0.691 | 0.712 | 0.960 | 21 | 17 |
| Improved Random Forest | 0.991 | 0.917 | 0.809 | 0.859 | 0.970 | 13 | 5 |

Logistic Regression detects the most failures but creates a large false-alert workload. The MVP reduces false alerts but misses more failures.

The improved forest provides the strongest balance. It detects 55 of 68 failures and produces five false alerts. Compared with the MVP, it detects eight additional failures and avoids 12 false alerts. Compared with Logistic Regression, it misses one additional failure while avoiding 334 false alerts.

Its remaining 13 missed failures show that it is not a complete maintenance decision system. Errors could affect downtime, inspections, repair costs, and safety, but actual costs and savings are not measured here.

Tool wear, rotational speed, and torque have the highest impurity importance, followed by the power proxy and temperature difference. These rankings describe this fitted model; they do not establish causation.

### Notebook Figures

The figures are explained in the notebook and linked here without embedding graphs:

- [Class distribution](Factory_Project_Submission.ipynb#3.-Explore-the-Data): shows why accuracy alone is insufficient.
- [Sensor boxplots and correlation heatmap](Factory_Project_Submission.ipynb#Sensor-Readings-and-Machine-Failure): compare sensor patterns and associations; failure-type flags shown in the exploratory heatmap are excluded from training.
- [Model comparison](Factory_Project_Submission.ipynb#Model-Comparison): compares the models on the same metrics.
- [Refined confusion matrix](Factory_Project_Submission.ipynb#Refined-Confusion-Matrix): shows 13 missed failures and five false alerts.
- [Refined feature importance](Factory_Project_Submission.ipynb#Refined-Feature-Importance): shows the relative importance of all nine improved-model inputs.

## Limitations

The dataset is synthetic, only 68 failures occur in the test set, and the random split may share time-related patterns across partitions. The holdout has also been examined during development. The results are a development benchmark rather than a fresh final test or evidence of future operational performance.

The analysis does not measure uncertainty, maintenance costs, or a future prediction horizon. Feature and tuning changes were combined, and the default alert threshold was not selected using real costs. Impurity importance can favor continuous or correlated inputs.

Future work includes time-based and external validation, uncertainty estimates, separate feature/tuning experiments, and threshold selection using realistic maintenance costs. These are extensions of the current analysis.

## How to Run

1. Download `Factory_Project_Submission.ipynb` and the CSV from the UCI dataset source above.
2. Use Python 3.12. The verified analysis used these scientific package versions:

   ```bash
   python -m pip install numpy==2.5.4 pandas==3.1.0 scipy==1.18.1 scikit-learn==1.9.1 matplotlib==3.11.2 seaborn==0.13.2 notebook ipykernel
   ```

3. In the notebook's **1. Load the Dataset** section, change the CSV path to where the file is saved on your computer. The supplied notebook uses the author's Windows Downloads path; it does not download the data automatically.
4. Start Jupyter with `python -m jupyter notebook`, open the notebook in the environment where the packages were installed, and select **Restart Kernel and Run All**.
5. Review the results and save the executed notebook.

All analysis code is inside the notebook. Running it creates `images/` and `results/` folders containing the figures, model scores, cross-validation results, and feature importance. No separate Python source file or API key is required.

The reported values come from the verified environment. Other library versions may change the selected parameters and results; any written interpretation should match the actual output of the submitted notebook.
