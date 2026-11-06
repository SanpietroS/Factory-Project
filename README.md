# Factory Machine Failure Prediction

## Problem Statement

This project uses machine sensor readings to identify failures. The goal is to detect failures while limiting unnecessary maintenance alerts.

Missed failures may lead to downtime, repairs, or safety problems. False alerts may cause unnecessary inspections.

The project compares Logistic Regression, an MVP Random Forest, and an improved Random Forest.

The dataset describes failure at the recorded observation. It does not establish how far ahead a failure can be predicted.

## Data Description

The project uses the synthetic AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository.

- **Observations:** 10,000
- **Failures:** 339
- **Nonfailures:** 9,661
- **Failure rate:** 3.39%
- **Target:** Machine failure
- **Inputs:** Product type, air temperature, process temperature, rotational speed, torque, and tool wear

**Source:** [AI4I 2020 Predictive Maintenance Dataset](https://doi.org/10.24432/C5HS5C)

**License:** [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)

### Data Cleaning

I checked missing values, duplicate source rows, and data types. The source has no missing values or duplicate rows, so no imputation or row removal was needed.

I removed UDI and Product ID because they identify records. I excluded TWF, HDF, PWF, OSF, and RNF from the predictors because they describe failure outcomes and could reveal the answer.

Product Type was converted into numeric indicator columns. Valid extreme sensor readings were kept because unusual conditions may help identify failures.

## Methodology

| Stage | Model | Purpose |
|---|---|---|
| Reference | Logistic Regression | Provide a simple linear comparison |
| MVP baseline | Random Forest with 100 trees | Capture nonlinear patterns using the original inputs |
| Improved model | Random Forest with added features and tuned settings | Improve the balance between failure detection and false alerts |

All models use balanced class weights.

The data is divided into 80% training and 20% testing using stratification and a random seed of 42. This creates 8,000 training observations and 2,000 test observations, including 68 test failures.

All models use the same test observations. Logistic Regression scaling is fitted only on training data.

### Improved Model

The improved Random Forest adds two features:

| Feature | Calculation | Reason |
|---|---|---|
| Temperature difference | Process temperature minus air temperature | Represent the temperature gap related to heat dissipation |
| Power proxy | Torque multiplied by rotational speed | Combine information about load and speed |

The power proxy uses torque times rpm and is not measured in watts.

The improved model tests eight configurations using three stratified cross-validation folds within the training data. F1 selects the settings.

The verified run selected 300 trees, unlimited maximum depth, and a minimum leaf size of 1. Mean training cross-validation F1 was 0.806.

Feature engineering and tuning were changed together, so their separate contributions have not been measured.

### Evaluation

The comparison includes accuracy, precision, recall, F1, and ROC-AUC.

Failures are rare. Always predicting no failure would achieve 96.61% accuracy while missing every failure. Precision, recall, and F1 therefore provide important information beyond accuracy.

All models use the default 0.5 probability threshold.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.825 | 0.142 | 0.824 | 0.242 | 0.907 |
| MVP Random Forest | 0.981 | 0.734 | 0.691 | 0.712 | 0.960 |
| Improved Random Forest | 0.991 | 0.917 | 0.809 | 0.859 | 0.970 |

Logistic Regression detects 56 of the 68 failures but produces 339 false alerts. Its high recall comes with a large alert workload.

The MVP Random Forest reduces false alerts to 17 but misses 21 failures.

The improved Random Forest detects 55 failures, misses 13, and produces five false alerts. Compared with the MVP, it detects eight additional failures while avoiding 12 false alerts.

It has the highest F1 score at 0.859 and provides the strongest precision/recall balance on this test set.

The remaining missed failures still matter. They could lead to downtime or repairs, while false alerts could trigger unnecessary inspections. These are possible consequences; the dataset does not measure actual maintenance costs or savings.

Tool wear, rotational speed, and torque have the highest impurity importance in the improved model. These rankings describe how the fitted model uses its inputs and do not prove cause and effect.

### Notebook Charts

The notebook contains the charts and their explanations:

- [Class distribution](Factory_Project_Submission.ipynb#3.-Explore-the-Data)
- [Sensor readings and correlations](Factory_Project_Submission.ipynb#Sensor-Readings-and-Machine-Failure)
- [Model comparison](Factory_Project_Submission.ipynb#Model-Comparison)
- [Refined confusion matrix](Factory_Project_Submission.ipynb#Refined-Confusion-Matrix)
- [Refined feature importance](Factory_Project_Submission.ipynb#Refined-Feature-Importance)

## Limitations

The dataset is synthetic, so the results may not transfer to real factory equipment. Only 68 failures appear in the test set, which limits certainty.

The random split may place similar time-related patterns in both groups. The test set has also been examined during development, so these results are a development comparison rather than a fresh final evaluation.

Feature engineering and tuning were changed together. The analysis does not establish their separate effects, estimate uncertainty, or select an alert threshold using real maintenance costs.

Future work includes time-based and external validation, threshold comparisons, uncertainty estimates, and testing on real machine histories.

## How to Run

1. Download `Factory_Project_Submission.ipynb` and the CSV from the UCI source linked above.

2. Use Python 3.12. The verified analysis used these scientific package versions:

   ```bash
   python -m pip install numpy==2.5.4 pandas==3.1.0 scipy==1.18.1 scikit-learn==1.9.1 matplotlib==3.11.2 seaborn==0.13.2 notebook ipykernel
   ```

3. Open the notebook and update the CSV path under **1. Load the Dataset** to match where the file is saved on your computer.

4. Select **Restart Kernel and Run All**.

5. Review the results and save the executed notebook.

All analysis code is inside the notebook. Running it creates `images` and `results` folders containing the charts, model scores, cross-validation results, and feature importance.

This version uses a local CSV path and does not download the dataset automatically.

The reported numbers describe the verified run. If different package versions change the results, update the written explanations to match the actual notebook output.
