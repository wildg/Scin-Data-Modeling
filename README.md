# SCIN Data Modeling: Dermatological Condition Prediction

**INSY 674 Group Project**

## Project Overview

This project develops a machine learning model to predict dermatological skin conditions based on clinical data from the SCIN (Skin Condition Image Network) dataset. The model aims to predict what diagnosis a dermatologist might assign to a case based on patient demographics, symptoms, physical characteristics, and clinical observations.

## Objectives

The primary goal is to build a predictive model that can:
- Accurately predict dermatologist-assigned skin condition diagnoses
- Handle multi-class classification across various skin conditions
- Account for diverse skin types and demographic factors
- Provide interpretable results for clinical decision support

## Dataset

The SCIN dataset contains dermatological cases with comprehensive clinical information:

### Files
- `dataset_scin_cases.csv` - Patient demographics, symptoms, affected body parts, and clinical metadata (5,033 cases, 57 features)
- `dataset_scin_labels.csv` - Dermatologist-assigned diagnoses, confidence scores, and skin type assessments (5,033 cases, 17 features)

### Key Features
**Demographics:**
- Age groups, sex at birth
- Fitzpatrick skin types (self-reported and dermatologist-assessed)
- Monk skin tone scales (India and US)
- Race and ethnicity information

**Clinical Characteristics:**
- Skin texture (raised/bumpy, flat, rough/flaky, fluid-filled)
- Affected body parts (head/neck, arms, legs, torso, etc.)
- Condition symptoms (itching, pain, bleeding, darkening, etc.)
- Other systemic symptoms (fever, fatigue, joint pain, etc.)
- Condition duration and category

**Target Variables:**
- Weighted skin condition labels (dermatologist consensus)
- Skin condition confidence scores
- Image gradability assessments

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Data loading and merging
- Missing value analysis
- Distribution analysis of demographics and clinical features
- Visualization of key patterns and relationships
- Understanding class distributions and imbalances

### 2. Data Preparation & Feature Engineering
- Handling missing values (imputation strategies)
- Feature encoding (one-hot encoding, label encoding)
- Feature scaling and normalization
- Feature selection based on importance
- Addressing class imbalance (if needed)
- Train-test split with stratification

### 3. Modeling
- Baseline model establishment
- Multiple algorithm comparison:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting (XGBoost, LightGBM)
  - Support Vector Machines
  - Neural Networks (if applicable)
- Cross-validation strategies

### 4. Model Evaluation & Hyperparameter Tuning
- Performance metrics:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrices
  - ROC-AUC curves (for multi-class)
  - Classification reports
- Hyperparameter optimization:
  - Grid Search
  - Random Search
  - Bayesian Optimization (optional)
- Feature importance analysis
- Error analysis

### 5. Model Selection & Final Evaluation
- Compare models based on multiple metrics
- Select best performing model
- Final evaluation on test set
- Model interpretability and insights
- Discussion of limitations and future improvements

## Repository Structure

```
Scin-Data-Modeling/
├── README.md
├── model.ipynb              # Main Jupyter notebook with complete workflow
├── data/
│   ├── dataset_scin_cases.csv
│   └── dataset_scin_labels.csv
└── images/                  # (Referenced in dataset, if applicable)
```

## Technical Requirements

### Python Libraries
- **Data Manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn, xgboost, lightgbm
- **Statistical Analysis:** scipy, statsmodels
- **Utilities:** ast, collections

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm scipy
```

## Usage

1. Clone the repository
2. Ensure all required libraries are installed
3. Open `model.ipynb` in Jupyter Notebook or JupyterLab
4. Run cells sequentially to:
   - Load and explore the data
   - Prepare features
   - Train models
   - Evaluate performance
   - Select final model

## Key Findings (To be updated)

This section will be updated with:
- Best performing model and its metrics
- Important features for prediction
- Insights about skin condition classification
- Model limitations and considerations

## Contributors

INSY 674 Group Project Team

## License

This project is for educational purposes as part of INSY 674 coursework.

## Acknowledgments

- SCIN (Skin Condition Image Network) dataset providers
- Course instructors and teaching assistants

---

*Last Updated: February 2026*
