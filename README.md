# Heart Disease Prediction and Patient Clustering using Machine Learning

A beginner-friendly Python project that applies **supervised learning (classification)** and **unsupervised learning (K-Means clustering)** to a heart disease dataset.

**Author:** Saniya BK

## Project Overview

| Part | Technique | What it does |
|---|---|---|
| 1 | Supervised learning: Decision Tree, Random Forest, KNN | Predicts whether a patient has heart disease from 13 medical features |
| 2 | Unsupervised learning: K-Means | Ignores the label and groups patients with similar health measurements |

## Dataset

- File: `heart.csv` (1,025 records, 13 features + `target`)
- Features include age, sex, chest pain type, blood pressure, cholesterol, maximum heart rate and more
- 723 duplicate rows were found and removed, leaving **302 unique patients**
- No missing values

> **Note:** This popular version of the dataset documents `target = 1` as heart disease, but some analysts report that the coding looks reversed. The models learn patterns either way, so the results are valid, but this project is a learning exercise and not medical advice.

## Steps Followed

1. Load data with Pandas
2. Clean data (missing values, duplicates)
3. Exploratory Data Analysis (count plots, histograms, box plots, correlation heatmap)
4. Train / test split (80% / 20%, stratified)
5. Train Decision Tree, Random Forest and KNN
6. Evaluate: accuracy, confusion matrix, precision, recall, F1-score, 5-fold cross-validation
7. K-Means: feature selection, `StandardScaler`, Elbow Method, silhouette score, cluster visualization (scatter plot and PCA)

## Results

**Classification**

| Model | Test accuracy | 5-fold CV accuracy |
|---|---|---|
| Decision Tree | 72.1% | 76.2% |
| Random Forest | 75.4% | 82.1% |
| KNN (k=5) | 78.7% | 81.1% |

The test set is small (61 patients), so cross-validation gives a more reliable comparison. Random Forest and KNN perform about equally well.

**K-Means clustering**

- 3 clusters chosen using the Elbow Method and silhouette scores
- Silhouette score: **0.223** (the clusters overlap, which is common with real medical data)

## Technologies Used

Python, Jupyter Notebook / Google Colab, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## How to Run

1. Download `Heart_ML_Project.ipynb` and `heart.csv` into the same folder
2. Install the libraries: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open the notebook in Jupyter or Google Colab (upload `heart.csv` in Colab) and choose **Run All**

## Files

- `Heart_ML_Project.ipynb`: full code with outputs and explanations
- `Heart_ML_Project.html`: the same notebook as a web page, for reading without Jupyter
- `heart.csv`: dataset
