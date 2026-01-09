# Customer Purchase Prediction 🛍️

## Project Overview

This project builds a machine learning classification model to predict whether an online store visitor will make a purchase (**Revenue**) based on their browsing behavior and demographic data.

## 🎯 Objective

To identify the key factors driving online purchases and build a predictive model to classify visitors as "Buyers" or "Non-Buyers".

## 📊 Dataset

Used the **Online Shoppers Purchasing Intention Dataset**.

- **12,330** sessions
- **18** features including:
  - `PageValues` (Key predictor)
  - `BounceRates`, `ExitRates`
  - `ProductRelated_Duration`
  - `Month`, `VisitorType`

## 🛠️ Workflow

1.  **Data Preprocessing**:
    - Handled missing values (dropped duplicates).
    - Encoded categorical variables (One-Hot Encoding).
    - Scaled numerical features using `StandardScaler`.
2.  **Exploratory Data Analysis (EDA)**:
    - Visualized correlations (Heatmap).
    - Analyzed distributions and behavior patterns.
3.  **Model Building**:
    - **Logistic Regression** (Baseline).
    - **Random Forest Classifier** (Ensemble).
4.  **Evaluation**:
    - Metrics: Accuracy, Precision, Recall, F1-Score.
    - Confusion Matrix visualization.

## 🏆 Key Results

| Model               | Accuracy  | F1-Score |
| :------------------ | :-------- | :------- |
| Logistic Regression | 89.1%     | 0.51     |
| **Random Forest**   | **90.5%** | **0.63** |

**Insight**: The **Random Forest** model performed better, successfully capturing complex patterns in user behavior. `PageValues` was identified as the strongest indicator of purchase intention.

## 🚀 How to Run

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/Xzone2486/Customer-purchase-prediction.git
    cd Customer-purchase-prediction
    ```

2.  **Set up Virtual Environment**:

    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Analysis**:
    ```bash
    python analysis.py
    ```
    _Or open `analysis.ipynb` in Jupyter Notebook._

## 📂 Project Structure

- `analysis.py`: Main python script for end-to-end analysis.
- `analysis.ipynb`: Interactive notebook with step-by-step documentation.
- `online_shoppers_intention.csv`: Dataset file.
- `*.png`: Generated plots (Confusion Matrices, EDA).

## 👤 Author

Ansh Kumar Prasad

## 📸 Project Visualizations

### Model Performance

|  **Random Forest Confusion Matrix**   |  **Logistic Regression Confusion Matrix**   |
| :-----------------------------------: | :-----------------------------------------: |
| ![RF CM](images/cm_Random_Forest.png) | ![LR CM](images/cm_Logistic_Regression.png) |

### Key Insights

**Correlation Heatmap**
![Correlation Heatmap](images/eda_heatmap.png)

**Purchase Patterns**
| PageValues vs Revenue | ExitRates vs Revenue |
|:---:|:---:|
| ![PageValues](images/eda_pattern_PageValues_vs_Revenue.png) | ![ExitRates](images/eda_pattern_ExitRates_vs_Revenue.png) |

### Feature Distributions

<details>
<summary>Click to view Distribution Plots</summary>

|                                                                  |                                                                |
| :--------------------------------------------------------------: | :------------------------------------------------------------: |
|            ![Bounce](images/eda_dist_BounceRates.png)            |             ![Exit](images/eda_dist_ExitRates.png)             |
|          ![PageValues](images/eda_dist_PageValues.png)           | ![Admin Duration](images/eda_dist_Administrative_Duration.png) |
| ![Product Duration](images/eda_dist_ProductRelated_Duration.png) |                                                                |

</details>
