# 🏠 House Price Prediction
-Developing and evaluating a regression model to predict house prices using the `E06_house price data less.csv` dataset.

## 📌 Objective

To develop a regression model and evaluate its performance (using at least one algorithm) to predict house prices based on property, location, and amenity features.

## 📂 Dataset

- **File:** `E06_house price data less.csv`
- **Rows:** 999
- **Columns:** 23 (before preprocessing)

### Key Features

| Category | Columns |
|---|---|
| Location | `State`, `City`, `Locality` |
| Property Info | `Property_Type`, `BHK`, `Size_in_SqFt`, `Year_Built`, `Floor_No`, `Total_Floors`, `Age_of_Property` |
| Amenities & Condition | `Furnished_Status`, `Parking_Space`, `Security`, `Amenities`, `Facing`, `Public_Transport_Accessibility` |
| Nearby Facilities | `Nearby_Schools`, `Nearby_Hospitals` |
| Ownership | `Owner_Type`, `Availability_Status` |
| Target | `Price_in_Lakhs` (also `Price_per_SqFt`) |

The dataset had **no missing values** across any column.

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:**
  - `pandas`, `numpy` — data handling
  - `matplotlib`, `seaborn` — visualization
  - `scikit-learn` — preprocessing, modeling, and evaluation
    - `OrdinalEncoder`, `DictVectorizer`
    - `LinearRegression`, `DecisionTreeRegressor`
    - `train_test_split`, `GridSearchCV`
    - `mean_absolute_error`, `mean_squared_error`, `r2_score`

## 🔄 Workflow

### 1. Exploratory Data Analysis (EDA)
- Inspected shape, data types, null values, and summary statistics (`df.info()`, `df.describe()`, `df.isnull().sum()`)

### 2. Data Preprocessing
- Dropped irrelevant columns: `ID`, `Locality`
- Applied **Ordinal Encoding** to ordered categorical features:
  - `Property_Type` → `Apartment < Independent House < Villa`
  - `Furnished_Status` → `Unfurnished < Semi-furnished < Furnished`
  - `Public_Transport_Accessibility` → `Low < Medium < High`
  - `Facing` → `South < East < West < North`
  - `Security` → `No < Yes`
- Split remaining columns into categorical and numerical groups
- Normalized text casing (lower-cased categorical strings) for consistency
- Generated a **correlation heatmap** to examine relationships between numerical features
- Randomly sampled 40% of the data to reduce training time (limited compute)
- Split into train/test sets (80/20)
- Used **`DictVectorizer`** to one-hot encode remaining categorical columns

### 3. Model Training & Evaluation

Two regression models were trained and compared:

| Model | MAE | MSE | RMSE | R² Score |
|---|---|---|---|---|
| Linear Regression | 118.77 | 21,936.66 | 148.11 | 0.031 |
| Decision Tree Regressor | — | — | — | **0.916** |

> Metrics computed using `mean_absolute_error`, `mean_squared_error`, and `r2_score` from scikit-learn.

## 📊 Results & Conclusion

Both **Linear Regression** and **Decision Tree Regressor** were trained and evaluated to predict house prices. Linear Regression offered a simple, interpretable baseline but struggled to capture the non-linear patterns in the data (R² ≈ 0.03). The **Decision Tree Regressor performed significantly better** (R² ≈ 0.92), indicating that house price relationships in this dataset are largely non-linear and better captured by tree-based models.

## 🚀 How to Run

1. Clone/download this repository and ensure `E06_house price data less.csv` is in the same directory as the notebook.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook E06_House_Price_Prediction.ipynb
   ```
4. Run all cells sequentially — from data loading through EDA, preprocessing, and model training/evaluation.

## 📁 Project Structure

```
├── E06_House_Price_Prediction.ipynb   # Main notebook
├── E06_house price data less.csv      # Dataset (not included — add your own)
└── README.md                          # Project documentation
```

## 🔮 Future Improvements

- Hyperparameter tuning with `GridSearchCV` (already imported but unused)
- Try ensemble models (Random Forest, Gradient Boosting, XGBoost)
- Feature engineering (e.g., price per amenity score, location-based clustering)
- Cross-validation for more robust performance estimates
- Use the full dataset instead of a 40% sample, if compute allows

---
*Part of a regression modeling experiment series.*
