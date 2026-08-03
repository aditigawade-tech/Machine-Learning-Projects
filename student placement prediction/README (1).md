# 🎓 Student Placement Predictor

## 📌 Project Overview
The **Student Placement Predictor** is a Machine Learning project that predicts whether a student is likely to be placed based on academic performance and aptitude-related features.

## 🚀 Features
- Predicts whether a student will be **Placed** or **Not Placed**.
- Data preprocessing and model training.
- Train/test split.
- Model comparison.
- Save model using Pickle.

## 🛠️ Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Pickle
- Jupyter Notebook

## 📂 Project Structure
```text
Student-Placement-Predictor/
├── student-placement-predictor.ipynb
├── students_placement.csv
├── model.pkl
└── README.md
```

## 📊 Workflow
1. Import libraries.
2. Load dataset.
3. Preprocess data.
4. Split train/test.
5. Train classification models.
6. Evaluate accuracy.
7. Save best model.

## 📈 Algorithms
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

## 📦 Install
```bash
pip install numpy pandas scikit-learn
```

## ▶️ Run
```bash
jupyter notebook
```

## 💾 Save Model
```python
import pickle
pickle.dump(model, open("model.pkl","wb"))
```

## 🔮 Future Improvements
- Flask/Streamlit deployment
- Hyperparameter tuning
- Cloud deployment

## 👩‍💻 Author
**Aditi Gawade**
AI & DS Engineering Student
