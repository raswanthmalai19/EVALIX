# EVALIX 🛡️

**Intelligent Credit Risk Assessment System**

A comprehensive machine learning platform for accurate credit risk prediction with explainable AI, fairness analysis, and interactive decision support.

---

## ✨ Features

- **Advanced ML Models**: XGBoost, Random Forest, Gradient Boosting, Logistic Regression, and SVM
- **Hyperparameter Optimization**: Automated tuning for maximum model performance
- **Explainability (SHAP)**: Understand model decisions with SHAP force plots and feature importance
- **Fairness Analysis**: Monitor and ensure fair lending decisions across demographic groups
- **Model Calibration**: Reliable probability estimates for better decision thresholds
- **Counterfactual Explanations (DiCE)**: Generate "what-if" scenarios for loan decisions
- **Interactive Web Dashboard**: User-friendly interface for real-time assessments
- **Production Ready**: FastAPI backend with REST API endpoints

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/raswanthmalai19/EVALIX.git
cd EVALIX

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

**Backend API:**
```bash
cd credit-risk-assessment
python app.py
# API available at http://localhost:8000
```

**Frontend Dashboard:**
```bash
cd pillar3_frontend
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

---

## 📁 Project Structure

```
EVALIX/
├── data/                          # Loan datasets
│   ├── Loan_Prediction.csv
│   └── Loan_Prediction_Realistic.csv
├── notebooks/                     # Jupyter analysis & development
│   ├── 01_Preprocessing.ipynb
│   ├── 02_Model_Building.ipynb
│   ├── 03_Hyperparameter_Tuning.ipynb
│   ├── 04_SHAP_Explainability.ipynb
│   ├── 05_DiCE_Counterfactuals.ipynb
│   ├── 06_Fairness_Summary.ipynb
│   └── 07_Model_Calibration.ipynb
├── src/                           # Core Python modules
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_utils.py
│   ├── evaluation.py
│   ├── calibration.py
│   └── augment_data.py
├── models/                        # Trained model files
├── outputs/                       # Results & visualizations
├── pillar3_frontend/              # Interactive web dashboard
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── app.py                         # FastAPI backend
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, Python 3.8+ |
| **ML/Data** | scikit-learn, XGBoost, pandas, numpy |
| **Explainability** | SHAP, DiCE |
| **Fairness** | Custom fairness metrics |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Deployment** | Uvicorn |

---

## 📊 Workflow

1. **Data Preprocessing** → Clean and prepare loan data
2. **Feature Engineering** → Extract meaningful features
3. **Model Selection** → Train multiple algorithms
4. **Hyperparameter Tuning** → Optimize model parameters
5. **Explainability** → Understand predictions with SHAP
6. **Fairness Analysis** → Ensure unbiased decisions
7. **Model Calibration** → Improve probability estimates
8. **Deployment** → Serve via REST API

---

## 📈 Model Performance

The system evaluates multiple models using:
- **Accuracy, Precision, Recall, F1-Score**
- **ROC-AUC, Confusion Matrix**
- **Calibration Curves**
- **Feature Importance Analysis**

See `outputs/model_comparison.csv` for detailed results.

---

## 🔐 Fairness & Ethics

EVALIX prioritizes fair lending practices:
- Demographic parity analysis
- Equalized odds checking
- Calibration across groups
- Bias mitigation strategies

Detailed findings in `notebooks/06_Fairness_Summary.ipynb`

---

## 📖 Documentation

Comprehensive documentation available in:
- `documentation.md` - Detailed technical guide
- `notebooks/` - Step-by-step analysis notebooks
- Inline code comments and docstrings

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## 👨‍💼 Author

**Raswanth Malai** - [@raswanthmalai19](https://github.com/raswanthmalai19)

---

## 🌟 Show Your Support

If you find this project useful, please give it a ⭐️

---

**Last Updated**: March 2026  
**Status**: Active Development
