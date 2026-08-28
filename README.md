# HTML Login Form Detector

## 📌 Project Overview
This project aims to detect malicious login forms (phishing pages) by analyzing the raw HTML structural signatures of web pages. Instead of relying on visual rendering or text content—which attackers frequently obfuscate—this project uses machine learning to identify the underlying structural "fingerprint" of a login form.

By extracting features like HTML nesting depth, N-grams of tag sequences (e.g., `(div (input)`), and specific tag counts, we can train models to accurately distinguish between safe non-form pages and malicious login forms.

---

## 📊 Dataset & Model Selection Strategy
The dataset for this project consists of approximately 1,200 samples, which is considered a relatively small amount of data for complex machine learning tasks. 

Because of this small dataset size:
- **XGBoost** is powerful but susceptible to overfitting if not perfectly tuned.
- **Random Forest** has the advantage of being a simpler, highly robust ensemble model that inherently resists overfitting, making it an excellent and stable baseline for this project.

Both models have been trained and evaluated on an exact 1:1 validation split to ensure fair comparison.

---

## 📁 Project Structure

```text
html-login-form-detector/
│
├── data/                       # Contains raw and processed datasets (ignored in git)
├── notebooks/                  # Interactive Jupyter Notebooks for exploration & training
│   ├── 01_data_exploration.ipynb
│   ├── 02_xgboost_training.ipynb
│   ├── random_forest_training.ipynb
│   └── random_forest_guide.ipynb
│
├── src/                        # Reusable Python source code
│   ├── preprocessing/          # Scripts for parsing HTML and engineering features
│   ├── random_forest/          # Training and evaluation logic
│   └── requirements.txt        # Python dependencies
│
└── README.md                   # Project documentation
```

---

## 🚀 How to Use This Project

You can run this project either in **Google Colab** (recommended for zero setup) or **Locally**.

### Option A: Run in Google Colab (Recommended)
You do not need to download any data or clone the repository manually. Simply click the badges below to open the notebooks directly in your browser. The notebooks are pre-configured to clone the repository and set up the environment automatically.

- **Data Exploration:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Smyles019/html-login-form-detector/blob/main/notebooks/01_data_exploration.ipynb)
- **XGBoost Training:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Smyles019/html-login-form-detector/blob/main/notebooks/02_xgboost_training.ipynb)
- **Random Forest Training:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Smyles019/html-login-form-detector/blob/main/notebooks/random_forest_training.ipynb)

### Option B: Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/Smyles019/html-login-form-detector.git
cd html-login-form-detector
```

**2. Install dependencies:**
Make sure you have Python 3.8+ installed. It is recommended to use a virtual environment.
```bash
pip install -r src/requirements.txt
```

**3. Run the Notebooks:**
```bash
jupyter notebook
```
Navigate to the `notebooks/` directory and open the training files.

---

## 🧠 Features Engineered
To prevent data leakage, all tokenizers (TF-IDF/CountVectorizer) and Scalers (RobustScaler) are fitted **strictly on the training data** and merely applied to the validation data.

The models utilize two main types of features:
1. **Structural Tag N-Grams:** Sequences of 1 to 3 HTML tags to capture layout context.
2. **Manual Topology Metrics:** 
   - `input_to_div_ratio`: High concentration of inputs inside divs.
   - `form_count`: Total `<form>` tags.
   - Maximum nesting depth of the DOM tree.

## 📈 Evaluation
The models are evaluated using **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**. 
For phishing detection, **Recall** is prioritized to minimize the number of malicious forms that slip through undetected.
