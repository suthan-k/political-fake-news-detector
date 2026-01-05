# 📰 Political Fake News Detector (Hybrid AI Agent)

An **Explainable Hybrid AI Agent** designed to detect political disinformation. This system combines the precision of a fine-tuned Transformer model with the interpretability of Generative AI to provide not just a verdict, but a reasoning.

🔗 [**Live Demo App**](https://political-fake-news-detector-sutharsan.streamlit.app/)



## 🧠 System Architecture

This project implements a **Hybrid** approach to misinformation detection:

1. **Discriminative Layer (The Brain):**
   * **Model:** `DistilBERT` (Fine-tuned on the ISOT Dataset).
   * **Task:** Binary Classification (Real vs. Fake).
   * **Performance:** ~99% Accuracy on unseen test data.
   * **Preprocessing:** Robust pipeline removing scraper artifacts while preserving linguistic structure.

2. **Explainability Layer (The Evidence):**
   * **Method:** **LIME** (Local Interpretable Model-agnostic Explanations).
   * **Task:** Perturbs the input text to identify specific words/tokens driving the prediction.

3. **Generative Layer (The Voice):**
   * **Model:** **Google Gemini 2.5 Flash**.
   * **Task:** Synthesizes the prediction and LIME evidence into a human-readable narrative explanation.



## ✨ Key Features

* **Real-time Inference:** Analyzes news articles in seconds.
* **Stylistic Analysis:** Detects fake news based on writing style (sensationalism, informal grammar) rather than just fact-checking.
* **Progressive Disclosure:** Simple UI for general users, with deep technical metrics hidden for experts.
* **Leakage Prevention:** Trained on a rigorously cleaned dataset (removed headers, footers, and image metadata) to ensure the model learns semantic patterns, not metadata shortcuts.



## 🛠️ Installation & Local Setup

### Prerequisites
* Python 3.8+
* Git LFS (Large File Storage)
* A Google Gemini API Key (Free tier)

### 1. Clone the Repository
```bash
git clone https://github.com/suthan-k/political-fake-news-detector.git
cd political-fake-news-detector
```
### 2. Download the Model (Git LFS)

The model file (`model.safetensors`) is approximately **260MB**.
Ensure Git LFS pulls it correctly:

```bash
git lfs pull
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Secrets

Create a (`.streamlit/secrets.toml`) file in the root directory:

```bash
GOOGLE_API_KEY = "Your_Gemini_API_Key_Here"
```

### 5. Run the App

```bash
streamlit run app.py
```

## 📊 Methodology & Results

The model was trained on **11,280 political news articles** from the [**ISOT Fake News Dataset**](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/).

- **Training Split:** 80%
- **Test Split:** 20%
- **Optimization Techniques:**
  - Dynamic padding
  - Stratified sampling
  - 3 training epochs

### Evaluation Metrics

| Metric    | Score   |
|----------|---------|
| Accuracy | 99.78%  |
| F1 Score | 99.78%  |
| ROC AUC  | 1.0000  |

## 🔬 Research & Analysis

The full research process can be found in the [`notebooks/`](./notebooks) directory.

* **EDA - [01_ISOT_EDA.ipynb](./notebooks/01_ISOT_EDA.ipynb):** Initial analysis of the dataset.
* **Data Cleaning and Preparation - [02b_ISOT_Data_Prep_Clean.ipynb](./notebooks/02b_ISOT_Data_Prep_Clean.ipynb):** Data cleaning and preparation script for the models.
* **Logistics Regression with TF-IDF (Basline Model) - [03b_Baseline_Model_Clean.ipynb](./notebooks/03b_Baseline_Model_Clean.ipynb):** Training script for the baseline model.
* **DistilBERT Model (Advanced Model) - [04_DistilBERT_Finetuning.ipynb](./notebooks/04_DistilBERT_Finetuning.ipynb):** Training script for the advanced model.
* **Model Evaluation and Comparison - [05_Model_Evaluation.ipynb](./notebooks/05_Model_Evaluation.ipynb):** Performance charts and metrics.
* **Explainable AI (LIME Analysis) - [06_XAI_LIME_Agent.ipynb](./notebooks/06_XAI_LIME_Agent.ipynb):** Verification of model behavior using Local Interpretable Model-agnostic Explanations.

## ⚠️ Scientific Disclaimer

This tool detects **stylistic and linguistic patterns** associated with misinformation (e.g., emotional loading, lack of attribution), **not factual truth**.  
It should be used as an **assistive tool for media literacy**, not an absolute oracle of truth.



## 📜 License

This project is licensed under the [MIT License](LICENSE).

