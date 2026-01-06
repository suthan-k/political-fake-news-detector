import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import re
import matplotlib.pyplot as plt
import google.generativeai as genai
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import os
import time

# ============================================================
# Streamlit Configuration
# ============================================================
st.set_page_config(
    page_title="Political Fake News Detector", layout="wide", page_icon="📰"
)

# ============================================================
# Path Setup
# ============================================================
MODEL_PATH = "./models/distilbert_finetuned_cased"

if not os.path.exists(MODEL_PATH):
    st.error(f"Critical Error: Model directory not found at {MODEL_PATH}")
    st.info("Ensure you have downloaded the model files into the 'models' folder.")
    st.stop()


# ============================================================
# 1. Model Loading (Cached)
# ============================================================
@st.cache_resource
def load_classifier():
    """
    Loads tokenizer and model once and caches them.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


tokenizer, model, device = load_classifier()


# ============================================================
# Cache LIME Explainer (Performance Optimization)
# ============================================================
@st.cache_resource
def load_lime_explainer():
    return LimeTextExplainer(
        class_names=["Fake News", "Real News"],
        bow=False,  # Better alignment with transformer behavior
    )


lime_explainer = load_lime_explainer()


# ============================================================
# 2. Preprocessing (Agent Input Pipeline)
# ============================================================
def agent_preprocess_pipeline(text):
    """
    Minimal preprocessing consistent with DistilBERT training.
    This protects the model's input contract.
    """
    if not isinstance(text, str):
        return ""

    # Remove Reuters-style dateline
    text = re.sub(
        r"^[A-Z\s\/,.\-]+\s*\(reuters\)\s*[—-]*\s*", " ", text, flags=re.IGNORECASE
    )

    # --- Remove standalone Reuters headers ---
    text = re.sub(r"^\(reuters\)\s*[—-]\s*", " ", text, flags=re.IGNORECASE)

    # Remove timestamp artifacts
    text = re.sub(r"\[\d+\s+est\].*$", " ", text, flags=re.IGNORECASE)

    # Remove common boilerplate endings
    text = re.sub(
        r"(?:via|read more):\s*[A-Za-z0-9\s\.]+$", " ", text, flags=re.IGNORECASE
    )

    text = re.sub(
        r"(read more:)?\s*(featured image|image|photo)\s*(by|via)\s*.*$",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove image credit sources
    text = re.sub(
        r"\b(getty images?|ap images?|afp|stringer)\b", " ", text, flags=re.IGNORECASE
    )

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"bit\.ly/\S+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 3. LIME Predictor (Model Wrapper)
# ============================================================
def lime_predictor(texts):
    """
    LIME-compatible prediction function.
    IMPORTANT: Applies SAME preprocessing as main inference.
    """
    processed_texts = [agent_preprocess_pipeline(t) for t in texts]

    inputs = tokenizer(
        processed_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)

    return probs.cpu().numpy()


# ============================================================
# 4. LLM Narrative Explainer (Gemini)
# ============================================================
def get_llm_explanation(api_key, article_snippet, label, confidence, lime_features):
    if not api_key:
        return "API key missing. Narrative explanation unavailable."

    genai.configure(api_key=api_key)

    # Include weights to give LLM better context
    features_text = ", ".join([f"'{w}' ({v:.2f})" for w, v in lime_features])

    prompt = f"""
You are an expert Disinformation Analyst.

IMPORTANT RULES:
- Do NOT claim the article is factually true or false.
- Do NOT invent facts.
- Explain ONLY stylistic, linguistic, or attributional signals.

AI VERDICT: {label} ({confidence:.1%})
STYLISTIC SIGNALS (Word & Importance): {features_text}

TEXT SNIPPET:
\"{article_snippet}...\"

TASK:
Explain why the Model classified this article as {label}
based ONLY on writing style, tone, and attribution cues. Use the signals as evidence.
"""

    models_to_try = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-pro"]

    for model_name in models_to_try:
        try:
            llm = genai.GenerativeModel(model_name)
            response = llm.generate_content(prompt)
            return response.text
        except Exception:
            time.sleep(1)  # Quick pause before retry
            continue

    return "⚠️ Narrative explanation currently unavailable (API Error)."


# ============================================================
# 5. UI Layout
# ============================================================
# Sidebar Configuration
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

# Main Title & Description
st.title("📰 Political Fake News Detector")
st.markdown("Paste a news article below to analyze.")

# Input Area
article = st.text_area(
    "Input Article:",
    height=200,
    label_visibility="collapsed",
    placeholder="Paste article text here...",
)

# Centered Analyze Button
_, col_btn, _ = st.columns([2, 1, 2])
with col_btn:
    analyze_clicked = st.button("Analyze Article", type="primary")


# ============================================================
# 6. Inference & Explanation Logic
# ============================================================
if analyze_clicked:
    if not article.strip():
        st.warning("Please enter text to analyze.")
    elif not model:
        st.error("Model failed to load.")
    else:
        with st.spinner("Analyzing content & style..."):

            # --- A. Preprocessing & Prediction ---
            clean_text = agent_preprocess_pipeline(article)

            inputs = tokenizer(
                clean_text, return_tensors="pt", truncation=True, max_length=512
            ).to(device)

            with torch.no_grad():
                probs = F.softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]

            fake_prob, real_prob = probs
            is_real = real_prob > 0.5
            verdict_text = "Real News" if is_real else "Fake News"

            # Clamp confidence for cleaner UI (avoids 100.0%)
            conf = min(max(max(fake_prob, real_prob), 0.01), 0.99)

            # --- B. LIME Explanation ---
            # Pass CLEAN text to LIME for faithfulness
            exp = lime_explainer.explain_instance(
                clean_text,
                lime_predictor,
                num_features=10,
                num_samples=100,  # High samples for stability
            )

            # --- C. LLM Narrative ---
            explanation = get_llm_explanation(
                api_key, clean_text[:500], verdict_text, conf, exp.as_list()
            )

        # ====================================================
        # 7. Results Display
        # ====================================================
        st.markdown("---")

        # --- Verdict Section ---
        st.subheader("⚖️ Prediction")

        st.metric(
            label="Result",
            value=verdict_text,
            delta=f"{conf:.1%} Confidence",
            delta_color="normal" if is_real else "inverse",
            label_visibility="collapsed",
        )

        # Scientific Disclaimer
        st.caption(
            "⚠️ This system detects stylistic and linguistic patterns, "
            "not factual truth. Predictions reflect learned writing cues "
            "from the training data."
        )

        # --- Explanation Section ---
        st.markdown("---")
        st.subheader("📝 Explanation")
        st.write(explanation)

        # --- Technical Deep Dive ---
        with st.expander("🔍 Click to view technical evidence (Influential Words)"):
            st.markdown(
                "The chart below shows which specific words pushed the model's decision."
            )

            fig = exp.as_pyplot_figure()
            fig.set_size_inches(6, 3)  # Compact size
            ax = fig.gca()
            ax.set_title("Top 10 Words Influencing the Decision")
            st.pyplot(fig, use_container_width=False)

            st.caption(
                f"Raw Probabilities → Real: {real_prob:.4f} | Fake: {fake_prob:.4f}"
            )
