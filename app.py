import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection")
st.write(
    "Enter a news headline or article. The machine-learning model predicts whether "
    "the text resembles the examples it learned from."
)
st.warning(
    "Educational demo: the included dataset is small and illustrative. "
    "This model does not independently verify facts and may be wrong."
)

@st.cache_resource
def train_model():
    data = pd.read_csv("dataset.csv")
    required_columns = {"text", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError("dataset.csv must contain columns named 'text' and 'label'.")

    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[data["label"].isin(["real", "fake"])]

    if len(data) < 4 or data["label"].nunique() < 2:
        raise ValueError("dataset.csv needs at least 4 rows and both real and fake labels.")

    # A pipeline makes sure training and prediction use the same TF-IDF conversion.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Use a holdout set only when each class has enough examples.
    counts = data["label"].value_counts()
    if len(data) >= 10 and counts.min() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            data["text"], data["label"], test_size=0.25,
            random_state=42, stratify=data["label"]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    else:
        model.fit(data["text"], data["label"])
        accuracy = None

    # Refit on all available examples for the final demo predictor.
    model.fit(data["text"], data["label"])
    return model, data, accuracy

try:
    model, data, accuracy = train_model()
except Exception as error:
    st.error(f"Could not load/train the model: {error}")
    st.stop()

col1, col2 = st.columns(2)
col1.metric("Training examples", len(data))
col2.metric("Labels", data["label"].nunique())

if accuracy is not None:
    st.caption(f"Illustrative holdout accuracy before final refit: {accuracy:.1%}. "
               "With a small dataset, this number is not a reliable measure of real-world performance.")

news_text = st.text_area(
    "Paste news text here",
    height=180,
    placeholder="Type or paste a headline or news article..."
)

if st.button("Analyze News", type="primary", use_container_width=True):
    if not news_text.strip():
        st.info("Please enter a headline or article first.")
    else:
        prediction = model.predict([news_text])[0]
        probabilities = model.predict_proba([news_text])[0]
        classes = list(model.named_steps["classifier"].classes_)
        confidence = float(probabilities[classes.index(prediction)])

        if prediction == "fake":
            st.error("Prediction: LIKELY FAKE")
        else:
            st.success("Prediction: LIKELY REAL")

        st.write(f"Model confidence for this label: **{confidence:.1%}**")
        st.caption(
            "Confidence is the model's estimated probability, not proof that the news is true."
        )

with st.expander("How does this project work?"):
    st.markdown("""
    1. **Dataset:** examples are labeled `real` or `fake`.
    2. **TF-IDF:** turns words into numerical features based on how informative they are.
    3. **Logistic Regression:** learns patterns associated with each label.
    4. **Prediction:** the model assigns one of the two labels to new text.
    """)

with st.expander("Dataset preview"):
    st.dataframe(data.head(10), use_container_width=True)
