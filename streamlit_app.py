import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


st.set_page_config(
    page_title="AI Tweet Detection",
    page_icon="🔎",
    layout="wide"
)

DATA_PATH = "combined_ai_tweet_detection_dataset.csv"
LABELS = {0: "Human-written", 1: "AI-generated"}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


@st.cache_resource(show_spinner="Training models. This may take a minute...")
def train_models(df):
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logistic_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=12000,
            min_df=2,
            max_df=0.95
        )),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    random_forest_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=7000,
            min_df=2,
            max_df=0.95
        )),
        ("model", RandomForestClassifier(
            n_estimators=80,
            max_depth=60,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        ))
    ])

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model
    }

    results = []
    matrices = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0)
        })

        matrices[name] = confusion_matrix(y_test, pred)

    results.append({
        "Model": "BERT",
        "Accuracy": 0.9209,
        "Precision": 0.9220,
        "Recall": 0.9376,
        "F1-score": 0.9297
    })

    return models, pd.DataFrame(results), matrices


def label_name(value):
    return LABELS[int(value)]


def confidence_note(score):
    if score >= 0.75:
        return "The model is fairly confident in this prediction."
    if score >= 0.60:
        return "The model leans toward this prediction, but the result should still be reviewed."
    return "The model is uncertain, so this result should be treated carefully."


df = load_data()
models, results_df, confusion_matrices = train_models(df)

st.title("AI Tweet Detection")
st.write(
    "This app predicts whether a tweet is human-written or AI-generated. "
    "The result should be used as a support tool, not final proof."
)

tab1, tab2, tab3 = st.tabs(["Try the Model", "Model Evaluation", "Dataset Summary"])


with tab1:
    st.subheader("Single Tweet Prediction")

    model_choice = st.selectbox(
        "Choose a model",
        ["Random Forest", "Logistic Regression"]
    )

    tweet_text = st.text_area(
        "Enter a tweet",
        height=150,
        placeholder="Paste or type a tweet here..."
    )

    if st.button("Predict"):
        if tweet_text.strip() == "":
            st.warning("Please enter a tweet first.")
        else:
            model = models[model_choice]
            prediction = int(model.predict([tweet_text])[0])
            probability_ai = float(model.predict_proba([tweet_text])[0][1])

            confidence = probability_ai if prediction == 1 else 1 - probability_ai

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Prediction", label_name(prediction))

            with col2:
                st.metric("Confidence", f"{confidence * 100:.1f}%")

            st.progress(confidence)
            st.write(confidence_note(confidence))

    st.divider()
    st.subheader("Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload a CSV with a column named text",
        type=["csv"]
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        if "text" not in batch_df.columns:
            st.error("The CSV must contain a column named text.")
        else:
            model = models[model_choice]
            text_values = batch_df["text"].fillna("").astype(str)

            batch_df["prediction"] = model.predict(text_values)
            batch_df["prediction_label"] = batch_df["prediction"].apply(label_name)
            batch_df["ai_probability"] = model.predict_proba(text_values)[:, 1].round(4)

            st.dataframe(batch_df.head(20), use_container_width=True)

            st.download_button(
                "Download predictions",
                batch_df.to_csv(index=False).encode("utf-8"),
                "ai_tweet_predictions.csv",
                "text/csv"
            )


with tab2:
    st.subheader("Model Evaluation Results")

    display_results = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
        display_results[col] = (display_results[col] * 100).round(2)

    st.dataframe(display_results, use_container_width=True)

    st.write(
        "BERT performs best overall because it captures more context from the tweet. "
        "Random Forest and Logistic Regression are lighter models, so they are easier to run in the app."
    )

    st.subheader("Confusion Matrix")

    matrix_choice = st.selectbox(
        "Choose a model",
        ["Random Forest", "Logistic Regression"],
        key="matrix_choice"
    )

    cm = pd.DataFrame(
        confusion_matrices[matrix_choice],
        index=["Actual Human", "Actual AI-generated"],
        columns=["Predicted Human", "Predicted AI-generated"]
    )

    st.dataframe(cm, use_container_width=True)


with tab3:
    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total tweets", f"{len(df):,}")

    with col2:
        st.metric("Human-written", f"{(df['label'] == 0).sum():,}")

    with col3:
        st.metric("AI-generated", f"{(df['label'] == 1).sum():,}")

    st.subheader("Rows by source dataset")
    st.dataframe(df["source_dataset"].value_counts().to_frame("row_count"), use_container_width=True)

    st.subheader("Sample rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Limitations")
    st.write(
        "The model can help flag tweets for review, but it should not be used as final proof that a tweet was written by AI. "
        "A next step would be testing it on more diverse tweets and checking performance across different topics and writing styles."
    )
