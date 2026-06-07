from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "names_gender.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "gender_name_clf.joblib"


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"name", "gender"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["name", "gender"]).copy()
    df["name"] = df["name"].astype(str).str.strip().str.lower()
    df["gender"] = df["gender"].astype(str).str.strip().str.lower()
    df = df[df["name"] != ""]
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def train() -> None:
    df = load_data(DATA_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        df["name"],
        df["gender"],
        test_size=0.2,
        random_state=42,
        stratify=df["gender"],
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
