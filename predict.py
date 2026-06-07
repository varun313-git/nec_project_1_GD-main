import argparse
from pathlib import Path

import joblib


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "gender_name_clf.joblib"


def predict_gender(name: str) -> tuple[str, float]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Train first using: python src/train.py"
        )

    model = joblib.load(MODEL_PATH)
    clean_name = name.strip().lower()
    if not clean_name:
        raise ValueError("Name cannot be empty.")

    pred = model.predict([clean_name])[0]
    probas = model.predict_proba([clean_name])[0]
    classes = list(model.classes_)
    confidence = float(probas[classes.index(pred)])
    return pred, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict gender from name")
    parser.add_argument("name", type=str, help="Name to classify")
    args = parser.parse_args()

    gender, confidence = predict_gender(args.name)
    print(f"Name: {args.name}")
    print(f"Predicted gender: {gender}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
