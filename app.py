from pathlib import Path

from flask import Flask, jsonify, render_template, request

from predict import predict_gender


BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or request.form
    name = str(payload.get("name", "")).strip()

    if not name:
        return jsonify({"error": "Name is required."}), 400

    try:
        gender, confidence = predict_gender(name)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "name": name,
            "predicted_gender": gender,
            "confidence": round(confidence, 4),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
