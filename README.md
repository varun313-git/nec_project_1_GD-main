# Gender Detection Using ML Classification

This project predicts gender from a given name using a machine learning classifier in Python.

## Project Structure

- `data/names_gender.csv` - Generated dataset with `name,gender`
- `src/train.py` - Trains the classifier and saves the model
- `src/predict.py` - Loads the saved model and predicts gender for a name
- `src/app.py` - Flask app with a JSON prediction endpoint and web UI
- `src/templates/index.html` - Interactive form for testing predictions
- `src/static/` - Styles and client-side JavaScript
- `models/gender_name_clf.joblib` - Saved model file (generated after training)
- `requirements.txt` - Python dependencies

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the web app

After training the model, start the Flask app:

```bash
python src/app.py
```

Open `http://127.0.0.1:5000` in your browser.

## API Endpoint

`POST /api/predict`

Example JSON body:

```json
{"name": "sneha"}
```

Example response:

```json
{"name": "sneha", "predicted_gender": "female", "confidence": 0.91}
```

## Train

```bash
python src/train.py
```

This prints classification metrics and saves the model to `models/gender_name_clf.joblib`.

## Predict

```bash
python src/predict.py "sneha"
python src/predict.py "jairam"
```

## Dataset format

CSV headers:

```csv
name,gender
jairam,male
sneha,female
```
You can append more names to improve model quality.
