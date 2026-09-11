# Income Bracket Estimator

A full-stack web app that serves a Random Forest model to predict whether a person earns above or below $50K/year, with a Flask backend and a custom-designed front end. Every prediction ships with a SHAP-based breakdown of the top 8 features that drove it, shown with direction and magnitude, so the result is explainable rather than a black box.

## Stack

- **Backend:** Flask, scikit-learn (Random Forest), SHAP (`TreeExplainer`)
- **Frontend:** Single-page HTML/CSS/JS template (no framework, no build step)

## Project Structure

```
income_app/
├── app.py              ← Flask backend + SHAP explanation logic
├── templates/
│   └── index.html      ← Front-end form + results UI
├── model.pkl            ← Trained Random Forest model
├── columns.pkl           ← Feature columns the model expects
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000**.

## How it works

1. The form collects every feature the model was trained on: age, workclass, education, occupation, marital status, relationship, race, gender, native country, capital gain/loss, and hours/week.
2. On submit, the frontend `POST`s JSON to `/predict`.
3. The backend reconstructs the one-hot-encoded feature vector to match `columns.pkl`, runs `model.predict` / `model.predict_proba`, and runs `shap.TreeExplainer` on the same row.
4. The response returns the predicted label, confidence, class probabilities, and the top 8 SHAP features (name, value, % impact, direction).
5. The UI renders this as a probability gauge (threshold at 50%) and a set of factor bars — green for features pushing the prediction toward >$50K, red for features pulling it down.

## Notes on this version

- Added the missing `templates/index.html` (referenced in the original structure but not present) — built from scratch to match every field `app.py` expects.
- Fixed a display bug in `friendly_name()`: columns like `marital_status_Divorced` were being split on the first underscore only, producing labels like "Marital: status_Divorced". It now matches against the known one-hot prefixes first, so it correctly reads "Marital Status: Divorced".
- The SHAP compatibility handling already in `app.py` (list output vs. 3-D array output, depending on `shap` version) was tested against the currently installed `shap` version and works as written — untouched.
- Model was trained with scikit-learn 1.6.1; a newer scikit-learn will load it with a harmless version-mismatch warning. Pin `scikit-learn==1.6.1` in `requirements.txt` if you want to silence that.

## Known limitation

`/predict` returns `500` (not `400`) if the request body isn't valid JSON, since the generic exception handler catches Flask's JSON-decode error along with everything else. Doesn't affect the UI (which always sends valid JSON) — only matters if you're hitting the API directly with malformed input.
