from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import shap
from huggingface_hub import hf_hub_download

app = Flask(__name__)


# ── Load model from Hugging Face ──────────────────────────────────────────────
try:
    model_path = hf_hub_download(
        repo_id="IronMan0071/income-bracket-model",
        filename="model.pkl"
    )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

except Exception as e:
    raise RuntimeError(f"Could not load model.pkl from Hugging Face: {e}")


# ── Load model columns ────────────────────────────────────────────────────────
try:
    with open("columns.pkl", "rb") as f:
        model_columns = pickle.load(f)

except FileNotFoundError:
    raise RuntimeError(
        "columns.pkl not found. Make sure it's in the same folder as app.py"
    )


# ── Build SHAP explainer once at startup ──────────────────────────────────────
explainer = shap.TreeExplainer(model)


# ── Categorical options ───────────────────────────────────────────────────────
WORKCLASS_OPTIONS = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
]

MARITAL_STATUS_OPTIONS = [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse"
]

OCCUPATION_OPTIONS = [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"
]

RELATIONSHIP_OPTIONS = [
    "Wife", "Own-child", "Husband", "Not-in-family",
    "Other-relative", "Unmarried"
]

RACE_OPTIONS = [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
    "Other", "Black"
]

GENDER_OPTIONS = ["Male", "Female"]

COUNTRY_OPTIONS = [
    "United-States", "Cuba", "Jamaica", "India", "Mexico",
    "South", "Japan", "Greece", "England", "China",
    "El-Salvador", "Philippines", "Columbia", "Iran", "Cambodia",
    "Canada", "Germany", "Taiwan", "Hungary", "Italy",
    "France", "Portugal", "Vietnam", "Ecuador", "Peru",
    "Thailand", "Laos", "Haiti", "Nicaragua", "Dominican-Republic",
    "Guatemala", "Yugoslavia", "Poland", "Ireland", "Trinadad&Tobago",
    "Hong", "Honduras", "Scotland",
    "Outlying-US(Guam-USVI-etc)", "Puerto-Rico"
]

EDUCATION_NUM_MAP = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
}


# ── Human-readable labels ────────────────────────────────────────────────────
FEATURE_LABELS = {
    "age": "Age",
    "educational-num": "Education Level",
    "hours-per-week": "Hours / Week",
    "capital-gain": "Capital Gain",
    "capital-loss": "Capital Loss",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
ONE_HOT_PREFIXES = [
    "gender",
    "occupation",
    "marital_status",
    "relp-status",
    "race",
    "country",
    "workclass",
]


def friendly_name(col):
    """Convert raw column names into readable labels."""

    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]

    for prefix in ONE_HOT_PREFIXES:
        if col.startswith(prefix + "_"):
            value = col[len(prefix) + 1:]
            label = prefix.replace("_", " ").replace("-", " ").title()

            return f"{label}: {value.replace('-', ' ')}"

    if "_" in col:
        prefix, value = col.split("_", 1)
        prefix = prefix.replace("-", " ").title()

        return f"{prefix}: {value.replace('-', ' ')}"

    return col.replace("-", " ").title()


def build_row(data):
    """Build the model input feature dictionary."""

    row = {col: 0 for col in model_columns}

    # Numeric fields
    row["age"] = int(data["age"])
    row["educational-num"] = int(data["educational_num"])
    row["hours-per-week"] = int(data["hours_per_week"])
    row["capital-gain"] = int(data.get("capital_gain", 0))
    row["capital-loss"] = int(data.get("capital_loss", 0))

    # One-hot encoded fields
    def set_ohe(prefix, value):
        col = f"{prefix}_{value}"

        if col in row:
            row[col] = 1

    set_ohe("gender", data["gender"])
    set_ohe("occupation", data["occupation"])
    set_ohe("marital_status", data["marital_status"])
    set_ohe("relp-status", data["relationship"])
    set_ohe("race", data["race"])
    set_ohe("country", data["native_country"])
    set_ohe("workclass", data["workclass"])

    return row


def compute_shap(input_df):
    """Return top 8 SHAP features with direction and magnitude."""

    shap_values = explainer.shap_values(input_df)

    # Older SHAP versions
    if isinstance(shap_values, list):
        sv = shap_values[1][0]

    # Newer SHAP versions
    else:
        sv = shap_values[0, :, 1]

    pairs = [
        (friendly_name(col), float(val))
        for col, val in zip(model_columns, sv)
        if abs(val) > 0.001
    ]

    pairs.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top = pairs[:8]

    if not top:
        return []

    max_abs = max(abs(v) for _, v in top)

    return [
        {
            "feature": name,
            "value": round(val, 4),
            "pct": round(abs(val) / max_abs * 100, 1),
            "direction": "up" if val > 0 else "down",
        }
        for name, val in top
    ]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        workclass_options=WORKCLASS_OPTIONS,
        marital_options=MARITAL_STATUS_OPTIONS,
        occupation_options=OCCUPATION_OPTIONS,
        relationship_options=RELATIONSHIP_OPTIONS,
        race_options=RACE_OPTIONS,
        gender_options=GENDER_OPTIONS,
        country_options=COUNTRY_OPTIONS,
        education_map=EDUCATION_NUM_MAP,
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON body received."
            }), 400

        required = [
            "age",
            "gender",
            "race",
            "native_country",
            "relationship",
            "marital_status",
            "workclass",
            "occupation",
            "educational_num",
            "hours_per_week",
        ]

        missing = [
            f for f in required
            if f not in data
        ]

        if missing:
            return jsonify({
                "error": f"Missing fields: {', '.join(missing)}"
            }), 400

        # Build input
        row = build_row(data)
        input_df = pd.DataFrame([row])

        # Prediction
        prediction = model.predict(input_df)[0]

        # Probabilities
        proba = model.predict_proba(input_df)[0]

        # SHAP explanation
        shap_result = compute_shap(input_df)

        return jsonify({
            "prediction": int(prediction),
            "label": ">50K" if prediction == 1 else "<=50K",
            "confidence": round(float(max(proba)) * 100, 1),
            "prob_high": round(float(proba[1]) * 100, 1),
            "prob_low": round(float(proba[0]) * 100, 1),
            "shap": shap_result,
        })

    except KeyError as e:
        return jsonify({
            "error": f"Missing expected field: {e}"
        }), 400

    except ValueError as e:
        return jsonify({
            "error": f"Invalid value: {e}"
        }), 400

    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
