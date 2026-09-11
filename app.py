from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import shap

app = Flask(__name__)

# ── Load model and columns ────────────────────────────────────────────────────
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError(
        "model.pkl not found. Make sure it's in the same folder as app.py")

try:
    with open("columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError(
        "columns.pkl not found. Make sure it's in the same folder as app.py")

# Build SHAP explainer once at startup (slow to init, fast per request after)
explainer = shap.TreeExplainer(model)

# ── Categorical options (from the adult dataset) ──────────────────────────────
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
    "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
]

RELATIONSHIP_OPTIONS = [
    "Wife", "Own-child", "Husband", "Not-in-family",
    "Other-relative", "Unmarried"
]

RACE_OPTIONS = [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
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
    "Hong", "Honduras", "Scotland", "Outlying-US(Guam-USVI-etc)", "Puerto-Rico"
]

EDUCATION_NUM_MAP = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8,
    "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
    "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
    "Prof-school": 15, "Doctorate": 16
}

# Human-readable labels for numeric features
FEATURE_LABELS = {
    "age": "Age",
    "educational-num": "Education Level",
    "hours-per-week": "Hours / Week",
    "capital-gain": "Capital Gain",
    "capital-loss": "Capital Loss",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
ONE_HOT_PREFIXES = [
    "gender", "occupation", "marital_status", "relp-status",
    "race", "country", "workclass",
]


def friendly_name(col):
    """Convert a raw column name to a readable label.
    e.g. 'occupation_Sales'        → 'Occupation: Sales'
         'marital_status_Divorced' → 'Marital Status: Divorced'
         'age'                     → 'Age'
    """
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]

    # Match against known one-hot prefixes first, since prefixes like
    # "marital_status" contain an underscore themselves and would
    # otherwise be split incorrectly.
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
    """Build a zero-filled feature dict from the request payload,
    then set the relevant numeric and one-hot fields."""
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
        # If the column isn't found it just stays 0 (unknown category)

    set_ohe("gender",        data["gender"])
    set_ohe("occupation",    data["occupation"])
    set_ohe("marital_status", data["marital_status"])
    set_ohe("relp-status",   data["relationship"])
    set_ohe("race",          data["race"])
    set_ohe("country",       data["native_country"])
    set_ohe("workclass",     data["workclass"])

    return row


def compute_shap(input_df):
    """Run SHAP and return the top-8 features with direction + magnitude.
    Handles both old shap (list output) and new shap (3-D array output).
    """
    shap_values = explainer.shap_values(input_df)

    # Older shap  → list of 2 arrays, one per class
    # Newer shap  → single 3-D array of shape (rows, features, classes)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]       # class 1, row 0
    else:
        sv = shap_values[0, :, 1]    # row 0, all features, class 1

    # Pair feature names with their SHAP values, drop near-zero noise
    pairs = [
        (friendly_name(col), float(val))
        for col, val in zip(model_columns, sv)
        if abs(val) > 0.001
    ]

    # Sort by absolute impact descending, keep top 8
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top = pairs[:8]

    if not top:
        return []

    max_abs = max(abs(v) for _, v in top)

    return [
        {
            "feature":   name,
            "value":     round(val, 4),
            "pct":       round(abs(val) / max_abs * 100, 1),
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
            return jsonify({"error": "No JSON body received."}), 400

        # Validate required fields
        required = [
            "age", "gender", "race", "native_country", "relationship",
            "marital_status", "workclass", "occupation",
            "educational_num", "hours_per_week",
        ]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        # Build feature vector
        row = build_row(data)
        input_df = pd.DataFrame([row])

        # Prediction + probabilities
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        # SHAP explanation
        shap_result = compute_shap(input_df)

        return jsonify({
            "prediction": int(prediction),
            "label":      ">50K" if prediction == 1 else "<=50K",
            "confidence": round(float(max(proba)) * 100, 1),
            "prob_high":  round(float(proba[1]) * 100, 1),
            "prob_low":   round(float(proba[0]) * 100, 1),
            "shap":       shap_result,
        })

    except KeyError as e:
        return jsonify({"error": f"Missing expected field: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
