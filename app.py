import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)


data = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"))
model = joblib.load(os.path.join(MODEL_DIR, "heart_model.pkl"))
rf_model = joblib.load(os.path.join(MODEL_DIR, "heart_rf_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "heart_scaler.pkl"))
columns = joblib.load(os.path.join(MODEL_DIR, "heart_columns.pkl"))
numerical_cols = joblib.load(os.path.join(MODEL_DIR, "heart_numerical_cols.pkl"))
comparison = pd.read_csv(os.path.join(MODEL_DIR, "heart_model_comparison.csv"))

ch_mean = data.loc[data["Cholesterol"] != 0, "Cholesterol"].mean()
bp_mean = data.loc[data["RestingBP"] != 0, "RestingBP"].mean()

data_clean = data.copy()
data_clean["Cholesterol"] = data_clean["Cholesterol"].replace(0, ch_mean).round(2)
data_clean["RestingBP"] = data_clean["RestingBP"].replace(0, bp_mean).round(2)
data_encode = pd.get_dummies(data_clean, drop_first=True).astype(int)

X_full = data_encode[columns].copy()
y_full = data_encode["HeartDisease"]
X_full[numerical_cols] = scaler.transform(X_full[numerical_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X_full,
    y_full,
    test_size=0.2,
    random_state=42,
    stratify=y_full
)

pred_test = model.predict(X_test)
proba_test = model.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, pred_test).tolist()
fpr, tpr, _ = roc_curve(y_test, proba_test)
roc_auc = roc_auc_score(y_test, proba_test)


def metric_row():
    row = comparison.loc[comparison["ROC AUC"].idxmax()]
    return {
        "model": str(row["Model"]),
        "accuracy": float(row["Accuracy"]),
        "precision": float(row["Precision"]),
        "recall": float(row["Recall"]),
        "f1": float(row["F1 Score"]),
        "roc_auc": float(row["ROC AUC"])
    }


def make_input(values):
    input_dict = {
        "Age": values["age"],
        "RestingBP": values["resting_bp"],
        "Cholesterol": values["cholesterol"],
        "FastingBS": values["fasting_bs"],
        "MaxHR": values["max_hr"],
        "Oldpeak": values["oldpeak"],
        "Sex_M": 1 if values["sex"] == "M" else 0,
        "ChestPainType_ATA": 1 if values["chest_pain"] == "ATA" else 0,
        "ChestPainType_NAP": 1 if values["chest_pain"] == "NAP" else 0,
        "ChestPainType_TA": 1 if values["chest_pain"] == "TA" else 0,
        "RestingECG_Normal": 1 if values["resting_ecg"] == "Normal" else 0,
        "RestingECG_ST": 1 if values["resting_ecg"] == "ST" else 0,
        "ExerciseAngina_Y": 1 if values["exercise_angina"] == "Y" else 0,
        "ST_Slope_Flat": 1 if values["st_slope"] == "Flat" else 0,
        "ST_Slope_Up": 1 if values["st_slope"] == "Up" else 0
    }
    input_df = pd.DataFrame([input_dict])[columns]
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    return input_df


@app.route("/")
def home():
    return render_template(
        "index.html",
        page="home",
        metrics=metric_row(),
        prediction=None,
        data_rows=int(data.shape[0]),
        feature_count=int(len(columns)),
        model_count=int(len(comparison)),
        model_name=type(model).__name__
    )


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()

    if not query:
        return render_template(
            "index.html",
            page="search",
            query=query,
            results=[],
            result_count=0
        )

    search_data = data.copy()
    row_text = search_data.astype(str).agg(" ".join, axis=1).str.lower()
    terms = query.lower().split()
    mask = pd.Series(True, index=search_data.index)

    for term in terms:
        mask = mask & row_text.str.contains(term, regex=False, na=False)

    results = search_data.loc[mask].copy()
    results.insert(0, "Record", results.index + 1)

    return render_template(
        "index.html",
        page="search",
        query=query,
        results=results.to_dict(orient="records"),
        result_count=len(results)
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {
            "age": int(request.form["age"]),
            "sex": request.form["sex"],
            "chest_pain": request.form["chest_pain"],
            "resting_bp": int(request.form["resting_bp"]),
            "cholesterol": int(request.form["cholesterol"]),
            "fasting_bs": int(request.form["fasting_bs"]),
            "resting_ecg": request.form["resting_ecg"],
            "max_hr": int(request.form["max_hr"]),
            "exercise_angina": request.form["exercise_angina"],
            "oldpeak": float(request.form["oldpeak"]),
            "st_slope": request.form["st_slope"]
        }
        input_df = make_input(values)
        pred = int(model.predict(input_df)[0])
        proba = float(model.predict_proba(input_df)[0][1])
        prediction = {
            "class": pred,
            "probability": proba,
            "low_probability": 1 - proba
        }
    except Exception as e:
        prediction = {"error": str(e)}

    return render_template(
        "index.html",
        page="home",
        metrics=metric_row(),
        prediction=prediction,
        form=request.form,
        data_rows=int(data.shape[0]),
        feature_count=int(len(columns)),
        model_count=int(len(comparison)),
        model_name=type(model).__name__
    )


@app.route("/eda")
def eda():
    numeric = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    categorical = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]
    return render_template(
        "index.html",
        page="eda",
        numeric=numeric,
        categorical=categorical,
        data_preview=data.head().to_html(classes="data-table", index=False),
        rows=int(data.shape[0]),
        cols=int(data.shape[1]),
        missing=int(data.isna().sum().sum())
    )


@app.route("/model-comparison")
def model_comparison():
    records = comparison.round(4).to_dict(orient="records")
    return render_template(
        "index.html",
        page="models",
        comparison=records
    )


@app.route("/confusion-matrix")
def confusion():
    return render_template(
        "index.html",
        page="confusion",
        matrix=cm
    )


@app.route("/roc-curve")
def roc_curve_page():
    return render_template(
        "index.html",
        page="roc",
        fpr=[float(x) for x in fpr],
        tpr=[float(x) for x in tpr],
        auc=float(roc_auc)
    )


@app.route("/explainability")
def explainability():
    importance = pd.Series(
        rf_model.feature_importances_,
        index=columns
    ).sort_values(ascending=False)
    return render_template(
        "index.html",
        page="explain",
        importance=[
            {"name": str(name), "value": float(value)}
            for name, value in importance.items()
        ]
    )


@app.route("/api/eda-chart")
def eda_chart():
    feature = request.args.get("feature", "Age")
    allowed = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    if feature not in allowed:
        feature = "Age"
    values = data[feature].replace(0, np.nan).dropna().tolist()
    return jsonify({"feature": feature, "values": [float(x) for x in values]})


if __name__ == "__main__":
    app.run(debug=True)
