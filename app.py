import os
import sys
import json
import logging
import tempfile
import pickle
import subprocess
import threading
import time
import shutil
import numpy as np
import pandas as pd
import shap
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import uuid
from flask import send_from_directory
from werkzeug.utils import secure_filename
from Preprocessing.Angiogram_Preprocessing.Angiogram_DICOM_KeyFrame_Extraction import process_angiogram
from Preprocessing.ECG_Preprocessing.ECG_Preprocessing_Functions import (
    process_single_ecg_image,
    TARGET_LEAD_LENGTH,
)

from Pipeline_Management.Metadata_Extraction_And_Tagging_System import (
    EncryptionManager,
    MetadataManager,
    PatientSessionManager,
    StorageManager,
    RetrievalManager,
)

#  FLASK APPLICATION


app = Flask(__name__)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = "Pipeline_Management/Patients"

#  GRAPH RISK PIPELINE  (model loading)


NEW_MODEL_DIR = os.path.join(APP_DIR, "Model", "models")


def _load_pkl(filename):
    path = os.path.join(NEW_MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


risk_model = _load_pkl("xgb_risk_model.pkl")
scaler_base = _load_pkl("scaler_base.pkl")
scaler_graph = _load_pkl("scaler_graph.pkl")
knn_graph = _load_pkl("knn_graph.pkl")
new_encoders = _load_pkl("encoders.pkl")
BASE_FEATURES = _load_pkl("feature_columns.pkl")
FINAL_FEATURES = _load_pkl("final_features.pkl")
degree_train = _load_pkl("degree_train.pkl")
clustering_train = _load_pkl("clustering_train.pkl")
community_train = _load_pkl("community_train.pkl")
mlb = _load_pkl("mlb.pkl")
rec_multilabel_models = _load_pkl("rec_multilabel_models.pkl")
xgb_rec_label_model = _load_pkl("xgb_rec_label_model.pkl")

try:
    risk_explainer = shap.TreeExplainer(risk_model)
    logging.info("SHAP TreeExplainer initialised.")
except Exception as e:
    risk_explainer = None
    logging.warning(f"SHAP explainer could not be initialised: {e}")

SMOKER_SAFE_MAP = {"Yes": "Yes", "No": "No", "Former": "No"}

#
#  TEN-YEAR CHD MODEL
#

TEN_YEAR_MODEL_DIR = os.path.join(APP_DIR, "ten_year_models", "ten_year_models")

TEN_YEAR_FEATURES = [
    "age", "male", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "pulsePressure", "bp_ratio",
    "BMI", "heartRate", "glucose",
]

try:
    with open(os.path.join(TEN_YEAR_MODEL_DIR, "ten_year_model.pkl"), "rb") as f:
        ten_year_model = pickle.load(f)
    with open(os.path.join(TEN_YEAR_MODEL_DIR, "ten_year_threshold.pkl"), "rb") as f:
        ten_year_threshold = float(pickle.load(f))
    logging.info(f"Ten-year CHD model loaded (threshold={ten_year_threshold:.4f})")
except FileNotFoundError as e:
    ten_year_model, ten_year_threshold = None, 0.5
    logging.warning(f"Ten-year CHD model NOT loaded: {e}")
except Exception as e:
    ten_year_model, ten_year_threshold = None, 0.5
    logging.error(f"Failed to load ten-year CHD model: {e}")

#
#  FIHAM'S RECOMMENDATION ADVICE
#

RECOMMENDATION_ADVICE = {
    "Low_Risk": {
        "title": "Low Cardiovascular Risk",
        "icon": "fas fa-check-circle",
        "colour": "success",
        "priority": "Low",
        "advice": (
            "Your cardiovascular risk profile is currently low. "
            "Keep up your healthy lifestyle — small consistent habits compound into "
            "long-term heart health."
        ),
        "actions": [
            "Schedule an annual cardiovascular screening",
            "Maintain a balanced diet rich in fruits, vegetables and whole grains",
            "Keep up regular physical activity (≥150 min/week)",
            "Avoid tobacco and limit alcohol intake",
        ],
    },
    "High_Risk": {
        "title": "High Cardiovascular Risk",
        "icon": "fas fa-exclamation-triangle",
        "colour": "danger",
        "priority": "High",
        "advice": (
            "Your profile indicates elevated cardiovascular risk. "
            "Immediate lifestyle modification and prompt medical consultation are "
            "strongly advised to reduce your risk of a cardiac event."
        ),
        "actions": [
            "Consult your cardiologist promptly",
            "Monitor blood pressure and cholesterol regularly",
            "Strictly adhere to any prescribed medications",
            "Adopt a heart-healthy diet and increase physical activity",
        ],
    },
    "Smoking_Cessation": {
        "title": "Smoking Cessation",
        "icon": "fas fa-smoking-ban",
        "colour": "warning",
        "priority": "High",
        "advice": (
            "Smoking is a major, modifiable cardiovascular risk factor. "
            "Quitting can reduce your heart disease risk by up to 50% within just one year."
        ),
        "actions": [
            "Speak to your doctor about a personalised cessation plan",
            "Consider nicotine replacement therapy (patch, gum, or inhaler)",
            "Explore prescription medications such as varenicline or bupropion",
            "Avoid secondhand smoke exposure and identify your personal triggers",
        ],
    },
    "Diet_Cholesterol": {
        "title": "Dietary & Cholesterol Management",
        "icon": "fas fa-apple-alt",
        "colour": "warning",
        "priority": "Medium",
        "advice": (
            "Your cholesterol or dietary indicators suggest that targeted nutritional "
            "changes are needed to protect your cardiovascular health long-term."
        ),
        "actions": [
            "Reduce saturated fats (fatty meats, full-fat dairy) and eliminate trans fats",
            "Increase fibre through oats, legumes, fruits and vegetables",
            "Limit processed foods, fried foods and added sugars",
            "Consider a referral to a registered dietitian for a tailored meal plan",
        ],
    },
    "Exercise": {
        "title": "Increase Physical Activity",
        "icon": "fas fa-running",
        "colour": "info",
        "priority": "Medium",
        "advice": (
            "Regular physical activity is one of the most powerful tools to reduce "
            "cardiovascular risk. Even modest increases in activity levels yield "
            "meaningful benefits."
        ),
        "actions": [
            "Aim for at least 150 minutes of moderate-intensity aerobic activity per week",
            "Start with 30-minute brisk walks 5 days a week if currently sedentary",
            "Add strength/resistance training at least twice weekly",
            "Break up prolonged sitting with movement every 30–60 minutes",
        ],
    },
    "BP_Control": {
        "title": "Blood Pressure Control",
        "icon": "fas fa-tachometer-alt",
        "colour": "danger",
        "priority": "High",
        "advice": (
            "Your blood pressure readings indicate hypertension management is needed. "
            "Uncontrolled high blood pressure is a leading cause of heart disease, "
            "stroke and kidney damage."
        ),
        "actions": [
            "Monitor your blood pressure at home daily and log the readings",
            "Reduce sodium intake to under 2,300 mg/day (ideally 1,500 mg)",
            "Follow the DASH diet (rich in potassium, calcium and magnesium)",
            "Take prescribed antihypertensive medications consistently",
        ],
    },
    "Maintenance": {
        "title": "Health Maintenance",
        "icon": "fas fa-shield-alt",
        "colour": "primary",
        "priority": "Low",
        "advice": (
            "Your current health metrics are within healthy ranges. "
            "The goal now is to sustain and protect these gains over the long term."
        ),
        "actions": [
            "Continue regular medical check-ups (at least annually)",
            "Maintain a healthy weight and stable BMI",
            "Stay consistent with your current exercise routine",
            "Manage stress through mindfulness, adequate sleep and social connection",
        ],
    },
}

# Priority ordering for stable sort of recommendations
_PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}

#
#  DOCTOR CREDENTIALS
#


UPLOAD_FOLDER = Path("uploads")
OUTPUT_ROOT = Path("Preprocessed_Angiogram_Output")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_ROOT.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["OUTPUT_ROOT"] = str(OUTPUT_ROOT)

DEEPSA_PORT = int(os.environ.get("DEEPSA_PORT", 7860))
DEEPSA_URL = os.environ.get("DEEPSA_URL", f"http://127.0.0.1:{DEEPSA_PORT}")
DEEPSA_SCRIPT = os.path.join(APP_DIR, "demo.py")

_deepsa_proc = None
_deepsa_lock = threading.Lock()


def _deepsa_running() -> bool:
    import socket
    try:
        with socket.create_connection(("127.0.0.1", DEEPSA_PORT), timeout=1):
            return True
    except OSError:
        return False


def ensure_deepsa_running():
    global _deepsa_proc
    with _deepsa_lock:
        if _deepsa_running():
            return
        logging.info(f"Starting DeepSA from: {DEEPSA_SCRIPT}")
        _deepsa_proc = subprocess.Popen(
            [sys.executable, DEEPSA_SCRIPT],
            cwd=os.path.dirname(DEEPSA_SCRIPT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    deadline = time.time() + 30
    while time.time() < deadline:
        if _deepsa_running():
            return
        time.sleep(0.5)
    raise RuntimeError(
        "DeepSA (demo.py) failed to start within 30 seconds. "
        "Check that the model checkpoint exists and all dependencies are installed."
    )


#
#  GRAPH FEATURE PIPELINE
#

def _encode_safe(encoder, value, safe_map=None):
    if safe_map:
        value = safe_map.get(str(value), str(value))
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        logging.warning(f"LabelEncoder could not encode '{value}'; defaulting to 0.")
        return 0


def prepare_new_features(data: dict):
    """
    Build the three feature matrices the risk and recommendation models need.
    Graph features (degree, clustering, community) are estimated by finding
    the nearest neighbours of this patient in the training graph.
    """
    age = float(data.get("age", 45))
    bmi = float(data.get("bmi", 25))
    systolic = float(data.get("systolic_bp", 120))
    diastolic = float(data.get("diastolic_bp", 80))
    cholesterol = float(data.get("cholesterol", 200))
    glucose = float(data.get("glucose", 100))
    exercise_hrs = float(data.get("exercise_hours", 0))

    high_chol = 1 if cholesterol > 200 else 0
    high_glucose = 1 if glucose > 100 else 0
    hypertension = 1 if (systolic >= 140 or diastolic >= 90) else 0
    obesity = 1 if bmi >= 30 else 0

    gender_enc = _encode_safe(new_encoders["Gender"], data.get("gender", "Male"))
    smoker_enc = _encode_safe(new_encoders["Smoker"], data.get("smoker", "No"), SMOKER_SAFE_MAP)

    base_dict = {
        "Age": age,
        "Gender_Encoded": gender_enc,
        "BMI": bmi,
        "Systolic_BP": systolic,
        "Diastolic_BP": diastolic,
        "Cholesterol mg/dL": cholesterol,
        "Glucose mg/dL": glucose,
        "Smoker_Encoded": smoker_enc,
        "High_Cholesterol": high_chol,
        "High_Glucose": high_glucose,
        "Hypertension": hypertension,
        "Obesity": obesity,
        "Exercise hours/week": exercise_hrs,
    }

    X_base = pd.DataFrame([base_dict])[BASE_FEATURES]
    X_base_scaled = scaler_base.transform(X_base)

    distances, indices = knn_graph.kneighbors(X_base_scaled)
    nbr = indices[0]
    degree_val = float(np.mean(degree_train[nbr]))
    clustering_val = float(np.mean(clustering_train[nbr]))
    community_val = float(np.mean(community_train[nbr]))

    full_dict = base_dict.copy()
    full_dict["degree"] = degree_val
    full_dict["clustering"] = clustering_val
    full_dict["community"] = community_val

    X_full = pd.DataFrame([full_dict])[FINAL_FEATURES]
    X_full_scaled = scaler_graph.transform(X_full)

    return X_base, X_full, X_full_scaled


#
#  TEN-YEAR CHD HELPERS
#

def _build_ten_year_features(data: dict) -> pd.DataFrame:
    """
    Translate the app's internal patient-data dict into the Framingham
    feature DataFrame the ten-year model expects.
    """
    sys_bp = float(data.get("systolic_bp", 120))
    dia_bp = float(data.get("diastolic_bp", 80))
    smoker = str(data.get("smoker", "No"))

    is_current_smoker = 1 if smoker == "Yes" else 0

    cigs_per_day = 0.0
    if smoker in ("Yes", "Former"):
        cigs_per_day = float(
            data.get("cigs_per_day", data.get("cigsPerDay", 0)) or 0
        )

    hyp_flag = data.get("prevalent_hyp", "")
    if hyp_flag == "Yes":
        prevalent_hyp = 1
    elif hyp_flag == "No":
        prevalent_hyp = 0
    else:
        prevalent_hyp = 1 if (sys_bp >= 140 or dia_bp >= 90) else 0

    features = {
        "age": float(data.get("age", 45)),
        "male": 1 if str(data.get("gender", "")).strip().lower() == "male" else 0,
        "currentSmoker": is_current_smoker,
        "cigsPerDay": cigs_per_day,
        "BPMeds": 1 if data.get("bp_treatment") == "Yes" else 0,
        "prevalentStroke": 1 if data.get("previous_stroke") == "Yes" else 0,
        "prevalentHyp": prevalent_hyp,
        "diabetes": 1 if data.get("diabetes") == "Yes" else 0,
        "totChol": float(data.get("cholesterol", data.get("totChol", 200))),
        "sysBP": sys_bp,
        "diaBP": dia_bp,
        "pulsePressure": sys_bp - dia_bp,
        "bp_ratio": sys_bp / (dia_bp + 1e-6),
        "BMI": float(data.get("bmi", data.get("BMI", 25))),
        "heartRate": float(data.get("heart_rate", data.get("heartRate", 72)) or 72),
        "glucose": float(data.get("glucose", 100)),
    }
    return pd.DataFrame([features])[TEN_YEAR_FEATURES]


def _compute_ten_year(data: dict) -> dict:
    """
    Run the Framingham based ten-year CHD risk model.
    """
    if ten_year_model is None:
        return {"success": False, "message": "Ten-year CHD model is not loaded."}
    try:
        X = _build_ten_year_features(data)
        prob = float(ten_year_model.predict_proba(X)[0][1])
        pct = min(round(prob * 100, 1), 100.0)

        if prob < 0.40:
            category, colour = "LOW", "low"
        elif prob < 0.50:
            category, colour = "MEDIUM", "medium"
        else:
            category, colour = "HIGH", "high"

        advice = (
            "Your projected 10-year risk is low." if category == "LOW" else
            "Your projected 10-year risk is moderate. View personalised recommendations."
            if category == "MEDIUM" else
            "Your projected 10-year risk is high. Please view personalised recommendations."
        )
        return {
            "success": True, "percent": pct, "category": category,
            "colour": colour, "probability": round(prob, 4), "advice": advice,
        }
    except Exception as e:
        logging.exception("_compute_ten_year failed")
        return {"success": False, "message": str(e)}


#
#  RECOMMENDATION LOGIC
#

def _get_recommendations(data: dict) -> dict:
    """
    Predict personalised recommendation categories for a patient.
    """
    try:
        X_base, X_full, X_full_scaled = prepare_new_features(data)

        # ── Risk assessment ──────────────────────────────────────────────
        risk_prob = float(risk_model.predict_proba(X_full_scaled)[0][1])
        risk_class = "HIGH_RISK" if risk_prob >= 0.5 else "LOW_RISK"
        risk_pct = round(risk_prob * 100, 1)

        # ── SHAP top factors ─────────────────────────────────────────────
        shap_top_factors = []
        if risk_explainer is not None:
            try:
                shap_vals = risk_explainer.shap_values(X_full_scaled)
                if isinstance(shap_vals, list):
                    class1_impacts = shap_vals[1][0]
                elif shap_vals.ndim == 3:
                    class1_impacts = shap_vals[0, :, 1]
                else:
                    class1_impacts = shap_vals[0]

                contributions = sorted(
                    [
                        (FINAL_FEATURES[i], class1_impacts[i],
                         X_full.iloc[0][FINAL_FEATURES[i]])
                        for i in range(len(FINAL_FEATURES))
                    ],
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                for feature, impact, value in contributions[:5]:
                    display_value = value
                    if feature == "Gender_Encoded" and "Gender" in new_encoders:
                        try:
                            display_value = new_encoders["Gender"].inverse_transform([int(value)])[0]
                        except Exception:
                            pass
                    elif feature == "Smoker_Encoded" and "Smoker" in new_encoders:
                        try:
                            display_value = new_encoders["Smoker"].inverse_transform([int(value)])[0]
                        except Exception:
                            pass
                    shap_top_factors.append({
                        "feature": feature.replace("_Encoded", "").replace("_", " "),
                        "impact": round(float(impact), 3),
                        "direction": "increases" if impact > 0 else "decreases",
                        "value": display_value,
                    })
            except Exception as shap_err:
                logging.warning(f"SHAP computation in recommendations failed: {shap_err}")

        # ── Graph features projected onto this patient ────────────────────
        distances, indices = knn_graph.kneighbors(scaler_base.transform(X_base))
        nbr = indices[0]
        graph_info = {
            "degree": round(float(np.mean(degree_train[nbr])), 4),
            "clustering": round(float(np.mean(clustering_train[nbr])), 4),
            "community": round(float(np.mean(community_train[nbr])), 4),
        }

        # ── Stage 1: per-category binary classifiers ─────────────────────
        active_categories = []  # list of (category_key, probability)
        for category, clf in rec_multilabel_models.items():
            try:
                pred = int(clf.predict(X_full_scaled)[0])
                if pred == 1:
                    # Retrieve probability if the classifier supports it
                    prob_val = None
                    if hasattr(clf, "predict_proba"):
                        try:
                            prob_val = round(float(clf.predict_proba(X_full_scaled)[0][1]), 4)
                        except Exception:
                            pass
                    active_categories.append((category, prob_val))
            except Exception as clf_err:
                logging.warning(f"Classifier for '{category}' failed: {clf_err}")

        # ── Stage 2: XGBoost multi-label fallback ────────────────────────
        if not active_categories:
            try:
                label_pred = xgb_rec_label_model.predict(X_full_scaled)[0]
                decoded = mlb.inverse_transform(np.array([[label_pred]]))[0]
                if decoded:
                    # Try to get per-label probabilities from predict_proba
                    proba_dict = {}
                    if hasattr(xgb_rec_label_model, "predict_proba"):
                        try:
                            proba_row = xgb_rec_label_model.predict_proba(X_full_scaled)[0]
                            for idx, label in enumerate(mlb.classes_):
                                proba_dict[label] = round(float(proba_row[idx]), 4)
                        except Exception:
                            pass
                    active_categories = [
                        (cat, proba_dict.get(cat)) for cat in decoded
                    ]
                else:
                    active_categories = [("Maintenance", None)]
            except Exception as fb_err:
                logging.warning(f"XGBoost fallback failed: {fb_err}")
                active_categories = [("Maintenance", None)]

        # ── Stage 3: absolute safety net ─────────────────────────────────
        if not active_categories:
            active_categories = [("Maintenance", None)]

        # ── Build enriched recommendation objects ─────────────────────────
        recommendations = []
        for cat, prob_val in active_categories:
            if cat not in RECOMMENDATION_ADVICE:
                # Unknown category key — skip gracefully
                logging.warning(f"Category '{cat}' not in RECOMMENDATION_ADVICE; skipping.")
                continue

            advice_entry = RECOMMENDATION_ADVICE[cat]

            # Build rationale based on patient data for known categories
            rationale = _build_rationale(cat, data)

            rec = {
                "category": cat,
                "title": advice_entry["title"],
                "icon": advice_entry["icon"],
                "colour": advice_entry["colour"],
                "priority": advice_entry["priority"],
                "advice": advice_entry["advice"],
                "actions": advice_entry["actions"],
                "rationale": rationale,
                "probability": prob_val,
            }
            recommendations.append(rec)

        # Fall back to Maintenance if nothing mapped
        if not recommendations:
            advice_entry = RECOMMENDATION_ADVICE["Maintenance"]
            recommendations.append({
                "category": "Maintenance",
                "title": advice_entry["title"],
                "icon": advice_entry["icon"],
                "colour": advice_entry["colour"],
                "priority": advice_entry["priority"],
                "advice": advice_entry["advice"],
                "actions": advice_entry["actions"],
                "rationale": "Your overall health indicators are within acceptable ranges.",
                "probability": None,
            })

        # Sort by priority (High → Medium → Low)
        recommendations.sort(
            key=lambda r: _PRIORITY_ORDER.get(r.get("priority", "Low"), 2)
        )

        return {
            "success": True,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_assessment": {
                "risk_probability": round(risk_prob, 4),
                "risk_class": risk_class,
                "risk_percentage": risk_pct,
            },
            "recommendations": recommendations,
            "total": len(recommendations),
            "shap_top_factors": shap_top_factors,
            "graph_features": graph_info,
        }

    except Exception as e:
        logging.exception("_get_recommendations failed")
        return {"success": False, "message": str(e)}


def _build_rationale(category: str, data: dict) -> str:
    cholesterol = float(data.get("cholesterol", 200))
    systolic = float(data.get("systolic_bp", 120))
    diastolic = float(data.get("diastolic_bp", 80))
    bmi = float(data.get("bmi", 25))
    exercise_hrs = float(data.get("exercise_hours", 0))
    smoker = str(data.get("smoker", "No"))
    glucose = float(data.get("glucose", 100))

    rationale_map = {
        "High_Risk": (
            "Your overall risk profile indicates potential cardiovascular concerns "
            "requiring prompt medical attention."
        ),
        "Low_Risk": (
            "Your overall health indicators are within acceptable ranges."
        ),
        "Smoking_Cessation": (
            f"You are {'a current smoker' if smoker == 'Yes' else 'a former smoker'}, "
            "which significantly elevates cardiovascular risk."
        ),
        "Diet_Cholesterol": (
                f"Your cholesterol level ({cholesterol:.0f} mg/dL) "
                + ("is elevated above the 200 mg/dL threshold." if cholesterol > 200
                   else "warrants dietary vigilance.")
        ),
        "Exercise": (
            f"Your current exercise level ({exercise_hrs:.1f} hours/week) "
            "is below the recommended 2.5 hours of moderate-intensity activity."
            if exercise_hrs < 2.5 else
            "Maintaining and building on your physical activity is recommended."
        ),
        "BP_Control": (
                f"Your blood pressure ({systolic:.0f}/{diastolic:.0f} mmHg) "
                + ("is in the hypertensive range." if systolic >= 140 or diastolic >= 90
                   else "is approaching hypertensive levels and should be monitored.")
        ),
        "Maintenance": (
            "Your current health metrics are within healthy ranges. "
            "Sustaining your lifestyle habits is key."
        ),
    }
    return rationale_map.get(
        category,
        "This recommendation is based on your overall health screening profile."
    )


#
#  ECG INFERENCE
#

INFERENCE_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "process_02_cardiac_analysis", "ecg_risk_prediction",
    "inference", "predict_ecg_risk.py",
)


def run_ecg_inference(csv_path: str) -> dict:
    project_root = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isfile(INFERENCE_SCRIPT_PATH):
        msg = f"Inference script not found at: {INFERENCE_SCRIPT_PATH}"
        logging.error(msg)
        return {"error": msg, "prediction": "Error", "confidence": "0%",
                "risk_level": "UNKNOWN", "suspected_vessel": "N/A"}

    if not os.path.isfile(csv_path):
        msg = f"Pre-processed CSV not found at: {csv_path}"
        logging.error(msg)
        return {"error": msg, "prediction": "Error", "confidence": "0%",
                "risk_level": "UNKNOWN", "suspected_vessel": "N/A"}

    try:
        result = subprocess.run(
            [sys.executable, INFERENCE_SCRIPT_PATH, csv_path],
            capture_output=True, text=True, timeout=120, cwd=project_root,
        )

        if result.stdout:
            logging.info(f"Inference stdout:\n{result.stdout}")
        if result.stderr:
            logging.error(f"Inference stderr:\n{result.stderr}")

        if result.returncode != 0:
            stderr_clean = result.stderr.strip() or "No stderr captured."
            return {"error": stderr_clean, "prediction": "Error", "confidence": "0%",
                    "risk_level": "UNKNOWN", "suspected_vessel": "N/A"}

        prediction_result = {
            "prediction": "Unknown", "confidence": "0%",
            "risk_level": "UNKNOWN", "suspected_vessel": "N/A", "plot_path": None,
        }
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Status:"):
                prediction_result["prediction"] = line.split(":", 1)[1].strip()
            elif line.startswith("Confidence:"):
                prediction_result["confidence"] = line.split(":", 1)[1].strip()
            elif line.startswith("Emergency Priority:"):
                prediction_result["risk_level"] = line.split(":", 1)[1].strip()
            elif line.startswith("Artery Localization:"):
                prediction_result["suspected_vessel"] = line.split(":", 1)[1].strip()

        return prediction_result

    except subprocess.TimeoutExpired:
        return {"error": "Inference script timed out (>120s).", "prediction": "Timeout",
                "confidence": "0%", "risk_level": "UNKNOWN", "suspected_vessel": "N/A"}
    except Exception as e:
        return {"error": f"Failed to launch inference: {str(e)}", "prediction": "Error",
                "confidence": "0%", "risk_level": "UNKNOWN", "suspected_vessel": "N/A"}


#
#  ANGIOGRAM HELPERS
#

def _find_session_path(session_id: str):
    for patient_dir in Path(BASE_DIR).iterdir():
        if not patient_dir.is_dir():
            continue
        candidate = patient_dir / "sessions" / session_id
        if candidate.exists():
            return candidate
    return None


def _read_pipeline_metadata(preprocessed_folder: str) -> dict:
    try:
        path = Path(preprocessed_folder) / "metadata.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        logging.warning("Could not read pipeline metadata.json", exc_info=True)
    return {}


def _copy_frames_to_angio_folder(variants, preprocessed_folder, angio_folder, metadata_file):
    enriched = []
    for v in variants:
        src = next(Path(preprocessed_folder).rglob(v["filename"]), None)

        if not src:
            logging.warning(f"Frame not found: {v['filename']}")
            enriched.append({**v, "angio_folder_path": None})
            continue

        try:
            with open(src, "rb") as f:
                file_bytes = f.read()

            encrypted_path = StorageManager.save_encrypted_angiogram(
                metadata_file,
                file_bytes,
                v["filename"]
            )

            print("DEBUG → Encrypted path:", encrypted_path)
            dst = Path(encrypted_path)

        except Exception:
            logging.warning(f"Could not process frame {src}", exc_info=True)
            dst = None

        enriched.append({**v, "angio_folder_path": str(dst) if dst else None})
    return enriched


def _write_angiogram_frame_metadata(angio_folder: Path, payload: dict) -> Path:
    out_path = angio_folder / "angiogram_frame_metadata.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)
    logging.info(f"angiogram_frame_metadata.json written → {out_path}")
    return out_path


#
#  ROUTES General Patient
#

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/patient_login.html")
def patient_login():
    return render_template("patient_login.html")


@app.route("/patient_register.html")
def patient_register():
    return render_template("patient_register.html")


@app.route("/patient_dashboard.html")
def patient_dashboard():
    return render_template("patient_dashboard.html")


@app.route("/upload_ecg.html", methods=["GET"])
def upload_page():
    return render_template("upload_ecg.html")


@app.route("/pre_screening.html")
def screening():
    return render_template("pre_screening.html")


@app.route("/pre_screening_results.html")
def result():
    return render_template("pre_screening_results.html")


#
#  ROUTES Doctor
#

@app.route("/doctor_upload.html")
def doctor_upload():
    return render_template("doctor_upload.html")


@app.route("/doctor_analysis_results.html")
def doctor_analysis_results():
    return render_template("doctor_analysis_results.html")


#
#  ROUTES Doctor login
#

@app.route("/doctor_login.html")
@app.route("/doctor_login", methods=["GET"])
def doctor_login():
    session_id = request.args.get("session_id", "")
    patient_id = request.args.get("patient_id", "")
    return render_template("doctor_login.html", session_id=session_id,
                           patient_id=patient_id, error=None)


@app.route("/doctor_login", methods=["POST"])
def doctor_login_post():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    session_id = request.form.get("session_id", "")
    patient_id = request.form.get("patient_id", "")


#
#  ROUTE Angiogram upload (doctor portal)
#

@app.route("/upload_angiogram", methods=["GET"])
def angiogram_upload_page():
    session_id = request.args.get("session_id", "")
    patient_id = request.args.get("patient_id", "")
    return render_template("angiogram_processing.html",
                           session_id=session_id, patient_id=patient_id)


@app.route("/upload_angiogram", methods=["POST"])
def upload_angiogram():
    try:
        session_id = request.form.get("session_id", "").strip()
        patient_id = request.form.get("patient_id", "").strip()
        angio_type = request.form.get("angio_type", "unknown")
        doctor_notes = request.form.get("doctor_notes", "")
        file = request.files.get("angio_file")

        if not file or file.filename == "":
            return "No angiogram file uploaded.", 400

        session_path = _find_session_path(session_id)
        if not session_path:
            return (
                f"Session '{session_id}' not found. "
                f"Ensure the patient completed the ECG step first."
            ), 404

        angio_folder = session_path / "angiogram"
        angio_folder.mkdir(parents=True, exist_ok=True)
        metadata_file = session_path / "metadata.json"

        print("DEBUG → Metadata path:", metadata_file)
        print("DEBUG → Exists?", metadata_file.exists())

        raw_filename = secure_filename(file.filename) or "angiogram"
        file_bytes = file.read()

        encrypted_path = StorageManager.save_encrypted_angiogram(
            metadata_file, file_bytes, raw_filename
        )
        logging.info(f"Encrypted angiogram saved → {encrypted_path}")

        preprocessed_root = angio_folder / "preprocessed"
        pipeline_result = None
        variants = []
        output_directory = ""

        with tempfile.NamedTemporaryFile(
                suffix=Path(raw_filename).suffix, delete=False
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            pipeline_result = process_angiogram(tmp_path, output_root=str(preprocessed_root))
            output_directory = pipeline_result["output_directory"]
            variants = pipeline_result["variants"]
            logging.info(
                f"process_angiogram succeeded — {len(variants)} frames, "
                f"output: {output_directory}"
            )
        except Exception as pp_err:
            logging.error(f"process_angiogram failed: {pp_err}", exc_info=True)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        enriched_variants = []
        if variants and output_directory:
            enriched_variants = _copy_frames_to_angio_folder(
                variants, output_directory, angio_folder, metadata_file
            )
            print("DEBUG → Enriched variants:", enriched_variants)

        MetadataManager.update_angiogram_metadata(
            metadata_file,
            dicom_path=str(encrypted_path),
            selected_frame_paths=[
                v.get("angio_folder_path")
                for v in enriched_variants
                if v.get("angio_folder_path")
            ],
            segmentation_mask_path="",
        )

        metadata = MetadataManager.read_metadata(metadata_file)
        metadata["angiogram"].update({
            "angio_type": angio_type,
            "doctor_notes": doctor_notes,
            "uploaded_by": "doctor",
            "preprocessed_folder": output_directory,
            "variants": enriched_variants or variants,
            "selected_variant": None,
            "selected_image_path": None,
            "localization_result": None,
        })
        MetadataManager.write_metadata(metadata_file, metadata)
        logging.info(f"Session metadata updated for session '{session_id}'.")

        if not variants:
            return (
                "<h2>Angiogram uploaded but preprocessing failed.</h2>"
                "<p>The raw file has been saved. "
                "Please check server logs and retry preprocessing.</p>"
            ), 500

        return redirect(url_for("angiogram_select", session_id=session_id))

    except Exception as e:
        logging.exception("Angiogram upload failed")
        return f"Upload failed: {str(e)}", 500


#
#  ROUTE Serve one preprocessed variant image
#

@app.route("/angiogram_image/<session_id>/<filename>")
def angiogram_image(session_id, filename):
    session_path = _find_session_path(session_id)
    if not session_path:
        return "Session not found", 404

    direct_path = session_path / "angiogram" / filename
    if direct_path.exists():
        return send_file(str(direct_path), mimetype="image/png")

    try:
        metadata = MetadataManager.read_metadata(session_path / "metadata.json")
        preprocessed_folder = metadata.get("angiogram", {}).get("preprocessed_folder")
        if preprocessed_folder:
            pp_path = Path(preprocessed_folder) / filename
            if pp_path.exists():
                return send_file(str(pp_path), mimetype="image/png")
    except Exception:
        pass

    for img_path in (session_path / "angiogram").rglob(filename):
        return send_file(str(img_path), mimetype="image/png")

    return "Image not found", 404


#
#  ROUTE Variant selection page
#

@app.route("/angiogram_select/<session_id>")
def angiogram_select(session_id):
    try:
        session_path = _find_session_path(session_id)
        if not session_path:
            return "Session not found", 404

        metadata = MetadataManager.read_metadata(session_path / "metadata.json")
        patient_id = metadata.get("patient_id", "Unknown")
        variants = metadata.get("angiogram", {}).get("variants", [])

        if not variants:
            return "No preprocessed variants found. Please re-upload the angiogram.", 400

        for v in variants:
            v["url"] = url_for("angiogram_image",
                               session_id=session_id, filename=v["filename"])

        return render_template("angiogram_select.html",
                               session_id=session_id,
                               patient_id=patient_id,
                               variants=variants)
    except Exception as e:
        logging.exception("Error loading angiogram selection page")
        return f"Error: {str(e)}", 500


#
#  ROUTE Confirm selected variant then save frame then launch DeepSA
#

@app.route("/angiogram_confirm", methods=["POST"])
def angiogram_confirm():
    import urllib.parse
    try:
        session_id = request.form.get("session_id", "").strip()
        patient_id_form = request.form.get("patient_id", "").strip()
        selected_filename = request.form.get("selected_variant", "").strip()

        if not selected_filename:
            return "Missing selected_variant — no frame was selected.", 400

        use_standalone = bool(patient_id_form) and not session_id

        if session_id:
            session_path = _find_session_path(session_id)
            if not session_path:
                return f"Session '{session_id}' not found.", 404

            metadata_file = session_path / "metadata.json"
            metadata = MetadataManager.read_metadata(metadata_file)
            angio = metadata.get("angiogram", {})
            dicom_path = angio.get("dicom_path", "")
            preprocessed_folder = angio.get("preprocessed_folder", "")
            existing_mask = angio.get("segmentation", {}).get("mask_path", "")
            angio_folder = session_path / "angiogram"

            source_frame_path = angio_folder / selected_filename
            if not source_frame_path.exists() and preprocessed_folder:
                pp_candidate = Path(preprocessed_folder) / selected_filename
                if pp_candidate.exists():
                    source_frame_path = pp_candidate
            if not source_frame_path.exists():
                found = next(angio_folder.rglob(selected_filename), None)
                if found:
                    source_frame_path = found

            if not source_frame_path.exists():
                logging.error(
                    f"Selected frame '{selected_filename}' not found "
                    f"anywhere under {angio_folder}"
                )
                return f"Selected frame '{selected_filename}' could not be located.", 500

            saved_frame_path = angio_folder / f"selected_{selected_filename}"
            shutil.copy2(source_frame_path, saved_frame_path)
            logging.info(f"Selected frame saved → {saved_frame_path}")

            pipeline_meta = _read_pipeline_metadata(preprocessed_folder)
            variant_info = next(
                (v for v in pipeline_meta.get("variants", [])
                 if v.get("filename") == selected_filename),
                {}
            )
            frame_details = {
                "label": variant_info.get("label", selected_filename),
                "filename": selected_filename,
                "saved_frame_path": str(saved_frame_path),
                "source_path": str(source_frame_path),
                "selected_frame_indices": pipeline_meta.get("selected_frame_indices", []),
                "source": pipeline_meta.get("source", "Unknown"),
                "number_of_original_frames": pipeline_meta.get("number_of_original_frames", 0),
                "pipeline_patient_id": pipeline_meta.get("patient_id", ""),
            }
            all_variants = pipeline_meta.get("variants", angio.get("variants", []))

            _write_angiogram_frame_metadata(angio_folder, {
                "selected_filename": selected_filename,
                "saved_frame_path": str(saved_frame_path),
                "source_preprocessed_path": str(source_frame_path),
                "preprocessed_folder": preprocessed_folder,
                "frame_details": frame_details,
                "all_pipeline_variants": all_variants,
                "saved_at": datetime.utcnow().isoformat(),
            })

            MetadataManager.update_angiogram_metadata(
                metadata_file,
                dicom_path=dicom_path,
                selected_frame_paths=[str(saved_frame_path)],
                segmentation_mask_path=existing_mask,
            )

            MetadataManager.update_angiogram_selected_frame(
                metadata_file,
                selected_filename=selected_filename,
                saved_frame_path=str(saved_frame_path),
                frame_details=frame_details,
                all_variants=all_variants,
            )

            logging.info(
                f"Angiogram frame '{selected_filename}' fully persisted "
                f"for session '{session_id}'."
            )

        if use_standalone:
            meta_path = OUTPUT_ROOT / patient_id_form / "metadata.json"
            if meta_path.exists():
                try:
                    meta = MetadataManager.read_metadata(meta_path)
                    meta["selected_variant"] = selected_filename
                    MetadataManager.write_metadata(meta_path, meta)
                except Exception:
                    logging.warning("Could not update standalone metadata.", exc_info=True)

        ensure_deepsa_running()

        if use_standalone:
            flask_image_url = url_for(
                "serve_preprocessed_frame",
                patient_id=patient_id_form,
                filename=selected_filename,
                _external=True,
            )
        else:
            if not session_id:
                return "Could not determine flow: neither session_id nor patient_id provided.", 400
            flask_image_url = url_for(
                "angiogram_image",
                session_id=session_id,
                filename=selected_filename,
                _external=True,
            )

        encoded_url = urllib.parse.quote(flask_image_url, safe="")
        deepsa_redirect = f"http://127.0.0.1:{DEEPSA_PORT}/?image={encoded_url}"
        return redirect(deepsa_redirect)

    except RuntimeError as e:
        return (
            f"<h2>DeepSA could not be started</h2><p>{str(e)}</p>"
            f"<p>Ensure demo.py dependencies are installed and the model checkpoint exists.</p>"
        ), 500
    except Exception as e:
        logging.exception("Angiogram confirmation failed")
        return f"Confirmation failed: {str(e)}", 500


#
#  ROUTE ECG Upload Handler
#

@app.route("/upload_ecg", methods=["POST"])
def upload_ecg():
    try:
        patient_id = request.form["patient_id"]
        age = int(request.form["age"])
        gender = request.form["gender"]
        notes = request.form.get("notes", "")
        ecg_type = request.form.get("ecg_type", "unknown")
        file = request.files["ecg_file"]

        if not file:
            return "No file uploaded", 400

        filename = file.filename
        file_bytes = file.read()

        patient_data = {
            "patient_id": patient_id, "age": age, "gender": gender,
            "notes": notes, "ecg_type": ecg_type,
        }

        metadata_path = PatientSessionManager.initialize_patient_session(
            patient_data, base_dir=BASE_DIR)
        session_folder = metadata_path.parent

        ecg_folder = session_folder / "ecg"
        ecg_folder.mkdir(parents=True, exist_ok=True)

        encrypted_path = StorageManager.save_encrypted_ecg(metadata_path, file_bytes, filename)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        vector = process_single_ecg_image(tmp_path, target_len=TARGET_LEAD_LENGTH)
        if vector is None:
            raise ValueError("ECG preprocessing failed")

        csv_filename = f"{Path(filename).stem}_preprocessed.csv"
        csv_path = ecg_folder / csv_filename
        columns = [f"Lead{lead}_{i}" for lead in range(1, 13)
                   for i in range(TARGET_LEAD_LENGTH)]
        pd.DataFrame([vector], columns=columns).to_csv(csv_path, index=False)
        os.unlink(tmp_path)

        prediction_result = run_ecg_inference(str(csv_path))

        MetadataManager.update_ecg_metadata(
            metadata_path,
            raw_image_path=str(encrypted_path),
            processed_csv_path=str(csv_path),
            classification_result=json.dumps(prediction_result),
        )

        metadata = MetadataManager.read_metadata(metadata_path)
        metadata["ecg"]["classification_result"] = json.dumps(prediction_result)

        # ── Screening form data (only present when patient came via pre-screening) ──
        # The upload_ecg.html page injects this from localStorage before form submit.
        # If missing, this was a direct/doctor upload — screening data is simply absent.
        screening_data_raw = request.form.get("screening_form_data", "").strip()
        logging.info(f"screening_form_data received: {bool(screening_data_raw)} "
                     f"(length={len(screening_data_raw)})")

        if screening_data_raw:
            try:
                sfd = json.loads(screening_data_raw)

                # Ensure fields the 10-year model and recommendation pipeline need
                # have safe defaults if the pre-screening form didn't collect them.
                sfd.setdefault("heart_rate", 72)
                sfd.setdefault("cigs_per_day", 0)
                sfd.setdefault("bp_treatment", "No")
                sfd.setdefault("previous_stroke", "No")
                sfd.setdefault("diabetes", "No")

                # Derive prevalent_hyp from BP values if not explicitly provided
                sfd.setdefault(
                    "prevalent_hyp",
                    "Yes" if (float(sfd.get("systolic_bp", 0)) >= 140 or
                              float(sfd.get("diastolic_bp", 0)) >= 90) else "No"
                )

                metadata["screening_form_data"] = sfd
                logging.info(f"screening_form_data saved for session "
                             f"'{session_folder.name}' — keys: {list(sfd.keys())}")

            except json.JSONDecodeError as je:
                logging.warning(f"screening_form_data JSON decode failed: {je}")
            except Exception:
                logging.warning("Could not parse screening_form_data", exc_info=True)
        else:
            logging.info(
                f"No screening_form_data in request for session '{session_folder.name}' "
                f"— direct/doctor upload, 10-year risk and recommendations will be hidden."
            )

        MetadataManager.write_metadata(metadata_path, metadata)

        confidence_str = prediction_result.get("confidence", "0%")
        try:
            risk_pct = float(str(confidence_str).replace("%", "").strip())
        except (ValueError, AttributeError):
            risk_pct = 0.0

        MetadataManager.update_risk_prediction(
            metadata_path,
            prediction_result.get("risk_level", "UNKNOWN"),
            risk_pct,
        )

        return redirect(url_for("ecg_result", session_id=session_folder.name))

    except Exception as e:
        logging.exception("Upload + prediction failed")
        return f"Upload failed: {str(e)}", 500


#
#  ROUTE ECG Result Display
#

@app.route("/ecg_result/<session_id>")
def ecg_result(session_id):
    try:
        session_path = _find_session_path(session_id)
        if not session_path:
            return "Session not found", 404

        metadata_file = session_path / "metadata.json"
        if not metadata_file.exists():
            return "Metadata not found", 404

        metadata = MetadataManager.read_metadata(metadata_file)
        patient_id = metadata.get("patient_id", "Unknown")
        classification_raw = metadata.get("ecg", {}).get("classification_result")

        if not classification_raw:
            prediction = {"error": "No prediction data found"}
        else:
            try:
                prediction = (
                    json.loads(classification_raw)
                    if isinstance(classification_raw, str)
                    else classification_raw
                )
            except Exception as e:
                prediction = {"error": f"JSON parse error: {str(e)}"}

        angio_info = metadata.get("angiogram", {})
        angiogram_uploaded = bool(angio_info.get("uploaded"))
        angio_selected = angio_info.get("selected_variant") if angio_info else None
        loc_result = angio_info.get("localization_result") if angio_info else None

        plot_url = None
        plot_path = Path(APP_DIR) / "outputs" / "lead_activity_report.png"
        if plot_path.exists():
            plot_url = url_for("ecg_plot", session_id=session_id,
                               filename="lead_activity_report.png")

        patient_data = metadata.get("screening_form_data")
        # Only show 10-year risk if patient went through pre-screening
        # (screening_form_data is only written during the patient-facing upload_ecg route,
        #  never when a doctor uploads directly via the doctor portal)
        has_screening_data = bool(patient_data)

        ten_year_data = None
        if has_screening_data:
            ten_year_data = _compute_ten_year(patient_data)

        return render_template(
            "ecg_result.html",
            session_id=session_id,
            patient_id=patient_id,
            prediction=prediction,
            plot_url=plot_url,
            angiogram_uploaded=angiogram_uploaded,
            angio_selected=angio_selected,
            loc_result=loc_result,
            patient_data=patient_data,
            show_ten_year_risk=has_screening_data,
            ten_year_model_available=(ten_year_model is not None),
            ten_year_data=ten_year_data,
            has_screening_data=has_screening_data,
        )

    except Exception as e:
        logging.exception("Error displaying results")
        return f"Error displaying results: {str(e)}", 500


#  ROUTE Serve lead activity plot


@app.route("/ecg_plot/<session_id>/<filename>")
def ecg_plot(session_id, filename):
    plot_path = Path(APP_DIR) / "outputs" / filename
    if plot_path.exists():
        return send_file(str(plot_path), mimetype="image/png")

    session_path = _find_session_path(session_id)
    if not session_path:
        return "Session not found", 404

    fallback_path = session_path / "ecg" / filename
    if not fallback_path.exists():
        return "Plot not found", 404
    return send_file(str(fallback_path), mimetype="image/png")


#  ROUTE Pre-screening


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No JSON body received"}), 400

        required = [
            "age", "gender", "bmi", "systolic_bp", "diastolic_bp",
            "cholesterol", "glucose", "smoker", "exercise_hours",
            "heart_rate", "bp_treatment", "previous_stroke",
            "prevalent_hyp", "cigs_per_day",
        ]
        missing_fields = [f for f in required if f not in data]
        if missing_fields:
            return jsonify({
                "success": False,
                "message": f"Missing fields: {', '.join(missing_fields)}",
            }), 400

        X_base, X_full, X_full_scaled = prepare_new_features(data)
        prob = float(risk_model.predict_proba(X_full_scaled)[0][1])
        pred_class = int(risk_model.predict(X_full_scaled)[0])

        # SHAP top-3 contributing features
        top_factors = []
        if risk_explainer is not None:
            try:
                shap_values = risk_explainer.shap_values(X_full_scaled)
                if isinstance(shap_values, list):
                    class_1_impacts = shap_values[1][0]
                elif shap_values.ndim == 3:
                    class_1_impacts = shap_values[0, :, 1]
                else:
                    class_1_impacts = shap_values[0]
                contributions = sorted(
                    [(FINAL_FEATURES[i], class_1_impacts[i], X_full.iloc[0][FINAL_FEATURES[i]])
                     for i in range(len(FINAL_FEATURES))],
                    key=lambda x: abs(x[1]), reverse=True,
                )
                for feature, impact, value in contributions[:3]:
                    effect = "higher risk" if impact > 0 else "lower risk"
                    display_value = value
                    if feature == "Gender_Encoded" and "Gender" in new_encoders:
                        try:
                            display_value = new_encoders["Gender"].inverse_transform([int(value)])[0]
                        except Exception:
                            pass
                    elif feature == "Smoker_Encoded" and "Smoker" in new_encoders:
                        try:
                            display_value = new_encoders["Smoker"].inverse_transform([int(value)])[0]
                        except Exception:
                            pass
                    top_factors.append({
                        "factor": feature.replace("_Encoded", "").replace("_", " "),
                        "impact": round(float(impact), 3),
                        "value": str(display_value),
                        "effect": effect,
                    })
            except Exception as shap_err:
                logging.warning(f"SHAP computation failed: {shap_err}")

        # Ten-year risk
        ten_year_result = _compute_ten_year(data)

        base = {
            "success": True,
            "risk_probability": prob,
            "explanation": top_factors,
            "patient_data": data,
            "ten_year_result": ten_year_result,
            "ten_year_model_available": ten_year_model is not None,
        }

        if pred_class == 1:
            base.update({
                "risk_status": "HIGH_RISK",
                "decision": "MANDATORY_ECG",
                "message": "You show signs of elevated CAD risk. Please upload your ECG.",
                "next_step": "UPLOAD_ECG",
                "requires_ecg": True,
                "color": "red",
            })
        else:
            base.update({
                "risk_status": "LOW_RISK",
                "decision": "OPTIONAL_ECG",
                "message": "You do not currently show significant CAD risk.",
                "next_step": "OPTIONAL_UPLOAD",
                "requires_ecg": False,
                "color": "green",
            })

        return jsonify(base)

    except Exception as e:
        logging.exception("/predict failed")
        return jsonify({"success": False, "message": str(e)}), 500


#  ROUTE Standalone ten-year risk


@app.route("/predict_ten_year", methods=["POST"])
def predict_ten_year():
    if ten_year_model is None:
        return jsonify({"success": False, "message": "Ten-year model is not loaded."}), 503
    data = request.get_json() or {}
    result = _compute_ten_year(data)
    return jsonify(result), (200 if result.get("success") else 500)


#  ROUTE Recommendation API


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Accepts patient data as JSON, runs the full recommendation pipeline,
    and returns the enriched recommendations response.

    Expected JSON fields (same as /predict):
        age, gender, bmi, systolic_bp, diastolic_bp, cholesterol, glucose,
        smoker, exercise_hours  (all others have sensible defaults)

    Returns:
        {
          success, timestamp,
          risk_assessment: { risk_probability, risk_class, risk_percentage },
          recommendations: [ { category, title, icon, colour, priority,
                                advice, actions, rationale, probability }, … ],
          total,
          shap_top_factors: [ { feature, impact, direction, value }, … ],
          graph_features:   { degree, clustering, community }
        }
    """
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No patient data received"}), 400

    result = _get_recommendations(data)
    return jsonify(result), (200 if result.get("success") else 500)


#  ROUTE Recommendations page


@app.route("/recommendations")
def recommendations_page():
    """
    Recommendations.html template.
    """
    session_id = request.args.get("session_id", "")
    source = request.args.get("source", "screening")
    return render_template("Recommendations.html", session_id=session_id, source=source)


#  ROUTE — Get saved screening data for a session


@app.route("/get_screening_data/<session_id>")
def get_screening_data(session_id):
    """
    Returns the pre-screening form data stored in the patient session
    metadata. Called by the Recommendations page when source='ecg'.
    """
    try:
        session_path = _find_session_path(session_id)
        if not session_path:
            return jsonify({"success": False, "message": "Session not found."}), 404
        metadata = MetadataManager.read_metadata(session_path / "metadata.json")
        sfd = metadata.get("screening_form_data")
        if sfd:
            return jsonify({"success": True, "patient_data": sfd})
        return jsonify({
            "success": False,
            "message": "No screening data found for this session.",
        }), 404
    except Exception as e:
        logging.exception("get_screening_data failed")
        return jsonify({"success": False, "message": str(e)}), 500


#  ROUTE — Full recommendations for a session  (GET)

@app.route("/session_recommendations/<session_id>")
def session_recommendations(session_id):
    """
    Convenience endpoint: fetches the stored screening data for a session
    and immediately runs the recommendation pipeline, returning JSON.

    """
    try:
        session_path = _find_session_path(session_id)
        if not session_path:
            return jsonify({"success": False, "message": "Session not found."}), 404

        metadata = MetadataManager.read_metadata(session_path / "metadata.json")
        sfd = metadata.get("screening_form_data")
        if not sfd:
            return jsonify({
                "success": False,
                "message": "No screening data available for this session.",
            }), 404

        result = _get_recommendations(sfd)
        return jsonify(result), (200 if result.get("success") else 500)

    except Exception as e:
        logging.exception("session_recommendations failed")
        return jsonify({"success": False, "message": str(e)}), 500


#  ROUTES angiogram processing


@app.route("/angiogram_processing")
def angiogram_upload_form():
    return render_template("angiogram_processing.html")


@app.route("/angiogram_process", methods=["POST"])
def process_angiogram_file():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    filename = secure_filename(file.filename) or "angiogram"
    temp_path = UPLOAD_FOLDER / filename
    file.save(str(temp_path))

    try:
        pipeline_result = process_angiogram(str(temp_path), output_root=str(OUTPUT_ROOT))
        patient_id = pipeline_result["patient_id"]
        return redirect(url_for("angiogram_results", patient_id=patient_id))
    except Exception as e:
        logging.exception("Standalone angiogram processing failed")
        return f"<h2>Error processing file</h2><p>{str(e)}</p>", 500
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.route("/angiogram_results")
def angiogram_results():
    patient_id = request.args.get("patient_id", "")
    if not patient_id:
        return "Missing patient_id", 400

    patient_dir = OUTPUT_ROOT / patient_id
    metadata_file = patient_dir / "metadata.json"
    if not metadata_file.exists():
        return f"Result folder not found for patient_id: {patient_id}", 404

    meta = MetadataManager.read_metadata(metadata_file)
    variants = meta.get("variants", [])
    selected_indices = meta.get("selected_frame_indices", [])

    for v in variants:
        v["url"] = url_for("serve_preprocessed_frame",
                           patient_id=patient_id, filename=v["filename"])

    return render_template(
        "angiogram_results1.html",
        patient_id=patient_id,
        variants=variants,
        selected_indices=selected_indices,
    )


@app.route("/angiogram_frame/<patient_id>/<filename>")
def serve_preprocessed_frame(patient_id, filename):
    img_path = OUTPUT_ROOT / patient_id / filename
    if not img_path.exists():
        return "Frame not found", 404
    return send_file(str(img_path), mimetype="image/png")


#  Entry point


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    app.run(debug=True)

