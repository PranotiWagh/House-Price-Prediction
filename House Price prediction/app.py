
# =====================================================
# HOUSING PRICE PREDICTION - FLASK BACKEND
# =====================================================

import os
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import r2_score


# =====================================================
# 1. FLASK APPLICATION
# =====================================================

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# =====================================================
# 2. LOAD DATASET
# =====================================================

DATASET_PATH = "HousingData.csv"

df = pd.read_csv(DATASET_PATH)

# Remove missing target values

df = df.dropna(subset=["MEDV"])

X = df.drop(columns=["MEDV"])
y = df["MEDV"]

# Store exact feature names and order

feature_names = X.columns.tolist()

print("Dataset loaded successfully")
print("Features:", feature_names)


# =====================================================
# 3. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# 4. MACHINE LEARNING MODELS
# =====================================================

models = {

    "Linear Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearRegression()
    ),

    "Ridge Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=1.0)
    ),

    "Lasso Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Lasso(alpha=0.1)
    ),

    "ElasticNet": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        ElasticNet(alpha=0.1)
    ),

    "Decision Tree": make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeRegressor(
            max_depth=8,
            random_state=42
        )
    ),

    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    ),

    "Extra Trees": make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(
            n_estimators=100,
            random_state=42
        )
    ),

    "Gradient Boosting": make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    ),

    "AdaBoost": make_pipeline(
        SimpleImputer(strategy="median"),
        AdaBoostRegressor(
            n_estimators=150,
            learning_rate=0.05,
            random_state=42
        )
    ),

    "Hist Gradient Boosting": make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42
        )
    ),

    "KNN": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=5)
    ),

    "SVR": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVR(C=10, epsilon=0.1, kernel="rbf")
    )
}


# =====================================================
# 5. OPTIONAL MODELS
# =====================================================

try:

    from xgboost import XGBRegressor

    models["XGBoost"] = make_pipeline(
        SimpleImputer(strategy="median"),
        XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    )

except ImportError:
    print("XGBoost not installed")


try:

    from lightgbm import LGBMRegressor

    models["LightGBM"] = make_pipeline(
        SimpleImputer(strategy="median"),
        LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1
        )
    )

except ImportError:
    print("LightGBM not installed")


try:

    from catboost import CatBoostRegressor

    from catboost import CatBoostRegressor

    models["CatBoost"] = make_pipeline(
        SimpleImputer(strategy="median"),
        CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_seed=42
        )
    )

except ImportError:
    print("CatBoost not installed")


# =====================================================
# 6. TRAIN ALL MODELS AND FIND BEST MODEL
# =====================================================

trained_models = {}
model_results = {}

print("\nTraining Machine Learning Models...\n")

for name, model in models.items():

    print("Training:", name)

    try:

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        score = r2_score(y_test, predictions)

        trained_models[name] = model

        model_results[name] = score

        print("R2 Score:", round(score, 4))

    except Exception as error:

        print("Error training", name, ":", error)


if not trained_models:
    raise RuntimeError("No machine learning models were trained successfully.")


# Find best model based on test R2 score

best_algorithm = max(
    model_results,
    key=model_results.get
)

best_model = trained_models[best_algorithm]

print("\n===================================")
print("BEST MODEL:", best_algorithm)
print("R2 SCORE:", model_results[best_algorithm])
print("===================================")


# =====================================================
# 7. WEB GUI & STATUS ROUTES
# =====================================================

@app.route("/")
def home():
    # Serve the frontend web GUI
    return send_from_directory(".", "index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "message": "Housing Price Prediction API is running",
        "best_algorithm": best_algorithm,
        "r2_score": round(
            model_results[best_algorithm], 4
        )
    })


# =====================================================
# 8. PREDICTION ROUTE
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Receive JSON data from frontend

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Invalid JSON request"
            }), 400

        # Check all required features

        missing_features = [
            feature for feature in feature_names
            if feature not in data
        ]

        if missing_features:

            return jsonify({

                "error": "Missing required features",

                "missing_features": missing_features

            }), 400

        # Construct input using exact training feature order

        input_data = pd.DataFrame(
            [[data[feature] for feature in feature_names]],
            columns=feature_names
        )

        # Validate numeric input

        input_data = input_data.apply(
            pd.to_numeric,
            errors="coerce"
        )

        if input_data.isna().any().any():

            return jsonify({
                "error": "All features must contain valid numeric values"
            }), 400

        # Predict using best model

        prediction_val = float(best_model.predict(input_data)[0])
        pred_rounded = round(prediction_val, 2)
        price_dollars = round(pred_rounded * 1000, 2)

        return jsonify({

            "success": True,

            "prediction": pred_rounded,

            "price": price_dollars,

            "price_thousands": pred_rounded,

            "unit": "thousand USD",

            "best_algorithm": best_algorithm,

            "r2_score": round(
                model_results[best_algorithm], 4
            )

        })

    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# =====================================================
# 9. MODEL COMPARISON ROUTE
# =====================================================

@app.route("/models", methods=["GET"])
def get_models():

    results = []

    for name, score in model_results.items():

        results.append({

            "algorithm": name,

            "r2_score": round(float(score), 4)

        })

    results.sort(
        key=lambda item: item["r2_score"],
        reverse=True
    )

    return jsonify({

        "models": results,

        "best_model": best_algorithm

    })


# =====================================================
# 10. RUN FLASK SERVER
# =====================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
