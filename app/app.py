"""
Interactive Flask Web Application for Titanic Survival Prediction.
Provides a modern UI and REST API for real-time inference and model exploration.
"""
import os
import sys
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_single_passenger, load_trained_model
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
BENCHMARK_PATH = os.path.join(REPORTS_DIR, "model_benchmark.csv")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats", methods=["GET"])
def get_stats():
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
            
    benchmark_data = []
    if os.path.exists(BENCHMARK_PATH):
        df_bench = pd.read_csv(BENCHMARK_PATH)
        benchmark_data = df_bench.to_dict(orient="records")
        
    return jsonify({
        "metadata": metadata,
        "benchmark": benchmark_data
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json or {}
        
        pclass = int(data.get("Pclass", 3))
        sex = str(data.get("Sex", "male")).lower()
        age = float(data.get("Age", 28.0))
        sibsp = int(data.get("SibSp", 0))
        parch = int(data.get("Parch", 0))
        fare = float(data.get("Fare", 15.0))
        embarked = str(data.get("Embarked", "S")).upper()
        name = str(data.get("Name", "Mr. John Doe"))
        cabin = str(data.get("Cabin", ""))
        
        passenger = {
            "Pclass": pclass,
            "Name": name,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Cabin": cabin if cabin else None,
            "Embarked": embarked,
            "Ticket": "A/5 21171"
        }
        
        result = predict_single_passenger(passenger)
        
        # Explainable factors for survival
        factors = []
        if sex == "female":
            factors.append({"type": "positive", "text": "Female passenger ('Women & Children First' protocol applied)"})
        else:
            factors.append({"type": "negative", "text": "Male passenger (lower lifeboat boarding priority)"})
            
        if pclass == 1:
            factors.append({"type": "positive", "text": "1st Class luxury ticket (upper boat deck access)"})
        elif pclass == 2:
            factors.append({"type": "positive", "text": "2nd Class ticket (moderate proximity to deck)"})
        else:
            factors.append({"type": "negative", "text": "3rd Class steerage (lower decks, delayed evacuation access)"})
            
        if age <= 12:
            factors.append({"type": "positive", "text": f"Young child ({int(age)} yrs) given lifeboat priority"})
        elif age > 60:
            factors.append({"type": "negative", "text": f"Senior passenger ({int(age)} yrs) mobility challenge"})
            
        fam_size = sibsp + parch + 1
        if 2 <= fam_size <= 4:
            factors.append({"type": "positive", "text": f"Optimal family size of {fam_size} (mutual support without delay)"})
        elif fam_size > 4:
            factors.append({"type": "negative", "text": f"Large family size of {fam_size} (difficult to coordinate evacuation)"})
        else:
            factors.append({"type": "neutral", "text": "Solo passenger (independent, but lacked support team)"})
            
        if cabin and cabin.strip():
            factors.append({"type": "positive", "text": f"Assigned cabin deck '{cabin[0].upper()}'"})
            
        result["factors"] = factors
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/archetypes", methods=["GET"])
def get_archetypes():
    archetypes = [
        {
            "title": "Rose DeWitt Bukater (Upper Class Debutante)",
            "Pclass": 1,
            "Name": "Bukater, Miss. Rose DeWitt",
            "Sex": "female",
            "Age": 17,
            "SibSp": 0,
            "Parch": 1,
            "Fare": 227.5,
            "Cabin": "B51",
            "Embarked": "C"
        },
        {
            "title": "Jack Dawson (Third Class Artist)",
            "Pclass": 3,
            "Name": "Dawson, Mr. Jack",
            "Sex": "male",
            "Age": 20,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 8.05,
            "Cabin": "",
            "Embarked": "S"
        },
        {
            "title": "Master William Carter (1st Class Young Boy)",
            "Pclass": 1,
            "Name": "Carter, Master. William Thornton II",
            "Sex": "male",
            "Age": 11,
            "SibSp": 1,
            "Parch": 2,
            "Fare": 120.0,
            "Cabin": "B96",
            "Embarked": "S"
        },
        {
            "title": "Third Class Father Traveling with Large Family",
            "Pclass": 3,
            "Name": "Andersson, Mr. Anders Johan",
            "Sex": "male",
            "Age": 39,
            "SibSp": 1,
            "Parch": 5,
            "Fare": 31.275,
            "Cabin": "",
            "Embarked": "S"
        }
    ]
    return jsonify(archetypes)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
