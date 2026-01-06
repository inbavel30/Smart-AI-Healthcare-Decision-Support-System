from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
from scipy.sparse import hstack

app = Flask(__name__)
CORS(app)

# -----------------------------
# Load dataset and models
# -----------------------------
try:
    data = pd.read_csv("structured_healthcare_dataset.csv")
    model = pickle.load(open("models/disease_model.pkl","rb"))
    vectorizer = pickle.load(open("models/vectorizer.pkl","rb"))
    scaler = pickle.load(open("models/scaler.pkl","rb"))
    label_encoder = pickle.load(open("models/label_encoder.pkl","rb"))
    print("✅ Dataset and models loaded successfully")
except Exception as e:
    print("❌ Error loading dataset or models:", e)
    exit(1)

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return "Backend running! POST /analyze to predict disease."

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        req = request.json
        symptoms = req.get("symptoms","").lower()
        age = float(req.get("age",0))
        height = float(req.get("height",0))
        weight = float(req.get("weight",0))
        previous_conditions = req.get("previous_conditions","")
        current_conditions = req.get("current_conditions","")

        # --------- Rule-based detection for 20+ common diseases ----------
        simple_rules = {
            "common cold":["cough","runny nose","sneezing","mild fever"],
            "flu":["fever","cough","body ache","fatigue"],
            "headache":["headache","migraine","mild nausea"],
            "vomiting":["vomiting","nausea"],
            "body pain":["body pain","muscle ache","joint pain"],
            "eye pain":["eye pain","red eyes","eye irritation","itchy eyes"],
            "ear pain":["ear pain","ear ache","ear infection"],
            "diarrhea":["diarrhea","loose stool","stomach upset"],
            "constipation":["constipation","hard stool"],
            "indigestion":["indigestion","acidity","heartburn"],
            "sore throat":["sore throat","throat pain"],
            "skin rash":["rash","itching","red spots"],
            "back pain":["back pain","lower back ache"],
            "uti":["burning urination","frequent urination"],
            "allergy":["sneezing","watery eyes","runny nose"],
            "fever":["fever","chills"],
        }

        remedies = {
            "common cold":"Steam inhalation + Tulsi tea",
            "flu":"Hydration + Ginger tea",
            "headache":"Dark room + Ginger tea",
            "vomiting":"Ginger tea + Light meals",
            "body pain":"Rest + Warm compress",
            "eye pain":"Wash eyes + Avoid strain",
            "ear pain":"Warm compress + Ear drops if mild",
            "diarrhea":"ORS + Hydration + Banana",
            "constipation":"Fiber-rich diet + Hydration",
            "indigestion":"Avoid oily food + Ginger tea",
            "sore throat":"Warm salt water gargle + Honey",
            "skin rash":"Aloe vera gel + Cold compress",
            "back pain":"Rest + Hot water bag",
            "uti":"Hydration + Cranberry juice",
            "allergy":"Avoid allergens + Antihistamines",
            "fever":"Hydration + Rest"
        }

        medicines = {
            "common cold":"Paracetamol",
            "flu":"Paracetamol",
            "headache":"Paracetamol",
            "vomiting":"Ondansetron",
            "body pain":"Paracetamol or Ibuprofen",
            "eye pain":"Artificial tears or mild eye drops",
            "ear pain":"Painkillers or ear drops",
            "diarrhea":"ORS + Probiotics",
            "constipation":"Laxatives",
            "indigestion":"Antacids",
            "sore throat":"Paracetamol",
            "skin rash":"Antihistamines",
            "back pain":"Paracetamol or Ibuprofen",
            "uti":"Antibiotics",
            "allergy":"Antihistamines",
            "fever":"Paracetamol"
        }

        # Check rule-based first
        for disease_name, sym_list in simple_rules.items():
            if any(sym in symptoms for sym in sym_list):
                return jsonify({
                    "disease":disease_name.title(),
                    "home_remedy":remedies[disease_name],
                    "medicine":medicines[disease_name],
                    "doctor_required":"No",
                    "note":"Simple/common disease detected based on symptoms."
                })

        # --------- ML prediction for complex/rare diseases ----------
        text = previous_conditions + " " + current_conditions + " " + symptoms
        X_text_input = vectorizer.transform([text])
        X_numeric_input = scaler.transform([[age,height,weight]])
        X_input = hstack([X_text_input, X_numeric_input])

        pred = model.predict(X_input)
        disease_name = label_encoder.inverse_transform(pred)[0]
        row = data[data["disease"]==disease_name].iloc[0]

        return jsonify({
            "disease":disease_name,
            "home_remedy":row["home_remedy"],
            "medicine":row["common_medicine"],
            "doctor_required":row["doctor_required"],
            "note":"Predicted using ML. Consult a doctor for serious conditions."
        })

    except Exception as e:
        return jsonify({"error":"Something went wrong", "details": str(e)})

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
