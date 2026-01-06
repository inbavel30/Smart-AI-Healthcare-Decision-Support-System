import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import pickle
import os

data = pd.read_csv("structured_healthcare_dataset.csv")

for col in ["previous_conditions","current_conditions","symptoms"]:
    data[col] = data[col].fillna("")

data["text_features"] = data["previous_conditions"] + " " + data["current_conditions"] + " " + data["symptoms"]

vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(data["text_features"])

scaler = StandardScaler()
X_numeric = scaler.fit_transform(data[["age","height","weight"]])

X = hstack([X_text, X_numeric])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["disease"])

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/disease_model.pkl","wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl","wb"))
pickle.dump(scaler, open("models/scaler.pkl","wb"))
pickle.dump(label_encoder, open("models/label_encoder.pkl","wb"))

print("✅ ML model trained and saved in models/")
