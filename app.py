from flask import Flask, render_template, request, redirect, session, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import numpy as np
import os
import base64

from database import conn, cursor

app = Flask(__name__)
app.secret_key = "skinproject123"

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

REPORT_FOLDER = "static/reports"
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Load model
model = load_model("model/skin_disease_model.keras")

# Class labels
CLASS_NAMES = [
    "Actinic Keratoses (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis-like Lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevus (nv)",
    "Vascular Lesions (vasc)"
]

# ---------------- PREDICTION ----------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array, verbose=0)[0]
    print("prediction values: ", prediction)
    class_index = np.argmax(prediction)

    disease = CLASS_NAMES[class_index]
    confidence = round(float(prediction[class_index]) * 100, 2)

    return disease, confidence


def get_risk(confidence):
    if confidence >= 90:
        return "High"
    elif confidence >= 70:
        return "Medium"
    else:
        return "Low"


def get_suggestion(disease, risk):
    suggestions = {
        "Actinic Keratoses (akiec)": "Avoid sun exposure and use sunscreen regularly.",
        "Basal Cell Carcinoma (bcc)": "Seek medical evaluation and protect skin from UV rays.",
        "Benign Keratosis-like Lesions (bkl)": "Usually harmless, monitor changes.",
        "Dermatofibroma (df)": "Generally harmless, consult doctor if changes.",
        "Melanoma (mel)": "URGENT: Consult dermatologist immediately.",
        "Melanocytic Nevus (nv)": "Monitor mole changes and use sun protection.",
        "Vascular Lesions (vasc)": "Consult doctor if swelling increases."
    }
    return suggestions.get(disease, "Consult a dermatologist.")


# ---------------- LOGIN (DATABASE FIXED) ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )

        user = cursor.fetchone()

        if user:
            session["user"] = username
            return redirect("/dashboard")

        return "Invalid Username or Password"

    return render_template("login.html")


# ---------------- REGISTER (DATABASE FIXED) ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        existing = cursor.fetchone()

        if existing:
            return "User already exists!"

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )

        conn.commit()

        return redirect("/")

    return render_template("register.html")


# ---------------- HOME ----------------
@app.route("/home")
def home():
    if "user" not in session:
        return redirect("/")
    return render_template("index.html")


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    cursor.execute("""
        SELECT patient_name, disease, confidence, risk, prediction_time
        FROM predictions
        ORDER BY id DESC
        LIMIT 3
    """)

    records = cursor.fetchall()
    total_records = len(records)

    return render_template("dashboard.html", records=records, total_records=total_records)


# ---------------- PREDICT (UPLOAD + CAMERA) ----------------
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("file")
    camera_image = request.form.get("cameraImage")

    filepath = None

    # FILE UPLOAD
    if file and file.filename != "":
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

    # CAMERA IMAGE
    elif camera_image:
        camera_image = camera_image.split(",")[1]
        img_data = base64.b64decode(camera_image)

        filename = f"camera_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        with open(filepath, "wb") as f:
            f.write(img_data)

    if filepath is None:
        return render_template("index.html", error="Please select or capture image.")

    disease, confidence = predict_image(filepath)
    risk = get_risk(confidence)
    suggestion = get_suggestion(disease, risk)
    prediction_time = datetime.now().strftime("%d-%m-%Y %I:%M %p")
    time_now = datetime.now().strftime("%d-%m-%Y %I:%M %p")

    session["last_prediction"] = {
        "patient_name": session.get("user"),
        "disease": disease,
        "confidence": confidence,
        "risk": risk,
        "suggestion": suggestion,
        "time": time_now,
        "image": filepath
    }

    cursor.execute("""
        INSERT INTO predictions
        (patient_name, disease, confidence, risk, suggestion, prediction_time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        session.get("user"),
        disease,
        confidence,
        risk,
        suggestion,
        time_now
    ))

    conn.commit()

    session["result_data"] = {
        "prediction": disease,
        "confidence": confidence,
        "risk": risk,
        "suggestion": suggestion,
        "prediction_time": prediction_time,
        "image_path": filepath
    }

    return redirect("/result")


# ---------------- PDF ----------------
@app.route("/download_pdf")
def download_pdf():

    if "last_prediction" not in session:
        return "No prediction found"

    data = session["last_prediction"]

    file_name = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    file_path = os.path.join(REPORT_FOLDER, file_name)

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("AI SKIN DISEASE REPORT", styles["Title"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Patient: {data['patient_name']}", styles["Normal"]))
    content.append(Paragraph(f"Disease: {data['disease']}", styles["Normal"]))
    content.append(Paragraph(f"Confidence: {data['confidence']}%", styles["Normal"]))
    content.append(Paragraph(f"Risk: {data['risk']}", styles["Normal"]))
    content.append(Paragraph(f"Time: {data['time']}", styles["Normal"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Advice:", styles["Heading2"]))
    content.append(Paragraph(data["suggestion"], styles["Normal"]))

    if os.path.exists(data["image"]):
        content.append(Spacer(1, 10))
        content.append(Image(data["image"], width=180, height=180))

    doc.build(content)

    return redirect(f"/static/reports/{file_name}")


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ---------------- DELETE HISTORY ----------------
@app.route("/delete_history")
def delete_history():
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    return "All history deleted successfully!"

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

from flask import jsonify

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()
    message = data["message"].lower()

    if "diet" in message or "eat" in message or "food" in message:
        reply = "Eat fresh fruits, green vegetables, whole grains, and drink plenty of water. Avoid excessive junk food and sugary drinks."

    elif "cream" in message or "ointment" in message:
        reply = "Use only creams recommended by a dermatologist. Avoid self-medicating with steroid creams."

    elif "sun" in message or "sunlight" in message:
        reply = "Use sunscreen (SPF 30 or higher), wear protective clothing, and avoid strong midday sunlight."

    elif "skin care" in message or "care" in message:
        reply = "Keep your skin clean, moisturize regularly, avoid scratching, and follow a healthy lifestyle."

    elif "avoid" in message:
        reply = "Avoid scratching the affected area, harsh chemicals, smoking, and excessive sun exposure."

    elif "water" in message:
        reply = "Drink at least 2 liters of water daily to help maintain healthy skin."

    elif "hello" in message or "hi" in message:
        reply = "Hello! Ask me about diet, skincare, sunscreen, creams, or general skin care."

    elif "thank you" in message:
        reply = "welcome!!!."

    else:
        reply = "I'm an educational AI assistant. I can answer general questions about skincare, diet, sun protection, and healthy habits. For diagnosis or treatment, please consult a dermatologist."

    return jsonify({"reply": reply})


@app.route("/result")
def result():
    if "result_data" not in session:
        return redirect("/home")

    data = session["result_data"]

    return render_template("result.html", **data)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)