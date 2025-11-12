from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("best_model_pipeline.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = 1 if request.form['GENDER'] == 'M' else 0
    age = int(request.form['AGE'])
    smoking = int(request.form['SMOKING'])
    yellow_fingers = int(request.form['YELLOW_FINGERS'])
    anxiety = int(request.form['ANXIETY'])
    peer_pressure = int(request.form['PEER_PRESSURE'])
    chronic_disease = int(request.form['CHRONIC_DISEASE'])
    fatigue = int(request.form['FATIGUE'])
    allergy = int(request.form['ALLERGY'])
    wheezing = int(request.form['WHEEZING'])
    alcohol = int(request.form['ALCOHOL_CONSUMING'])
    coughing = int(request.form['COUGHING'])
    shortness = int(request.form['SHORTNESS_OF_BREATH'])
    swallowing = int(request.form['SWALLOWING_DIFFICULTY'])
    chest_pain = int(request.form['CHEST_PAIN'])

    features = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                          chronic_disease, fatigue, allergy, wheezing, alcohol, coughing,
                          shortness, swallowing, chest_pain]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if probability > 0.5:
        risk = "Alto riesgo"
        color = "text-danger"
        message = "El modelo indica una alta probabilidad de que el usuario presente signos relacionados con cáncer de pulmón."
    else:
        risk = "Bajo riesgo"
        color = "text-success"
        message = "El modelo indica baja probabilidad de cáncer de pulmón. Aun así, se recomienda chequeo médico si hay síntomas."

    return render_template('result.html',
                           prediction=risk,
                           probability=round(probability*100, 2),
                           color=color,
                           message=message,
                           form=request.form)

if __name__ == '__main__':
    app.run(debug=True)
