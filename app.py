from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("best_model_pipeline.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None

    if request.method == "POST":
        data = {
            'GENDER': [request.form['gender']], 
            'AGE': [int(request.form['age'])],
            'SMOKING': [int(request.form['smoking'])],
            'YELLOW_FINGERS': [int(request.form['yellow_fingers'])],
            'ANXIETY': [int(request.form['anxiety'])],
            'PEER_PRESSURE': [int(request.form['peer_pressure'])],
            'CHRONIC DISEASE': [int(request.form['chronic_disease'])],
            'FATIGUE': [int(request.form['fatigue'])],
            'ALLERGY': [int(request.form['allergy'])],
            'WHEEZING': [int(request.form['wheezing'])],
            'ALCOHOL CONSUMING': [int(request.form['alcohol_consuming'])],
            'COUGHING': [int(request.form['coughing'])],
            'SHORTNESS OF BREATH': [int(request.form['shortness_breath'])],
            'SWALLOWING DIFFICULTY': [int(request.form['swallowing_difficulty'])],
            'CHEST PAIN': [int(request.form['chest_pain'])]
        }

        df = pd.DataFrame(data)

        # Predicción con el modelo
        pred = model.predict(df)[0]

        if pred == "YES":
            prediction_text = "Si hay riesgo de cáncer de pulmón."
        else:
            prediction_text = "NO hay riesgo de cáncer de pulmón."

    return render_template("index.html", prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
