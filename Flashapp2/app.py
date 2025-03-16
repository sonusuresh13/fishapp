from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("fish_market_model.pkl")

@app.route('/', methods=["GET", "POST"])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame(data, index=[0])
        prediction = model.predict(input_df)
        return jsonify({"prediction": prediction[0]})
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {e}"})
    except Exception as e:
        return jsonify({"error": f"Error: {e}"})

if __name__ == '__main__':
    app.run(debug=True)