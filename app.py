import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Model features as defined in randomforest_model_automobile.pkl
FEATURE_NAMES = [
    'Make', 'Model', 'Year', 'Fuel_Type', 'Transmission', 
    'Engine_Size', 'Mileage', 'Horsepower', 'Torque', 'Owners', 
    'Accident_History', 'Service_History', 'Color', 'Body_Type', 
    'Drivetrain', 'Fuel_Efficiency', 'Location'
]

# Load serialized model
MODEL_PATH = 'randomforest_model_automobile.pkl'
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: '{MODEL_PATH}' not found in working directory.")

# Embedded HTML Template with Modern Styling
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoValuation - Car Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-color: #38bdf8;
            --accent-hover: #0284c7;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --input-bg: #0f172a;
            --success-color: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2.5rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .result-card {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success-color);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            margin-bottom: 2rem;
        }

        .result-card h3 {
            color: var(--text-muted);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .result-card .price {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--success-color);
            margin-top: 0.25rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        .form-group input {
            background-color: var(--input-bg);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .form-group input:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background-color: var(--accent-color);
            color: #000;
            font-weight: 600;
            font-size: 1rem;
            padding: 0.85rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s ease, transform 0.1s ease;
        }

        .btn-submit:hover {
            background-color: var(--accent-hover);
            color: #fff;
        }

        .btn-submit:active {
            transform: scale(0.99);
        }

        .error-banner {
            background-color: rgba(239, 68, 68, 0.1);
            border: 1px solid #ef4444;
            color: #f87171;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>AutoValuation Predictor</h1>
        <p>Enter car specifications to estimate market value using Random Forest Regression</p>
    </header>

    {% if error %}
        <div class="error-banner">
            {{ error }}
        </div>
    {% endif %}

    {% if prediction is not none %}
        <div class="result-card">
            <h3>Estimated Market Value</h3>
            <div class="price">${{ "{:,.2f}".format(prediction) }}</div>
        </div>
    {% endif %}

    <form method="POST" action="/predict" class="grid-form">
        {% for feature in features %}
            <div class="form-group">
                <label for="{{ feature }}">{{ feature.replace('_', ' ') }}</label>
                <input 
                    type="number" 
                    step="any" 
                    id="{{ feature }}" 
                    name="{{ feature }}" 
                    placeholder="Enter {{ feature.replace('_', ' ') }}"
                    value="{{ request.form.get(feature, '') }}"
                    required
                >
            </div>
        {% endfor %}

        <button type="submit" class="btn-submit">Calculate Estimated Value</button>
    </form>
</div>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_LAYOUT, features=FEATURE_NAMES, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(
            HTML_LAYOUT, 
            features=FEATURE_NAMES, 
            prediction=None, 
            error="Model file 'randomforest_model_automobile.pkl' is missing or not loaded correctly."
        )

    try:
        # Extract features in the precise order expected by the model
        input_data = [float(request.form.get(feature, 0)) for feature in FEATURE_NAMES]
        features_array = np.array([input_data])
        
        # Perform prediction
        prediction_val = model.predict(features_array)[0]
        
        return render_template_string(
            HTML_LAYOUT, 
            features=FEATURE_NAMES, 
            prediction=prediction_val
        )
    except ValueError as e:
        return render_template_string(
            HTML_LAYOUT, 
            features=FEATURE_NAMES, 
            prediction=None, 
            error="Invalid input. Please ensure numerical values are entered across all fields."
        )
    except Exception as e:
        return render_template_string(
            HTML_LAYOUT, 
            features=FEATURE_NAMES, 
            prediction=None, 
            error=f"Prediction error: {str(e)}"
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
