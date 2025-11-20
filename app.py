from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = "secret_key"  # Required for session handling

# Load trained ML model
model = pickle.load(open('model.pkl', 'rb'))

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "admin":
            session['user'] = username  # ✅ Store user session correctly
            return redirect(url_for('ckd_form'))

        else:
            return render_template('login.html', error="Invalid username or password!")

    return render_template('login.html')

# CKD Form Page (Only After Login)
@app.route('/ckd_form')
def ckd_form():
    if 'user' not in session:
        return redirect(url_for('login'))  # ✅ Redirects to login if not logged in
    return render_template('index2.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define categorical mappings
        categorical_mappings = {
            "red_blood_cells": {"normal": 0, "abnormal": 1},
            "pus_cells": {"normal": 0, "abnormal": 1},
            "pus_cell_clumps": {"notpresent": 0, "present": 1},
            "bacteria": {"notpresent": 0, "present": 1},
            "hypertension": {"no": 0, "yes": 1},
            "diabetes_mellitus": {"no": 0, "yes": 1},
            "coronary_artery_disease": {"no": 0, "yes": 1},
            "appetite": {"poor": 0, "good": 1},
            "pedal_edema": {"no": 0, "yes": 1},
            "anemia": {"no": 0, "yes": 1}
        }

        input_features = []
        for key, value in request.form.items():
            if key in categorical_mappings:
                # Convert categorical value to number
                if value in categorical_mappings[key]:
                    input_features.append(categorical_mappings[key][value])
                else:
                    return jsonify({'prediction': f"Invalid input for {key}. Please select a valid option."})
            else:
                try:
                    input_features.append(float(value))  # Convert numeric values
                except ValueError:
                    return jsonify({'prediction': f"Invalid input for {key}. Please enter numeric values."})

        # Convert input into NumPy array for model prediction
        input_array = np.array(input_features).reshape(1, -1)

        # Make prediction
        probability = model.predict_proba(input_array)[0][1]  # Probability of CKD Present
        print(f"Model Probability: {probability:.2f}")  # ✅ Debugging in console

        # Apply probability threshold
        prediction = 1 if probability > 0.4 else 0  # Fix logic here

        # Convert output to readable format
        result = "CKD Present" if prediction == 1 else "CKD Not Present"

        return jsonify({'prediction': result})  # ✅ Return only prediction, NOT probability

    except Exception as e:
        return jsonify({'error': str(e)})


# Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)  # ✅ Clears session on logout
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
