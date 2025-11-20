import pickle
import numpy as np
import pandas as pd

# Load trained model, encoders, and scaler
model = pickle.load(open('model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Features required for input
features = ['Age', 'BloodPressure', 'SpecificGravity', 'Albumin', 'Sugar', 'RedBloodCells', 'PusCell', 'PusCellClumps', 'Bacteria', 'BloodGlucoseRandom', 'BloodUrea', 'SerumCreatinine',
            'Sodium', 'Potassium', 'Hemoglobin', 'PackedCellVolume', 'WhiteBloodCellCount', 'RedBloodCellCount', 'Hypertension', 'DiabetesMellitus', 'CoronaryArteryDisease', 'Appetite',
            'PedalEdema', 'Anemia']

def predict_ckd(input_data):
    input_df = pd.DataFrame([input_data], columns=features)
    for col in label_encoders:
      if col in input_df.columns and input_df[col].dtype == 'object':  # Ensure categorical
          if input_df[col].values[0] in label_encoders[col].classes_:
              input_df[col] = label_encoders[col].transform([input_df[col].values[0]])[0]
          else:
              print(f"⚠️ Unseen label detected in {col}: {input_df[col].values[0]}")
              most_frequent_label = label_encoders[col].classes_[0]
              input_df[col] = label_encoders[col].transform([most_frequent_label])[0]


    # Encode categorical features
   # Encode only categorical features



    # Scale input data
    input_df = pd.DataFrame(scaler.transform(input_df), columns=features)

    # Predict probability
    probability = model.predict_proba(input_df)[0][1]

    # Debugging - print probability values
    # print(f"Raw Probability: {probability:.4f}")

    # Adjust classification threshold
    threshold = 0.5 # Adjust threshold to fine-tune sensitivity
    prediction = 1 if probability < 0.5 else 0
    result = "CKD Present" if prediction == 1 else "CKD Not Present"

    return result

# Example Test Case
user_input = {

    'Age': 12,
    'BloodPressure': 80,
    'SpecificGravity': 1.02,
    'Albumin': 0,
    'Sugar': 0,
    'RedBloodCells': 'normal',
    'PusCell': 'normal',
    'PusCellClumps': 'notpresent',
    'Bacteria': 'notpresent',
    'BloodGlucoseRandom': 100,
    'BloodUrea': 26,
    'SerumCreatinine': 0.6,
    'Sodium': 137,
    'Potassium': 4.4,
    'Hemoglobin': 15.8,
    'PackedCellVolume': 49,
    'WhiteBloodCellCount': 6600,
    'RedBloodCellCount': 5.4,
    'Hypertension': 'no',
    'DiabetesMellitus': 'no',
    'CoronaryArteryDisease': 'no',
    'Appetite': 'good',
    'PedalEdema': 'no',
    'Anemia': 'no'

}

result = predict_ckd(user_input)
print(f"Prediction: {result} ")

