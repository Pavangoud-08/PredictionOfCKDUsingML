****<h1>💡 Prediction of Chronic Kidney Disease Using Machine Learning</h1>****

This project focuses on developing a Machine Learning-based Chronic Kidney Disease (CKD) prediction system capable of identifying early signs of CKD using clinical patient data. The system uses data preprocessing, feature engineering, and classification algorithms—primarily the Random Forest model—to classify whether a patient has CKD or not.

Early detection of CKD helps prevent severe kidney damage and improves patient outcomes. This project aims to support healthcare professionals through an automated, accurate, and efficient CKD prediction tool.

**<h2>📌 Project Features
🔬 Machine Learning Techniques</h2>**

Random Forest (Primary Model)

Support Vector Machine

Decision Trees

XGBoost

Hyperparameter tuning (Grid Search / Randomized Search)

**<h2>🩺 Medical Features Used</h2>**

19+ clinical parameters, including:

Serum Creatinine

Blood Urea

Haemoglobin

Blood Pressure

Blood Glucose Random

Sodium, Potassium

Specific Gravity

Albumin, Sugar

Diabetes Mellitus, Hypertension

Anemia, Appetite, Pedal Edema

**<h2>🛠 Technologies Used</h2>**
<h3>Programming & Frameworks</h3>

Python

Flask / Django

Jupyter Notebook / Google Colab

<h3>Libraries</h3>

Scikit-learn

Pandas, NumPy

Matplotlib, Seaborn

**<h2>📊 System Architecture</h2>**

<img width="693" height="382" alt="Screenshot (6)" src="https://github.com/user-attachments/assets/f5339abb-d75d-45d7-990a-811752ae8373" />




<h3>The workflow includes:</h3>

Data Collection

Data Preprocessing

Feature Selection

Model Training

Evaluation

Prediction

Deployment

**<h2>🧪 Results</h2>**

The CKD Prediction System was evaluated through multiple machine learning models, and the Random Forest Classifier produced the best results.

<img width="910" height="512" alt="image" src="https://github.com/user-attachments/assets/8df083dc-b80c-4872-9497-67a37960d26e" />


**<h2>✅ Overall Model Performance</h2>**

Accuracy: 93%

Precision: High

Recall: High

F1-Score: Balanced performance

ROC-AUC: Strong class separability

**<h3>📈 Accuracy Graph</h3>**

<img width="547" height="418" alt="image" src="https://github.com/user-attachments/assets/dd8ef8e1-0811-45dd-a999-353c5b8f0768" />

<h3>📉 ROC Curves</h3>

<h4>ROC curves were generated for both:</h4>

<h4>Random Forest</h4>

<img width="586" height="441" alt="image" src="https://github.com/user-attachments/assets/2a006c5b-b6bc-44f8-929c-29cfc331602c" />

<h4>XGBoost</h4>

<img width="591" height="436" alt="image" src="https://github.com/user-attachments/assets/469d2e7e-74fa-45dc-82a0-55238cae9583" />




Both models achieved high AUC values, with Random Forest performing slightly better in distinguishing CKD vs Non-CKD classes.

<h2>🔍 Predicted Output Example</h2>

A sample prediction demonstrates the system correctly classifying a patient’s clinical parameters as “CKD Present” or “CKD Not Present.”

**<h2>✔️ Key Observations</h2>**

Serum Creatinine, Hemoglobin, and Blood Urea showed strong influence on model prediction.

Feature selection helped improve the accuracy and reduce noise.

The model handled missing values effectively after preprocessing.

Performance remained stable even when evaluated with cross-validation.

The system proves to be:

Accurate (93%)

Reliable

Consistent

Clinically meaningful

These results confirm that the ML-based CKD prediction system can be used as an early diagnosis support tool in healthcare environments.

**<h2>📂 Dataset Information</h2>**

Records: 400

Features: 26 total, 19 used after preprocessing

Contains clinical and categorical medical parameters

Preprocessed using scaling, encoding, and missing value imputation

**<h2>🖥️ System Modules</h2>**

Preprocessing Module

Feature Engineering Module

Model Training Module

Prediction Module

Results & Reports Module

Deployment Module

**<h2>🔮 Future Scope</h2>**

Deep Learning integration (ANN, CNN)

Real-time CKD monitoring using IoT wearables

Cloud-based model deployment

Expansion into multi-disease prediction

Blockchain-based secure medical storage

**<h2>📚 References</h2>**

A complete list of research papers and IEEE sources used in the literature survey is included in the project report. These papers cover CKD prediction using ML, CNNs, ensemble learning, and explainable AI.

**<h2>👨‍⚕️ Conclusion</h2>**

This CKD Prediction System demonstrates the effectiveness of machine learning in healthcare diagnostics. By using clean clinical data, optimized ML models, and structured feature engineering, the system achieves high accuracy (93%) and serves as a reliable early-diagnosis support tool.

The project shows strong potential for real-world implementation and further enhancement.
# PredictionOfCKDUsingML
