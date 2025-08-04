# Childhood-Malnutrition-Prediction_-Capstone-Project

Names:Tesi Divine 
ID: 26017
Course: Introduction to Big Data Analytics
Email: tesidivine02@gmail.com
--------------------------------------------------------------------------------------------------------------------------------------------


## 🎯 Project Overview


Childhood malnutrition remains a persistent public health crisis, especially in developing nations like Rwanda. 
It contributes significantly to child morbidity and mortality. 
Early detection is critical but often delayed due to lack of resources, trained staff, or tools.
This capstone project aims to **predict the risk of malnutrition in children under five** using demographic
and anthropometric data (e.g., age, weight, height, MUAC). The goal is to **enable early intervention** and support health professionals with data-driven insights.

Dataset: https://dhsprogram.com/data/dataset/Rwanda_Standard-DHS_2019.cfm
-----------------------------------------------------------------------------------------------------------------------------------------------------

1️⃣ Part 1: Problem Definition & Planning

✅ Sector Selection

☑ Health

✅ Problem Statement

📁 Dataset Details

Dataset Title	: Childhood-Malnutrition-Prediction

Source Link	https://dhsprogram.com/data/dataset/Rwanda_Standard-DHS_2019.cfm

Number of Rows	5,000 (filtered sample for children under 5 years)

Number of Columns	7 (key features used in the prediction model)

Data Structure	☑ Structured (CSV) ☐ Unstructured (Text, Images)

Data Status	☐ Clean ☑ Requires Preprocessing (handled in Python)
-----------------------------------------------------------------------------------------------------------------------------------------------------------

2️⃣ Part 2: Python Analytics Tasks


🧹 Data Cleaning


- Removed duplicates
 
- Imputed missing values
 
- Encoded the `nutrition_status` column into numeric form
 
- Scaled features using `StandardScaler`


# === 1. LIBRARIES ===
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings

warnings.filterwarnings('ignore')



📊 Exploratory Data Analysis (EDA)


- Descriptive statistics
- Correlation heatmaps
- Target distribution plots

def run_eda(df):

   print(df.describe())

  # Target distribution
  
  sns.countplot(x='nutrition_status', data=df)
  
   plt.title("Nutrition Status Distribution")
   
   plt.show()

  # Correlation heatmap
  
  plt.figure(figsize=(8, 6))
  
   sns.heatmap(df.drop('nutrition_status', axis=1).corr(), annot=True, cmap='coolwarm')
   
   plt.title("Feature Correlation Matrix")
   
   plt.show()

run_eda(df)

<img width="1187" height="569" alt="image" src="https://github.com/user-attachments/assets/257dced2-6a67-4cb6-bded-0741e590a1de" />



🤖 Machine Learning Model


- Used **Random Forest** and **Logistic Regression**
- Built a **Voting Ensemble Classifier** (Innovation)
- Target: `nutrition_status_encoded` (multi-class classification)

  # Features and target
  
X = df[['age_months', 'weight_kg', 'height_cm', 'muac_cm', 'bmi']]

y = df['nutrition_status_encoded']

# Scale features

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



📈 Model Evaluation


- Confusion matrix
- Accuracy Score: ~88–91%
- Classification report (Precision, Recall, F1-Score)

 def evaluate_model(model, X_test, y_test, le):
 
   y_pred = model.predict(X_test)
    
   print("Classification Report:")
   
   print(classification_report(y_test, y_pred, target_names=le.classes_))

   print("\nConfusion Matrix:")
   
   cm = confusion_matrix(y_test, y_pred)
   
   sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
   
   plt.xlabel("Predicted")
   
   plt.ylabel("True")
   
   plt.title("Confusion Matrix")
   
   plt.show()

   print("Accuracy:", accuracy_score(y_test, y_pred))

evaluate_model(ensemble, X_test, y_test, le)

<img width="735" height="301" alt="image" src="https://github.com/user-attachments/assets/eefdf8d1-2d4c-4d92-a30e-ba642630d4ed" />



  💡 Innovations



- Developed a **custom rule-based risk scoring function** for malnutrition prediction using simplified health criteria (`age`, `muac`, `weight`)
- Added ensemble learning for improved performance

  def custom_risk_score(age, weight, muac):
  
    if age < 24 and weight < 5 and muac < 11.5:
  
  return "High Risk"
  
   elif muac < 12.5:
  
   return "Moderate Risk"
  
    else:
  
   return "Low Risk"

# Example

print(custom_risk_score(18, 4.5, 11.0))

<img width="1098" height="429" alt="image" src="https://github.com/user-attachments/assets/407bf582-a546-405a-ab1c-1bd6535e91aa" />



📈Model Training



# Base Models

rf = RandomForestClassifier(n_estimators=100, random_state=42)

lr = LogisticRegression()

# Ensemble Voting Classifier

ensemble = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')

ensemble.fit(X_train, y_train)

<img width="1084" height="189" alt="image" src="https://github.com/user-attachments/assets/9f5b501a-4d76-4de6-9cb2-de7db5afa567" />
-----------------------------------------------------------------------------------------------------------------------------------------------


3️⃣ Part 3: Power BI Dashboard Tasks

🧩 1. Communicate the Problem & Insights Clearly

📌 Objective Summary (Text box on dashboard):

  This dashboard presents insights from Rwanda DHS 2019–20 child nutrition data. The goal is to detect malnutrition patterns among children under five using demographic and health indicators. The visuals below help identify high-risk groups based on BMI, MUAC, age, and nutrition status.

Key Summary Cards:

  📌 Total number of children analyzed

  📌 Percentage of severely malnourished

  📌 Average BMI, MUAC, and weight

  🧩 2. Incorporate Interactivity

Add Slicers/Filters to allow user exploration:

  ✔️ Nutrition Status (Normal, Moderate, Severe)

  ✔️ Age Group (using DAX-calculated column)

  ✔️ BMI Range

  ✔️ MUAC Range

  🧩 3. Use Appropriate Visuals

Pie Chart:	Nutrition status distribution	

Bar Chart	:Avg MUAC per nutrition class	

Scatter Plot:	BMI vs Age colored by status	

Line Chart:	Avg weight by age group	

Box Plot (custom visual):	MUAC by nutrition status	

KPI Cards:	Summary metrics	

🧩 4. Ensure Design Clarity

  ✔️ Use a clean 2-color or 3-color theme:

  🟢 Green for “Normal”

  🟠 Orange for “Moderate”

  🔴 Red for “Severe”

  ✔️ Align visuals in grids or sections (e.g., Overview, Analysis, Trends)

  ✔️ Label all axes, filters, and titles clearly

  ✔️ Use tooltips to explain chart meanings
  

<img width="1865" height="922" alt="image" src="https://github.com/user-attachments/assets/7c640de8-47ba-4e23-ab11-36e30cbc007c" />




  
  





