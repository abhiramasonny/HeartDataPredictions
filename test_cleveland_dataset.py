import numpy as np
from keras.models import load_model

categorical_model = load_model("models/categorical_model_dropped.h5")
binary_model = load_model("models/binary_model_dropped.h5")

print("enter ur data (its secure): ")
age = float(input("Age: "))
sex = float(input("Sex (1=male, 0=female): "))
cp = float(input("Chest pain type (1-4): "))
trestbps = float(input("Resting blood pressure: "))
chol = float(input("Serum cholesterol: "))
fbs = float(input("Fasting blood sugar > 120 mg/dl (1=true, 0=false): "))
thalach = float(input("Maximum heart rate achieved: "))
exang = float(input("Exercise induced angina (1=yes, 0=no): "))
thal = float(input("Thalassemia (3=normal, 6=fixed defect, 7=reversible defect): "))

X_new_test = np.array([[age, sex, cp, trestbps, chol, fbs, thalach, exang, thal]])
categorical_predictions = np.argmax(categorical_model.predict(X_new_test), axis=1)
binary_predictions = np.round(binary_model.predict(X_new_test)).astype(int)

print("Categorical Prediction:", categorical_predictions)
print("Binary Prediction:", binary_predictions)


print("Yay!!!")