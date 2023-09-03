import pickle
import pandas as pd

with open("models/DTs_model.pkl", "rb") as file:
    DTs_loaded = pickle.load(file)

with open("models/RF_model.pkl", "rb") as file:
    RF_loaded = pickle.load(file)

with open("models/knn_model.pkl", "rb") as file:
    knn_loaded = pickle.load(file)

def get_user_data():
    age = float(input("Please enter your age: "))
    gender = input("Please enter your gender (e.g., Male, Female, etc.): ")
    impluse = float(input("Please enter your impluse value: "))
    pressurehight = float(input("Please enter your high blood pressure value: "))
    pressurelow = float(input("Please enter your low blood pressure value: "))
    glucose = float(input("Please enter your glucose level: "))
    kcm = float(input("Please enter your KCM value: "))
    troponin = float(input("Please enter your troponin level: "))

    user_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'impluse': [impluse],
        'pressurehight': [pressurehight],
        'pressurelow': [pressurelow],
        'glucose': [glucose],
        'kcm': [kcm],
        'troponin': [troponin]
    })

    return user_data

user_data = get_user_data()

DTs_prediction = DTs_loaded.predict(user_data)
RF_prediction = RF_loaded.predict(user_data)
knn_prediction = knn_loaded.predict(user_data)

print(f"Prediction from Decision Tree model: {DTs_prediction[0]}")
print(f"Prediction from Random Forest model: {RF_prediction[0]}")
print(f"Prediction from K-Nearest Neighbors model: {knn_prediction[0]}")
