import pickle
import pandas as pd

with open("models/DTs_model.pkl", "rb") as file:
    DTs_loaded = pickle.load(file)

with open("models/RF_model.pkl", "rb") as file:
    RF_loaded = pickle.load(file)

with open("models/knn_model.pkl", "rb") as file:
    knn_loaded = pickle.load(file)

def get_user_data():
    age = float(input("enter age: "))
    gender = input("1 male 0 female: ")
    impluse = float(input("enter impluse value: "))
    pressurehight = float(input("high blood pressure value: "))
    pressurelow = float(input("low blood pressure value: "))
    glucose = float(input("glucose level: "))
    kcm = float(input("KCM value: "))
    troponin = float(input("troponin level: "))

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
