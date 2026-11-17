import streamlit as st
import pandas as pd
import joblib

model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details below to predict whether they would have survived:")

pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0.0, 80.0, 28.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.number_input("Passenger Fare ($)", 0.0, 500.0, 32.2)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

sex_val = 0 if sex == "Male" else 1
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
embarked_val = embarked_mapping[embarked]

input_data = pd.DataFrame([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]],
                          columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("🎉 This passenger likely **SURVIVED**!")
    else:
        st.error("❌ This passenger likely **DID NOT SURVIVE**.")
