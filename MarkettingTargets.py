import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pickle
from openai import AzureOpenAI

def generate_response(input):

    client = AzureOpenAI( api_version='2024-06-01',azure_endpoint='https://hexavarsity-secureapi.azurewebsites.net/api/azureai',api_key='4ceeaa9071277c5b')
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role': 'user', 'content': input}],
        temperature=0.7,
        max_tokens=256,
        top_p=0.6,
        frequency_penalty=0.7)

    return res.choices[0].message.content

st.title("Marketting Targets Prediction")
st.title("Please fill details of the customer")
age = st.slider('Age of the Customer', 0, 130, 25)
job = st.selectbox("Profession",
    ("management", "technician", "admin","entrepreneur",
     "blue-collar","unknown","retired","services"),)

marriage = st.radio(
        "Marital status of the customer",
        ('single', 'married', 'divorced'))
education = st.radio(
        "Highest education level of customer",
        ('secondary', 'primary', 'tertiary'))
Family = st.slider('Total Family members of the customer', 1, 10, 2)
balance = st.slider('balance of the Customer(in thousands)', 0, 500, 25)
house = st.toggle('Owns a House')
loan = st.toggle('has a loan')
if st.button('Predict'):
    input_data = {
        "age": [age],
        "job": [job],
        "marital": [marriage],
        "education": [education],
        "default": ["no"],
        "balance": [balance],
        "housing": [house],
        "loan": [loan],
        "contact": ["cellular"],
        "day": [15],
        "month": ["may"],
        "duration": [120],
        "campaign": [2],
        "pdays": [100],
        "previous": [1],
        "poutcome": ["success"]
    }
    input_df = pd.DataFrame(input_data)

    train_data = pd.read_csv('dataset.csv', sep=';')
    train_data = pd.get_dummies(train_data, drop_first=True)

    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=train_data.columns, fill_value=0)

    scaler = StandardScaler()
    scaler.fit(train_data)

    input_scaled = scaler.transform(input_df)
    input_scaled = input_scaled[:, :-1]
    models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "KNN": KNeighborsClassifier(),
            "Neural Network": MLPClassifier()
        }


    with open('Random Forest'+'.pkl', 'rb') as file:
        model = pickle.load(file)
        predictions = model.predict(input_scaled)
    if predictions:
        st.info("Term Deposit can be suggested", icon="ℹ️")
    else:
        st.info("Term Deposit can't be suggested")

    text = f'i will give you some details about a person, please give some banking service suggestions we can provide him age i created a Ml model with historic data to predict whether we can offer term deposit its output is {predictions} so suggest accordingly dont suggest   term deposit  unless it is suggested by ML' + str(
                age)   + ', balance in Thousaunds-' + str(
                balance) + ', ZIP code-' + str(zip) + ',family members-' + str(
                Family) + ',education- ' + education
    bot_response = generate_response(text)
    st.write("Services that could be offered:")
    st.write(bot_response)