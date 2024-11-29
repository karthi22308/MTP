import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle

globalfile = ""
def dumpmodel(uploaded_file):

    global globalfile
    globalfile= pd.read_csv(uploaded_file, sep=';')
    data = pd.get_dummies(globalfile, drop_first=True)
    X = data.drop('y_yes', axis=1)
    y = data['y_yes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "KNN": KNeighborsClassifier(),
        "Neural Network": MLPClassifier()
    }
    st.info("model training in progress")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'{name} Accuracy: {accuracy:.2f}')
        with open(f'{name}.pkl', 'wb') as file:
            pickle.dump(model, file)
    return


st.title("Re-Train ML model for Prediction")


uploaded_file = st.file_uploader("Choose a csv file with required data")
st.text('please upload a csv file with last column as expected outcome')
if uploaded_file is not None and st.button('Train'):
    dumpmodel(uploaded_file)
    st.info("model training in completed")
