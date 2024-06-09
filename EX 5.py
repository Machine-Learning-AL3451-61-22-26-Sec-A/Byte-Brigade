import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    data = pd.read_csv('tennisdata.csv')
    return data

def preprocess_data(data):
    # Print column names to debug
    st.write("Columns in the dataset:", data.columns.tolist())
    
    # Check for correct column names
    required_columns = ['Outlook', 'Temperature', 'Humidity', 'Windy', 'PlayTennis']
    for column in required_columns:
        if column not in data.columns:
            raise KeyError(f"Column '{column}' is missing from the dataset")
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    le_outlook = LabelEncoder()
    X['Outlook'] = le_outlook.fit_transform(X['Outlook'])
    
    le_temperature = LabelEncoder()
    X['Temperature'] = le_temperature.fit_transform(X['Temperature'])
    
    le_humidity = LabelEncoder()
    X['Humidity'] = le_humidity.fit_transform(X['Humidity'])
    
    le_windy = LabelEncoder()
    X['Windy'] = le_windy.fit_transform(X['Windy'])
    
    le_play_tennis = LabelEncoder()
    y = le_play_tennis.fit_transform(y)
    
    return X, y

def train_model(X_train, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier

def main():
    st.title("BYTES BRIGADES")
    st.title("Play Tennis Predictor")
    
    # Load data
    data = load_data()
    st.subheader("Dataset")
    st.write(data.head())
    
    # Preprocess data
    try:
        X, y = preprocess_data(data)
        st.subheader("Processed Data")
        st.write(X.head())
        st.write("Labels:", y)
    except KeyError as e:
        st.error(f"Error: {e}")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    # Train model
    classifier = train_model(X_train, y_train)
    
    # Evaluate model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Model Performance")
    st.write("Accuracy: {:.2f}%".format(accuracy * 100))
    
    st.subheader("Make a Prediction")
    outlook = st.selectbox("Outlook", ["Sunny", "Overcast", "Rainy"])
    temperature = st.selectbox("Temperature", ["Hot", "Mild", "Cool"])
    humidity = st.selectbox("Humidity", ["High", "Normal"])
    windy = st.selectbox("Windy", ["False", "True"])
    
    if st.button("Predict"):
        le_outlook = LabelEncoder().fit(["Sunny", "Overcast", "Rainy"])
        le_temperature = LabelEncoder().fit(["Hot", "Mild", "Cool"])
        le_humidity = LabelEncoder().fit(["High", "Normal"])
        le_windy = LabelEncoder().fit(["False", "True"])
        
        input_data = [[
            le_outlook.transform([outlook])[0],
            le_temperature.transform([temperature])[0],
            le_humidity.transform([humidity])[0],
            le_windy.transform([windy])[0]
        ]]
        
        prediction = classifier.predict(input_data)
        play_tennis = "Yes" if prediction[0] == 1 else "No"
        st.write(f"The model predicts: Play Tennis = {play_tennis}")

if __name__ == "__main__":
