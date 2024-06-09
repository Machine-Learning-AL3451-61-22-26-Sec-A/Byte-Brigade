import streamlit as st
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def load_data():
    return pd.read_csv("coronadata.csv")

def preprocess_data(data):
    # Drop any rows with missing values
    data.dropna(inplace=True)
    return data

def train_model(data):
    model = BayesianModel([
        ('Fever', 'CORONA Diagnosis'),
        ('Cough', 'CORONA Diagnosis'),
        ('Shortness of Breath', 'CORONA Diagnosis'),
        ('Fatigue', 'CORONA Diagnosis'),
        ('Body Aches', 'CORONA Diagnosis'),
        ('Loss of Taste/Smell', 'CORONA Diagnosis')
    ])
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    return model

def main():
    st.tile("BYTES BRIGADE")
    st.title("CORONA Infection Diagnosis")
    
    # Load data
    data = load_data()
    st.subheader("Dataset")
    st.write(data)
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Train model
    model = train_model(data)
    
    # User input for symptoms
    st.subheader("Enter Symptoms")
    fever = st.checkbox("Fever")
    cough = st.checkbox("Cough")
    breathlessness = st.checkbox("Shortness of Breath")
    fatigue = st.checkbox("Fatigue")
    body_aches = st.checkbox("Body Aches")
    loss_of_taste_smell = st.checkbox("Loss of Taste/Smell")
    
    # Collect evidence
    evidence = {
        "Fever": 1 if fever else 0,
        "Cough": 1 if cough else 0,
        "Shortness of Breath": 1 if breathlessness else 0,
        "Fatigue": 1 if fatigue else 0,
        "Body Aches": 1 if body_aches else 0,
        "Loss of Taste/Smell": 1 if loss_of_taste_smell else 0,
    }
    
    # Perform inference
    infer = VariableElimination(model)
    probability = infer.map_query(variables=['CORONA Diagnosis'], evidence=evidence)
    probability_corona = probability['CORONA Diagnosis']
    
    # Display diagnosis
    st.subheader("Diagnosis")
    if probability_corona == 'Positive':
        st.write("Based on the symptoms entered, there is a high probability of CORONA infection.")
    else:
        st.write("Based on the symptoms entered, CORONA infection is unlikely.")

if __name__ == "__main__":
    main()
