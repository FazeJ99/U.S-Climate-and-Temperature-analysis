# Step 5: Streamlit App
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def load_data():
    df = pd.read_csv("Average Temperature 1900-2023(1).csv")
    return df
    
def preprocess_data(df):
    df['Year'] = df['Year'].astype(int)
    df.rename(columns = {'Average_Fahrenheit_Temperature': 'Temperature'}, inplace = True)
    df.index.name = None
    df['Temperature'] = df['Temperature'].astype(float)
    return df

df = load_data()
df = preprocess_data(df)


def predict_future(model, start_year, end_year):
    future_years = np.array(range(start_year, end_year + 1)).reshape(-1, 1)
    predictions = model.predict(future_years)
    predictions = np.round(predictions, 2)
    return future_years, predictions
    
def train_model1(df):
    X = df[["Year"]]
    y = df["Temperature"]
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_model(df):
    X = df[["Year"]]
    y = df["Temperature"]
    model = RandomForestRegressor()
    model.fit(X, y)
    return model


def main():
    st.title("Climate Change Analysis: US Temperature Trends")

    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    # Display data
    st.subheader("Historical Data")
    # Train the model
    model = train_model(df)

    # Predict future temperatures
    start_year = 2024
    end_year = 2050
    future_years, predictions = predict_future(train_model1(df), start_year, end_year)

    # Plot historical and predicted data
    st.subheader("Temperature Trends")
    plt.figure(figsize=(10, 6))
    plt.plot(df["Year"], df["Temperature"], label="Historical Data")
    plt.plot(future_years, predictions, label="Predicted Data", linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    st.pyplot(plt)

    # Future temperature predictions
    st.subheader("Predictions")
    predictions_df = pd.DataFrame({"Year": future_years.flatten(), "Predicted Temperature": predictions})
    col1, col2 = st.columns(2)

# Display the first table in the first column
    with col1:
        st.write("Temp over last Century")
        st.dataframe(df)

    # Display the second table in the second column
    with col2:
        st.write("Temp Predictions for the next 20 years") 
        st.dataframe(predictions_df)

if __name__ == "__main__":
    main()
