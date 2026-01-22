import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Exploratory Data Analysis (Phase 1)")

df = pd.read_csv("data/processed/cleaned.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Adjusted Close Price Distribution")
fig, ax = plt.subplots()
sns.kdeplot(df["adj_close"], ax=ax, fill=True)
st.pyplot(fig)

st.subheader("Price vs Volume")
fig, ax = plt.subplots()
sns.scatterplot(data=df.sample(5000), x="volume", y="adj_close", ax=ax)
st.pyplot(fig)
