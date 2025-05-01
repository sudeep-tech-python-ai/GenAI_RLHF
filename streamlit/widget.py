import streamlit as st
import pandas as pd

st.title("Stream Text Input")
#Take user input
name=st.text_input("Enter Your name:")

#create a slider
age=st.slider("select your age:",0,100,25)

st.write(f"Your age is {age}.")

options=['Python','Java','C++','JavaScript']
# creating a selection box
choice=st.selectbox("Choose your favourite language:",options)
st.write(f"You selected {choice}.")

if name:
    st.write(f"Hello, {name}")


uploaded_file=st.file_uploader("Choose a CSV file",type='csv')
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write(df)


