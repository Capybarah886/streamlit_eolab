import streamlit as st
import pandas as pd
import altair as alt
import torch
import io
from inference import InferenceModel
from transformer import TransformerModel



st.title("EOlab stroke detection")
st.write("This is a simple web app to give a prediction on the stroke type")
st.write("Upload files from both sensors")
uploaded_file_1 = st.file_uploader("Your left sensor file here", type=["csv", "txt"])
uploaded_file_2 = st.file_uploader("Your right sensor file here", type=["csv", "txt"])

def create_Chart(dataframe, column_names):
    chart = alt.Chart(dataframe).mark_line().encode(
        x=dataframe.index,
        y=column_names
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

def create_Chart(dataframe, column_names):
    st.line_chart(dataframe[column_names])

try:
    if uploaded_file_1 is not None and uploaded_file_2 is not None:
        dataframe_1 = pd.read_csv(uploaded_file_1)
        dataframe_2 = pd.read_csv(uploaded_file_2)
        # Load the model
        try:
            with open("./model.pth", "rb") as f:
                buffer = io.BytesIO(f.read())
            model = torch.load(buffer)
            st.write('Model loaded successfully')
        except Exception as e:
            st.write(f"Error loading model: {e}")
            raise
        
        # Initialize inference model
        try:
            inference_model = InferenceModel(model, dataframe_1, dataframe_2)
            result = inference_model.inference()
            st.write("Inference successful")
            st.write(result)
        except Exception as e:
            st.write(f"Error initializing inference model: {e}")
            raise
        
except Exception as e:
    st.write(f"An error occurred")
    st.write("Inference model not loaded")