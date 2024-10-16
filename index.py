import streamlit as st
import pandas as pd
import altair as alt
import torch
import io
from inference import InferenceModel
from transformer import TransformerModel
from sktime.utils import mlflow_sktime  
# from model_loader import load_model
from utils import combined_flow
import mlflow
import os

st.title("EOlab stroke detection")
st.write("This is a simple web app to give a prediction on the stroke type")
st.write("Upload files from both sensors")
model_type = st.radio(
    "Which your model?",
    [":rainbow[Rocket model(sktime)]", "***Torch model(.pth)***"],
    index=0,
)
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
            if model_type == ":rainbow[Rocket model(sktime)]":
                # Ensure model URI is correct
                model_uri = './sktime_models'
                if not os.path.exists(model_uri):
                    st.write(f"Model URI does not exist: {model_uri}")
                    raise Exception(f"Model URI does not exist: {model_uri}")
                else:
                    st.write("Model URI exists")
                # Load the sktime model using mlflow
                model = mlflow_sktime.load_model(model_uri=model_uri)
                print(model.get_params())
                st.write('Model loaded successfully')
            else:
                with open("./models/model.pth", "rb") as f:
                    buffer = io.BytesIO(f.read())
                model = torch.load(buffer)
                st.write('Model loaded successfully')
        except Exception as e:
            st.write(f"Error loading model: {e}")
            raise
        
        # Initialize inference model
        try:
            if model_type == ":rainbow[Rocket model(sktime)]":
                # add a loading bar
                with st.spinner("Inference in progress"):
                        
                    results=combined_flow(dataframe_1, dataframe_2, model)
                    st.write("Inference successful")
                    stroke_dic={0:'Freestyle', 1:'Backstroke', 2:'Butterfly'}
                    results = [stroke_dic[i] for i in results]
                    st.write('There are ', len(results), 'batch results')
                    st.write(results)                    
                    st.write("Among them, the stroke with the highest frequency is: ", max(set(results), key = results.count))
            else:
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