import streamlit as st
import pandas as pd
import altair as alt
import torch
import io
import plotly.express as px
from torch_models import LSTMModel

from inference import InferenceModel
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
                with open("./models/lstm_model.pth", "rb") as f:
                    buffer = io.BytesIO(f.read())
                model = torch.load(buffer, map_location=torch.device('cpu'))
                model = LSTMModel(18,256,3)  # Initialize the model instance
                model.eval()  # Set the model to evaluation mode

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
                    result = [stroke_dic[i] for i in results]
                    st.write('There are ', len(result), 'batch results')
                    st.write('There are', result.count('Freestyle'), 'Freestyle')
                    st.write('There are', result.count('Backstroke'), 'Backstroke')
                    st.write('There are', result.count('Butterfly'), 'Butterfly')
                    # add a pie chart using plotly
                    fig = px.pie(values=[result.count('Freestyle'), result.count('Backstroke'), result.count('Butterfly')], names=['Freestyle', 'Backstroke', 'Butterfly'])
                    st.plotly_chart(fig, use_container_width=True)
                    max_stroke = max(result, key=result.count)
                    st.write('The prediction of the stroke type is:', max_stroke)
            else:
                inference_model = InferenceModel(model, dataframe_1, dataframe_2)
                result = inference_model.inference()
                st.write("Inference successful")
                stroke_counts = {'Freestyle': 2, 'Butterfly': 1, 'Backstroke': 0}
                for stroke in result['batch_result']:
                    if stroke in stroke_counts:
                        stroke_counts[stroke] += 1

                fig = px.pie(values=list(stroke_counts.values()), names=list(stroke_counts.keys()))
                st.plotly_chart(fig, use_container_width=True)

                max_stroke = max(stroke_counts, key=stroke_counts.get)
                st.write('The prediction of the stroke type is:', max_stroke)
        except Exception as e:
            st.write(f"Error initializing inference model: {e}")
            raise
        
except Exception as e:
    st.write(f"An error occurred")
    st.write("Inference model not loaded")