import streamlit as st
import pandas as pd
import altair as alt

from io import StringIO

def merge_dataframes(dataframe_1, dataframe_2):
    # add suffix to columns to differentiate between the two dataframes
    dataframe_1.columns = [str(col) + '_1' for col in dataframe_1.columns]
    dataframe_2.columns = [str(col) + '_2' for col in dataframe_2.columns]
    merged_dataframe = pd.concat([dataframe_1, dataframe_2], axis=1)
    merged_dataframe.drop(columns=["timestamp_1", "timestamp_2", 'quaternion.accuracy_1', 'quaternion.accuracy_2'], inplace=True)
    return merged_dataframe

st.title("EOlab stroke detection")
st.write("This is a simple web app to give a prediction on the stroke type")
st.write("Upload files from both sensors")
uploaded_file_1 = st.file_uploader("Your left sensor file here", type=["csv", "txt"])
uploaded_file_2 = st.file_uploader("Your right sensor file here", type=["csv", "txt"])

def create_altchart(dataframe,column_names):
    chart=alt.Chart(dataframe).mark_line().encode(
        x=dataframe.index,
        y=column_names
    ).interactive()
    st.altair_chart(chart, use_container_width=True)


def create_Chart(dataframe,column_names):

    st.line_chart(dataframe[column_names])
    
try:
    if uploaded_file_1 is not None and uploaded_file_2 is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe_1 = pd.read_csv(uploaded_file_1)
        dataframe_2 = pd.read_csv(uploaded_file_2)
        merge_df=merge_dataframes(dataframe_1, dataframe_2)
        # create_Chart(merge_df,["fPressureSide_1","fPressureSide_2"])
        create_Chart(merge_df,["fPressureFront_1","fPressureFront_2"])
        create_altchart(merge_df,["fPressureFront_1","fPressureFront_2"])
except Exception:
    st.write(Exception)
    st.write("Please upload valid files")

