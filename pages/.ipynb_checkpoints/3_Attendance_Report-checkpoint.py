import streamlit as st 
from Main import recognition
from streamlit_webrtc import webrtc_streamer
import av
import time

st.subheader('Report')

log_name = 'attendance:logs'

def load_logs(log_name,end=-1):
    logs_list = recognition.r.lrange(log_name,start=0,end=end) # extract all data from the redis database
    return logs_list

logs_list = load_logs(log_name = log_name)

