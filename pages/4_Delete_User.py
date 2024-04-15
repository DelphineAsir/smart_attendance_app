import streamlit as st
from Main import recognition
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

st.subheader('Delete User')
#del_form = recognition.RegistrationForm()
#db_name='airene:asir'
db_name= 'face:db'
log_name = 'logs:db'

tab1,tab2 = st.tabs(['Delete User', ' Delete ALL Logs'])
with tab1:
    p_name = st.text_input(label=' User Name',placeholder='First & Last Name')
    role = st.selectbox(label='Select your Role',options=('Student',
                                                        'Teacher'))

    key_to_del = f'{p_name}@{role}'
    if st.button('Submit'):
        return_val =  recognition.r.hdel(db_name,key_to_del)
        if return_val == True:
            st.success(f"{p_name} Deleted successfully")
        elif return_val == False:
            st.error('Please enter the correct name')
        
with tab2:
    if st.button('Delete ALL Logs'):
        return_val = recognition.r.lrem(log_name, 1, 0)
        st.write(return_val)
        st.success("All logs from Redis db deleted sucessfully.")

    st.warning(" Warning:  If you click this button ALL logs will be deleted!!!")