import streamlit as st
from Main import recognition
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

st.subheader('Registration Form')

## init registration form
registration_form = recognition.NewRegistration()

# Collect person name and role
p_name = st.text_input(label='Name',placeholder='First & Last Name')
role = st.selectbox(label='Select your Role',options=('Student',
                                                      'Teacher'))

# Collect facial embedding
def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24') # 3d array bgr
    reg_img, embedding = registration_form.get_embedding(img)
    #  save data into local computer txt
    if embedding is not None:
        with open('embedding.txt',mode='ab') as f:
            np.savetxt(f,embedding)
    
    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')

webrtc_streamer(key='registration',video_frame_callback=video_callback_func,
                rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })


# save the data in redis database
if st.button('Submit'):
    return_val = registration_form.save_data_in_redis_db(p_name,role)
    if return_val == True:
        st.success(f"{p_name} registered successfully")
    elif return_val == 'name_false':
        st.error('Please enter the name: Name cannot be empty or spaces')        
    elif return_val == 'file_false':
        st.error('embeddings.txt is not found. Please refresh the page and execute again.')
        
