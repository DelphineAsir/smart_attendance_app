import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import time

st.set_page_config(page_title='Smart Attendance System',layout='wide')
st.header('SMART ATTENDANCE FACE RECOGNITION SYSTEM')
st.text(' ')

with st.spinner("Connecting to Redis db ..."):
    import recognition

db_name= 'face:db'
    
tab1,tab2 = st.tabs(['Main', ' '])
with tab1:
    st.success("Database and Models have been connected SUCCESSFULLY!!!")    
    st.subheader('Real-Time Face Prediction')

    # Retrieve data from Redis db
    redis_face_db = recognition.retrieve_data(name=db_name)
    # time in sec
    screen_waitTime = 10 
    startTime = time.time()
    prediction = recognition.realPrediction() 

    # Real Time Prediction   
    def video_frame_callback(frame):
        global startTime
        # 3d numpy array
        input_Image = frame.to_ndarray(format="bgr24") 
        # operation that you can perform on the array
        image_predicted = prediction.face_prediction(input_Image,redis_face_db,
                                            'facial_features',['Name','Role'],thresh=0.5)        
        timenow = time.time()
        difference = timenow - startTime
        if difference >= screen_waitTime:
            prediction.saveLogs_redis()
            startTime = time.time()       
            print('Save prediction to redis db')        

        return av.VideoFrame.from_ndarray(image_predicted, format="bgr24")

    webrtc_streamer(key="Prediction", video_frame_callback=video_frame_callback
                    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
                    


