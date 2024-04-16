import streamlit as st 
from Main import recognition

st.subheader('View Logs and Database ')

tab1, tab2 = st.tabs(['Registered Data','Logs'])

#db_name = 'airene:asir'
db_name= 'face:db'
log_name = 'logs:db'

def load_db(db_name):
    with st.spinner('Retrieving Data from Redis DB ...'):           
        redis_face_db = recognition.retrieve_data(db_name)
        st.dataframe(redis_face_db)            

def load_logs(log_name,end=-1):
    logs_list = recognition.r.lrange(log_name,start=0,end=end) 
    return logs_list

with tab1:    
    if st.button('Refresh Data'):
       load_db(db_name)
       st.success("Data retrived from Redis sucessfully")

with tab2:
    if st.button('Refresh Logs'):
        st.write(load_logs(log_name=log_name))
        st.success("Logs retrived from Redis sucessfully")
    
        

