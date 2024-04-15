import streamlit as st 
from Main import recognition
from streamlit_webrtc import webrtc_streamer
import av
import time
import pandas as pd

st.subheader('Report')
db_name= 'face:db'
log_name = 'logs:db'

def load_logs(log_name,end=-1):
    logs_list = recognition.r.lrange(log_name,start=0,end=end) 
    return logs_list

logs_list = load_logs(log_name = log_name)

byte_tostring = lambda x:x.decode('UTF-8')

string_list = list(map(byte_tostring,logs_list))

split_string =  lambda x:x.split('@')
string_nestlist = list(map(split_string,string_list))

list_df = pd.DataFrame(string_nestlist,columns=['Name','Role','Timestamp'])
list_df['Timestamp'] = pd.to_datetime(list_df['Timestamp'])
list_df['Date'] = list_df['Timestamp'].dt.date
list_df['Time'] = list_df['Timestamp'].dt.time

#calculate intime and outtime
final_df = list_df.groupby(by=['Date','Name','Role']).agg(
                    intime = pd.NamedAgg('Timestamp','min'),
                    outtime = pd.NamedAgg('Timestamp','max')
).reset_index()

final_df['intime'] = pd.to_datetime(final_df['intime'])
final_df['outtime'] = pd.to_datetime(final_df['outtime'])
final_df['Duration'] = final_df['outtime']-final_df['intime']

allDates = final_df['Date'].unique()
nameRole = final_df[['Name','Role']].drop_duplicates().values.tolist()
merge_df =[]
for df in allDates:
    for name, role in nameRole:
        merge_df.append([df,name,role])

merge_df =pd.DataFrame(merge_df, columns=['Date','Name','Role'])
merge_df =pd.merge(merge_df,final_df,how='left',on=['Date','Name','Role'])
merge_df['Seconds']=merge_df['Duration'].dt.seconds
merge_df['Hours'] = merge_df['Seconds'] /(60*60)

def marker(x):
    if pd.Series(x).isnull().all():
        return 'Absent'
    else:
        return 'Present'
    
merge_df['Status'] = merge_df['Hours'].apply(marker)

st.dataframe(merge_df[['Date','Name','Role','Duration','Status']])