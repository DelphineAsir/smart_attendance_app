import numpy as np
import pandas as pd
import cv2
import redis

import time
from datetime import datetime
import os

from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

db_name= 'face:db'
log_name = 'logs:db'
#db_name='airene:asir'

#r = redis.Redis.from_url(url="redis://:UK3uSEHS5lEsrbtf014zVwLDPD6Ds9In@redis-11588.c326.us-east-1-3.ec2.cloud.redislabs.com:11588")
# Connect to Redis Client
hostname = 'redis-17629.c270.us-east-1-3.ec2.cloud.redislabs.com'
port = 17629
password = 'esLogs7keVlsi213Hdwri9AJniVYFahL'

r = redis.StrictRedis(host=hostname,
                      port=port,
                      password=password)

# Retrieve Data from database
def retrieve_data(name):
    retrieve_dict= r.hgetall(name)
    retrieve_series = pd.Series(retrieve_dict)
    retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrieve_series.index
    index = list(map(lambda x: x.decode(), index))
    retrieve_series.index = index
    retrieve_df =  retrieve_series.to_frame().reset_index()
    retrieve_df.columns = ['name_role','facial_features']
    retrieve_df[['Name','Role']] = retrieve_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrieve_df[['Name','Role','facial_features']]


# configure insightFace model
insightFace = FaceAnalysis(name='buffalo_sc',root='insightface_model', providers = ['CPUExecutionProvider'])
insightFace.prepare(ctx_id = 0, det_size=(640,640), det_thresh = 0.5)

# ML Search Algorithm
def ml_search_algorithm(dataframe,feature_column,test_vector,
                        name_role=['Name','Role'],thresh=0.5):
    """
    cosine similarity base search algorithm
    """
    # get dataframe
    dataframe = dataframe.copy()
    # face embedding dataframe and convert into array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)
    
    # calculate cosine similarity
    cosine_similar = pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    cosine_array = np.array(cosine_similar).flatten()
    dataframe['cosine'] = cosine_array

    # filter the data using thresh
    filtered_data = dataframe.query(f'cosine >= {thresh}')
    if len(filtered_data) > 0:
        # get the person name
        filtered_data.reset_index(drop=True,inplace=True)
        argmax = filtered_data['cosine'].argmax()
        p_name, p_role = filtered_data.loc[argmax][name_role]
        
    else:
        p_name = 'Unknown'
        p_role = 'Unknown'
        
    return p_name, p_role


# Real Time Prediction
class realPrediction:
    def __init__(self):
        self.logs = dict(name=[],role=[],current_time=[])
        
    def reset_dict(self):
        self.logs = dict(name=[],role=[],current_time=[])
        
    def saveLogs_redis(self):
        #  create a logs dataframe
        dataframe = pd.DataFrame(self.logs)        
        #  drop the duplicate name
        dataframe.drop_duplicates('name',inplace=True) 
        # push data to redis list)      
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []
        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)
                
        if len(encoded_data) >0:
            r.lpush(log_name,*encoded_data)        
                    
        self.reset_dict()     
        
        
    def face_prediction(self,test_image, dataframe,feature_column,
                            name_role=['Name','Role'],thresh=0.5):        
        current_time = str(datetime.now())        
        # apply insight face model on test image
        results = insightFace.get(test_image)
        test_copy = test_image.copy()
        # extract each embedding and pass to ml_search_algorithm
        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            p_name, p_role = ml_search_algorithm(dataframe,
                                                        feature_column,
                                                        test_vector=embeddings,
                                                        name_role=name_role,
                                                        thresh=thresh)
            if p_name == 'Unknown':
                color =(0,0,255) # bgr
            else:
                color = (0,255,0)

            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
            text_gen = p_name
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            # save info in logs dict
            self.logs['name'].append(p_name)
            self.logs['role'].append(p_role)
            self.logs['current_time'].append(current_time)          

        return test_copy


#Registration Form
class NewRegistration:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0
        
    def get_embedding(self,frame):
        # get results from insightface model
        results = insightFace.get(frame,max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
            # put text samples info
            text = f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            
            # facial features
            embeddings = res['embedding']

        return frame, embeddings
    
    def save_data_in_redis_db(self,name,role):
        # validation name
        if name is not None:
            if name.strip() != '':
                key = f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'name_false'
        
        # if embedding.txt exists
        if 'embedding.txt' not in os.listdir():
            return 'file_false'
                
        # load "embedding.txt"
        x_array = np.loadtxt('embedding.txt',dtype=np.float32) # flatten array            
        
        # convert into array (proper shape)
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples,512)
        x_array = np.asarray(x_array)       
        
        # calculate mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()
        
        # save this into redis database       
        r.hset(name=db_name ,key=key,value=x_mean_bytes)
        
        # remove old file
        #os.remove('embedding.txt')
        try:
          os.system('rm embedding.txt')
          print("File deleted successfully")
        except Exception as e:
          print(f"Error deleting file: {e}")
        self.reset()
        
        return True
    
    def delete_data(name,role):
        # validation name
        if name is not None:
            if name.strip() != '':
                key_to_del = f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'name_false'
        
        r.hdel(db_name,key_to_del)
        return True
