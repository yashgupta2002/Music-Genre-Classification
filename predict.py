import librosa
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
model = pickle.load(open(r'C:\Users\Dell\Desktop\Machine learning Robo project\model.pkl','rb'))
scaler = pickle.load(open(r'C:\Users\Dell\Desktop\Machine learning Robo project\Scale.pkl','rb'))



# In[25]:
def predict(file_path):

    y, sr = librosa.load(file_path, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    arr = np.array([[np.mean(chroma_stft), np.var(chroma_stft) ,np.mean(rmse) ,np.var(rmse) ,np.mean(spec_cent) ,np.var(spec_cent) ,np.mean(spec_bw) ,np.var(spec_bw) ,np.mean(rolloff) ,np.var(rolloff) ,np.mean(zcr) ,np.var(zcr)]])
    for e in mfcc:
        arr=np.append(arr,np.mean(e))
    ar=np.array([arr])  

    a=scaler.transform(np.array(ar,dtype=float))


    pred=model.predict(a)

    if(pred==0):
        return "Blue"
    elif(pred==1):
        return "Classical"
    elif(pred==2):
        return "Country"
    elif(pred==3):
        return "Disco"
    elif(pred==4):
        return "Hiphop"
    elif(pred==5):
        return "Jazz"
    elif(pred==6):
        return "Metal"
    elif(pred==7):
        return "Pop"
    elif(pred==8):
        return "Reggae"
    elif(pred==9):
        return "Rock"
    else:
        return "Sorry...!Not able to predict"





