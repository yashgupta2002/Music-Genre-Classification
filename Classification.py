
# In[1]:


import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

#Preprocesing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# from ScaleTransform import scaler
scaler = StandardScaler()

# In[2]:


data = pd.read_csv(r'C:\Users\Dell\Desktop\Machine learning Robo project\data.csv')
data.head()


# In[3]:


data = data.drop(['filename'],axis=1)
data.head()


# In[4]:


data.info()


# In[5]:


genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
print(y)


# In[6]:



X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))


# In[7]:





# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40,random_state=100)


# In[9]:


X_test


# In[10]:


from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)


# In[11]:


prediction = model.predict(X_test)
type(prediction)


# In[12]:


prediction[170]


# In[13]:


from sklearn.metrics import classification_report,confusion_matrix


# In[14]:


print(classification_report(y_test,prediction))


# In[15]:


print(confusion_matrix(y_test,prediction))


# In[24]:
# input_file = r'C:\Users\Dell\Desktop\Machine learning Robo project\genres\blues\blues.00005.wav'


# # In[25]:


# y, sr = librosa.load(input_file, mono=True, duration=30)
# chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
# rmse = librosa.feature.rms(y=y)
# spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
# spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
# rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
# zcr = librosa.feature.zero_crossing_rate(y)
# mfcc = librosa.feature.mfcc(y=y, sr=sr)

# arr = np.array([[np.mean(chroma_stft), np.var(chroma_stft) ,np.mean(rmse) ,np.var(rmse) ,np.mean(spec_cent) ,np.var(spec_cent) ,np.mean(spec_bw) ,np.var(spec_bw) ,np.mean(rolloff) ,np.var(rolloff) ,np.mean(zcr) ,np.var(zcr)]])
# for e in mfcc:
#     arr=np.append(arr,np.mean(e))
# ar=np.array([arr])  


# # In[26]:


# arr.shape


# # In[27]:


# type(ar)


# # In[28]:


# len(ar[0])


# # In[29]:

# a=scaler.transform(np.array(ar,dtype=float))



# # In[30]:

# # model = pickle.load(open(r'C:\Users\Dell\Desktop\Machine learning Robo project\model.pkl','rb'))
# pred=model.predict(a)


# # In[31]:


# if(pred==0):
#     print("Blue")
# elif(pred==1):
#     print("Classical")
# elif(pred==2):
#     print("Country")
# elif(pred==3):
#     print("Disco")
# elif(pred==4):
#     print("Hiphop")
# elif(pred==5):
#     print("Jazz")
# elif(pred==6):
#     print("Metal")
# elif(pred==7):
#     print("Pop")
# elif(pred==8):
#     print("Reggae")
# elif(pred==9):
#     print("Rock")

# 
pickle.dump(model, open(r'C:\Users\Dell\Desktop\Machine learning Robo project\model.pkl','wb'))

pickle.dump(scaler, open(r'C:\Users\Dell\Desktop\Machine learning Robo project\Scale.pkl','wb'))


# model = pickle.load(open(r'C:\Users\Dell\Desktop\Machine learning Robo project\model.pkl','rb'))
# print(model)
# print(model.predict(r'C:\Users\Dell\Desktop\Machine learning Robo project\genres\rock\rock.00005.wav'))

# In[ ]:







# In[ ]:




