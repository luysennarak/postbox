
# coding: utf-8

# In[32]:


import numpy as np
from os.path import isfile,join
from os import listdir
from scipy import signal as sg
from scipy import hamming
import soundfile as sf
from python_speech_features import mfcc
from python_speech_features import logfbank
import librosa
from scipy.fftpack import fft
from scipy.io import wavfile
import soundfile
import math
import tensorflow
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers.convolutional import*
from keras.layers.core import*
from keras import backend
from keras.layers import Embedding
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardbScaler
from pyAudioAnalysis import audioFeatureExtraction as aFE
from pyAudioAnalysis import audioBasicIO as aIO

# In[33]:

babycry=[f for f in listdir('./train data/301 - Crying baby') if isfile(join('./train data/301 - Crying baby', f))]
babysilence=[f for f in listdir('./train data/baby_cry/901 - Silence') if isfile(join('./train data/baby_cry/901 - Silence', f))]
babynoise=[f for f in listdir('./train data/baby_cry/902 - Noise') if isfile(join('./train data/baby_cry/902 - Noise', f))]
babylaugh=[f for f in listdir('./train data/baby_cry/903 - Baby laugh') if isfile(join('./train data/baby_cry/903 - Baby laugh', f))]


# In[34]:

baby_cry=[]
for each in range(0,len(babycry)) :
    cry='./train data/301 - Crying baby' + '/' + babycry[each]
    baby_cry.append(cry)


# In[35]:

baby_silence=[]
for each in range(0,len(babysilence)) :
    silence='./train data/baby_cry/901 - Silence' + '/' + babysilence[each]
    baby_silence.append(silence)


# In[36]:

baby_noise=[]
for each in range(0,len(babynoise)) :
    noise='./train data/baby_cry/902 - Noise' + '/' + babynoise[each]
    baby_silence.append(noise)


# In[37]:

baby_laugh=[]
for each in range(0,len(babylaugh)) :
    laugh='./train data/baby_cry/903 - Baby laugh' + '/' + babylaugh[each]
    baby_laugh.append(laugh)
  


# In[38]:

nonbabycry=baby_silence+baby_noise+baby_laugh


# In[52]:

def VAD(frate, signal):
    win = int(0.025 * frate)
    stFeats = aFE.stFeatureExtraction(signal, frate, win, win)
    energy = stFeats[1, :]
    eth = (np.max(energy) + np.min(energy))/2

    newSignal = []
    for sig in signal:
        if sig > eth: newSignal.append(sig)

    return newSignal


def newFeatures(inFile, isVAD):
    # read audio data from file
    #[Fs, x] = aIO.readAudioFile(inFile)
    x, Fs = librosa.load(inFile, sr=16000) #sf.read(inFile)

    # VAD
    if isVAD:
        x = VAD(Fs, x)

    # window and overlap size
    win = int(0.025 * Fs)
    step = int(0.010 * Fs)

    # get short-time features
    Feats = aFE.stFeatureExtraction(x, Fs, win, step)

    # saveFeats(Feats)
    Feats = np.transpose(Feats[8:21, :])

    newFeat = []
    for row in range(len(Feats)):
        newFeat.append(Feats[row, :])

    return newFeat

def Mfcc(audiofile):
    p_s,r=sf.read(audiofile)
    s = VAD(r, p_s)

    x=np.mean(librosa.feature.mfcc(y=s,sr=r, n_mfcc=40).T,axis=0)
    return x


# In[53]:

lengthofFeat = 14
dim = (len(baby_cry+nonbabycry),lengthofFeat)
mydata = np.zeros(dim)


# In[69]:
# crying class
for i in range(0,len(baby_cry)):
    data=[]
    print("read %s..." % baby_cry[i])
    #mfc=Mfcc(baby_cry[i])
    feat = newFeatures(baby_cry[i], True)
    feat1 = np.mean(feat, axis=0)
    for j in feat1:
        data.append(j)
    data.append(1)
    mydata[i,:] = data


# other class
all=baby_cry+nonbabycry
for i in range(len(baby_cry)-1,len(all)-1):
     data=[]
     print("read %s..." % all[i])
     feat = newFeatures(all[i], False)
     feat1 = np.mean(feat, axis=0)
     for j in feat1:
         data.append(j)
     data.append(0)
     mydata[i, :] = data


# In[82]:

input=mydata[:,0:13]
output=mydata[:,13]


# In[83]:

print("start training...")
mymodel=Sequential()
mymodel.add(Dense(32, input_dim = 13, init='uniform', activation ='relu'))
mymodel.add(Dense(16,init='uniform', activation='relu'))
mymodel.add(Dense(1, init='uniform', activation='sigmoid'))


# In[84]:

mymodel.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[85]:

mymodel.fit(input,output, epochs=500, batch_size=20, verbose=2)


# In[86]:
print("evaluate...")
myrmse=mymodel.evaluate(input,output)


# In[87]:
print("save json and model...")
with open('hmodel.json','w') as f:
    f.write(mymodel.to_json())


# In[88]:

mymodel.save_weights("hmodelweight.h5")

print("successfully finished.")