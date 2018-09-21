# sudo pip3 install pysoundfile
# sudo pip3 install python_speech_features --upgrade
# sudo apt-get install python3-pyaudio

import os
import numpy as np
import math
import threading
import pyaudio
import wave
import soundfile as sf
import librosa
import numpy
import time
from scipy.io import wavfile
from scipy import signal as sg
from scipy import hamming
import tensorflow
import librosa
from keras.models import model_from_json
from python_speech_features import mfcc
from python_speech_features import logfbank
from scipy import signal
from scipy.fftpack import fft
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from pyAudioAnalysis import audioFeatureExtraction as aFE
from pyAudioAnalysis import audioBasicIO as aIO

fs=16000
#MAX_INT = 32768.0
lowpass = 300 # Remove lower frequencies.
highpass = 8000 # Remove higher frequencies.


def STE(signal):
    return sum([abs(x)**2 for x in signal])

def ZCR(signal):
    sign=np.sign(signal)
    sign[sign==0]=-1
    return len(np.where(np.diff(sign))[0])/len(signal)

def freqpitch(signal):
    #signal=signal[1,:]
    ms1=fs/1000 #maximum freq at 1000hz
    ms2=fs/50 #minimumfrequency at 50hz
    Y=fft(signal*sg.hamming(len(signal)))
    u=abs(Y)+2.2204e-16
    C=[]
    for i in u:
        c=math.log(i)
        C.append(c)
    C=fft(C)
    z=abs(C[int(math.floor(ms1)):int(math.floor(ms2))+1])
    z=list(z)
    maxamp_pitch=max(z)
    fx=z.index(maxamp_pitch)
    freqpitch=fs/(ms1+fx-1)
    return [freqpitch, maxamp_pitch]

def MFCC(signal):
    mfcc_feat=mfcc(signal,fs, winlen=0.025,winstep=0.01, numcep=13, nfilt=26, nfft=1103,lowfreq=0,highfreq=None, ceplifter=22, appendEnergy=True)
    l= np.asarray(mfcc_feat,dtype='float32')
    if len(l)>499:
        n=len(l)-499
        l=l[:-n,:]
    a=np.zeros((1,13))
    if len(l)<499:
        n=499-len(l)
        for i in range(0,n):
            l=np.concatenate((l,a))   
        
    flat=[x for sublist in l for x in sublist]
    return flat

def LOGFBANK(signal,fs):
    #signal, fs =sf.read(audiofile)
    fbank_feat=logfbank(signal,fs, nfft=1103)
    l= np.asarray(fbank_feat,dtype='float32')
    if len(l)>499:
        n=len(l)-499
        l=l[:-n,:]
    a=np.zeros((1,26))
    if len(l)<499:
        n=499-len(l)
        for i in range(0,n):
            l=np.concatenate((l,a)) 
    flat=[x for sublist in l for x in sublist]
    return flat

def pw(signal):
    amplitude=max(signal)
    db=20*math.log10(amplitude)
    return db

def Mfcc(audiofile):
    #s,r=soundfile.read(audiofile)
    x=np.mean(librosa.feature.mfcc(y=audiofile,sr=44100, n_mfcc=40).T,axis=0)
    return x

def removeNoise(orig_signal, fs):
    b, a = signal.butter(5, 800 / (fs / 2), btype='highpass')  # ButterWorth filter 4350
    filteredSignal = signal.lfilter(b, a, orig_signal)

    c, d = signal.butter(5, 300 / (fs / 2), btype='lowpass')  # ButterWorth low-filter
    newFilteredSignal = signal.lfilter(c, d, filteredSignal)  # Applying the filter to the signal

    return filteredSignal

def Loudness(orig_signal):
    max_val1 = numpy.max(orig_signal)
    max_val2 = numpy.min(orig_signal)
    max_val2 = numpy.fabs(max_val2)
    if max_val1 < max_val2:
        max_val1 = max_val2

    # normalize
    new_signal = orig_signal * (1.0 / max_val1)
    return new_signal

with open('hmodel.json','r') as f:
    mymodel=model_from_json(f.read())

mymodel.load_weights("hmodelweight.h5")


def VAD(frate, signal):
    return signal


def newFeatures(Fs, signal):

    # VAD
    x = VAD(Fs, signal)

    # window and overlap size
    win = int(0.025 * Fs)
    step = int(0.010 * Fs)

    # get short-time features
    Feats = aFE.stFeatureExtraction(x, Fs, win, step)

    # saveFeats(Feats)
    Feats = np.transpose(Feats[8:21, :])
    #ff0 = Feats[8:21, :]
    #Feats = np.reshape(ff0,(len(ff0[0]), 14))

    newFeat = []
    for row in range(len(Feats)):
        newFeat.append(Feats[row, :])

    return np.mean(newFeat, axis=0)


def doafter5():
    l=None
    livesound=None
    l=pyaudio.PyAudio()
    livesound=l.open(format=pyaudio.paInt16,
                 channels=1,
                 rate=fs,input=True,frames_per_buffer=8192
                     )
    livesound.start_stream() 
    Livesound=None
    li=[]
    
    timeout=time.time()+20
    for f in range(0,int(fs/8192*2)):
        Livesound=livesound.read(8192)
        li.append(Livesound)
        
   
    waves=wave.open('rec.wav','w')
    waves.setnchannels(1)
    waves.setsampwidth(l.get_sample_size(pyaudio.paInt16))
    waves.setframerate(fs)
    waves.writeframes(b''.join(li))
    waves.close()

    l.terminate()

    fs1 = 16000
    livesignal,fsd=librosa.load('rec.wav', sr=fs1) #sf.read('rec.wav')

    #noNoiseSignal = removeNoise(livesignal, fs)

    #livesignal = Loudness(livesignal)


    newdata=[]
    #newdata.append(STE(livesignal))
    #newdata.append(ZCR(livesignal))
    feats=newFeatures(fsd, livesignal)
    for j in feats:
        newdata.append(j)
    #lfb=LOGFBANK(livesignal,fs)
    #for t in lfb:
    #        newdata.append(t)         
    #f=freqpitch(livesignal)
    #newdata.append(f[0])
    #newdata.append(f[1])

    newdata=np.reshape(newdata,(1,13))

   
    soundclass=int(mymodel.predict_classes(newdata))
   
    #if soundclass==1:
    #    os.system('python /home/pi/Downloads/gsmsendsms.py')
    print('Detecting......')
    if soundclass == 1:
        print("baby is crying now.")
    os.remove('rec.wav')

    threading.Timer(2.0, doafter5).start()

if __name__ == '__main__':
    doafter5()

