import sys
import time
import os
import glob
import numpy
import math

from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
import matplotlib.pyplot as plt
from . import audioBasicIO
from . import utilities
from scipy.signal import lfilter, hamming
import librosa
#from scikits.talkbox import lpc

import tkinter as Tkinter
#root = Tkinter.Tk()
#import tkSnack
#tkSnack.initializeSnack(root)


#reload(sys)
#sys.setdefaultencoding('utf8')

eps = 0.00000001

""" Time-domain audio features """
class librosa_audio_feature():
    sample_rate = 8000             # Sample rate, defualt 16000 Hz
    feat_len = 40               # Feature size such as MFCC or Mel spectrogram energy
    frame_step = 80                # Frame step samples, 160(10ms) for 16000 Hz
    frame_win  = 200                # Frame window size samples, 400(25ms) for 16000 Hz
    fft_win  = 256                  # FFT window size samples, 512(32 ms) for 16000 Hz

    # Mfcc option
    include_delta = True
    include_acceleration = True
    include_mfcc0 = True

    __real_feat_len = 0
    __feature_type_ = 2             # 1: MFCC, 2: Mel filter bank energy

    @classmethod
    def get_mfcc_matrix(cls, wav=[], sr=0, feat_len=0, fft_win=0, hop_len=0, include_delta=True, include_acceleration=True, include_mfcc0=True):
        #extract mfcc
        S = librosa.power_to_db(librosa.feature.melspectrogram(y=wav, sr=sr,n_fft=fft_win, hop_length=fft_win+1))
        mfcc = librosa.feature.mfcc(wav, sr, S=S, n_mfcc=feat_len)
        #print("file name: {}".format(file_name))
        return mfcc.T[0]

    @classmethod
    def get_melfilter_bank_matrix(cls, wav=[], sr=0, feat_len=0, fft_win=0, hop_len=0):
        # set feature information
        n_mels = feat_len
        htk = False
        fmax = None

        S = librosa.power_to_db(librosa.feature.melspectrogram(y=wav, sr=sr,n_fft=fft_win, hop_length=fft_win+1))
        mel_basis = librosa.filters.mel(sr=sr, n_fft=fft_win, n_mels=n_mels,fmax=fmax, htk=htk)
        if len(S) < len(mel_basis[0]):
            mel_basis = mel_basis[:, numpy.array(range(len(S)))]
        mel_spectrum = numpy.dot(mel_basis, S)

        # print("len: {}, dim, {}".format(len(mel_spectrum[0]), len(mel_spectrum)))
        return mel_spectrum.T[0]

    @classmethod
    def get_audio_feature(cls, file_name="", frame_win=0, frame_step=0, sample_rate=0):
        wav, sr = librosa.load(file_name, sr=sample_rate, mono=True)

        DC = wav.mean()
        MAX = (numpy.abs(wav)).max()
        wav = (wav - DC) / (MAX + 0.0000000001)

        # Split wav into frames
        frame_num = (len(wav)-frame_win) // frame_step + 1

        #extract wav feature
        feat_len = librosa_audio_feature.feat_len
        fft_win = librosa_audio_feature.fft_win
        hop_len = librosa_audio_feature.frame_step
        include_delta = librosa_audio_feature.include_delta
        include_acceleration = librosa_audio_feature.include_acceleration
        include_mfcc0 = librosa_audio_feature.include_mfcc0
        mfcc_feature = librosa_audio_feature.__feature_type_ == 1 # 1 for mfcc, 2 mel filterbank

        audio_feat = []
        assert librosa_audio_feature.fft_win >= frame_win
        frames = numpy.zeros(librosa_audio_feature.fft_win, dtype=float)
        for frm_idx in range(frame_num):
            frames[0:frame_win] = wav[frm_idx*frame_step:frm_idx*frame_step+frame_win]
            if mfcc_feature:
                feat = librosa_audio_feature.get_mfcc_matrix(frames, sr=sr, feat_len=feat_len,fft_win=fft_win,hop_len=hop_len,include_delta=include_delta, include_acceleration=include_acceleration, include_mfcc0=include_mfcc0)
            else:
                feat = librosa_audio_feature.get_melfilter_bank_matrix(frames, sr=sr, feat_len=feat_len, fft_win=fft_win, hop_len=hop_len)
            audio_feat.append(feat)
        #concat rate audio feature
        #audio_feat = numpy.concatenate(audio_feat)
        audio_feat = numpy.array(audio_feat)

        if mfcc_feature:
            audio_feat = audio_feat.T
            # Collect the feature matrix
            feature_matrix = audio_feat
            if include_delta:
                # Delta coefficients
                mfcc_delta = librosa.feature.delta(audio_feat)
                # Add Delta Coefficients to feature matrix
                feature_matrix = numpy.vstack((feature_matrix, mfcc_delta))

            if include_acceleration:
                # Acceleration coefficients (aka delta)
                mfcc_delta2 = librosa.feature.delta(audio_feat, order=2)
                # Add Acceleration Coefficients to feature matrix
                feature_matrix = numpy.vstack((feature_matrix, mfcc_delta2))

            if not include_mfcc0:
                # Omit mfcc0
                feature_matrix = feature_matrix[1:, :]

            audio_feat = feature_matrix.T

        return audio_feat, wav, sr

    @classmethod
    def get_frame_step(cls): #s
        return librosa_audio_feature.frame_step*1.0/librosa_audio_feature.sample_rate

    @classmethod
    def real_feat_len(cls):
        librosa_audio_feature.__real_feat_len = librosa_audio_feature.feat_len

        if librosa_audio_feature.__feature_type_ == 1: # MFCC
            if librosa_audio_feature.include_delta:
                librosa_audio_feature.__real_feat_len += librosa_audio_feature.feat_len

            if librosa_audio_feature.include_acceleration:
                librosa_audio_feature.__real_feat_len += librosa_audio_feature.feat_len

            if not librosa_audio_feature.include_mfcc0:
                # Omit mfcc0
                librosa_audio_feature.__real_feat_len -= 1

        return librosa_audio_feature.__real_feat_len

    @classmethod
    def normalizeFeatures(cls, features):
        MEAN = numpy.mean(features, axis=0) + 0.00000000000001;
        STD = numpy.std(features, axis=0) + 0.00000000000001;

        featuresNorm = numpy.zeros((len(features), len(features[0])), dtype=float)
        for idx, f in enumerate(features):
            featuresNorm[idx] = (f - MEAN) / STD
        return (featuresNorm, MEAN, STD)

def stZCR(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
    return (numpy.float64(countZ) / numpy.float64(count-1.0))

def stNormEnergy(frame):
    return numpy.sqrt(numpy.sum(frame ** 2)) / numpy.float64(len(frame))

def stEnergy(frame):
    """Computes signal energy of frame"""
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))

def stF0Raw(fs, frame):
    chunk = len(frame)
    # Take the fft and square each value
    fftData = abs(numpy.fft.rfft(frame)) ** 2
    # find the maximum
    which = fftData[1:].argmax() + 1
    # use quadratic interpolation around the max
    if which != len(fftData) - 1:
        y0, y1, y2 = numpy.log(fftData[which - 1:which + 2:])
        x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
        # find the frequency and output it
        thefreq = (which + x1) * fs / chunk
    else:
        thefreq = which * fs / chunk
    return thefreq

def stIntensity(frame):
    return numpy.sum((frame * numpy.hamming(len(frame))) ** 2) / numpy.float64(len(frame))

def stEnergyEntropy(frame, numOfShortBlocks=10):
    """Computes entropy of energy"""
    Eol = numpy.sum(frame ** 2)    # total frame energy
    L = len(frame)
    subWinLength = int(numpy.floor(L / numOfShortBlocks))
    if L != subWinLength * numOfShortBlocks:
            frame = frame[0:subWinLength * numOfShortBlocks]
    # subWindows is of size [numOfShortBlocks x L]
    subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)

    # Compute entropy of the normalized sub-frame energies:
    Entropy = -numpy.sum(s * numpy.log2(s + eps))
    return Entropy

""" pitch """
FRAME_NUM = 100
def stPitch(fnFullPath, minPitch, maxPitch):
    '''
    Former default pitch values: male (50, 350); female (75, 450)
    '''
    soundObj = tkSnack.Sound(load=fnFullPath)
    output = soundObj.pitch(method="ESPS",
                            minpitch=minPitch,
                            maxpitch=maxPitch)

    pitchList = numpy.zeros(len(output), dtype=float)
    for idx, value in enumerate(output):
        value = value[0]
        if value == 0:
            #value = int(value)
            continue

        pitchList[idx] = value

    return pitchList, FRAME_NUM

def stPitchAtTime(pitchList, startTime, endTime):
    startIndex = int(startTime * FRAME_NUM)
    endIndex = int(endTime * FRAME_NUM)

    return pitchList[startIndex:endIndex]


""" Frequency-domain audio features """

def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (numpy.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = numpy.sum(ind * Xt)
    DEN = numpy.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = numpy.sqrt(numpy.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)


def stSpectralEntropy(X, numOfShortBlocks=10):
    """Computes the spectral entropy"""
    L = len(X)                         # number of frame samples
    Eol = numpy.sum(X ** 2)            # total spectral energy

    subWinLength = int(numpy.floor(L / numOfShortBlocks))   # length of sub-frame
    if L != subWinLength * numOfShortBlocks:
        X = X[0:subWinLength * numOfShortBlocks]

    subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)                      # compute spectral sub-energies
    En = -numpy.sum(s*numpy.log2(s + eps))                                    # compute spectral entropy

    return En


def stSpectralFlux(X, Xprev):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        X:        the abs(fft) of the current frame
        Xpre:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    sumX = numpy.sum(X + eps)
    sumPrevX = numpy.sum(Xprev + eps)
    F = numpy.sum((X / sumX - Xprev/sumPrevX) ** 2)

    return F


def stSpectralRollOff(X, c, fs):
    """Computes spectral roll-off"""
    totalEnergy = numpy.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Ffind the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    CumSum = numpy.cumsum(X ** 2) + eps
    [a, ] = numpy.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = numpy.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)


def stHarmonic(frame, fs):
    """
    Computes harmonic ratio and pitch
    """
    M = numpy.round(0.016 * fs) - 1
    R = numpy.correlate(frame, frame, mode='full')
    M = int(M)

    g = R[len(frame)-1]
    R = R[len(frame):-1]

    # estimate m0 (as the first zero crossing of R)
    [a, ] = numpy.nonzero(numpy.diff(numpy.sign(R)))

    if len(a) == 0:
        m0 = len(R)-1
    else:
        m0 = a[0]
    if M > len(R):
        M = len(R) - 1

    Gamma = numpy.zeros((M), dtype=numpy.float64)
    CSum = numpy.cumsum(frame ** 2)
    Gamma[m0:M] = R[m0:M] / (numpy.sqrt((g * CSum[M:m0:-1])) + eps)

    ZCR = stZCR(Gamma)

    if ZCR > 0.15:
        HR = 0.0
        f0 = 0.0
    else:
        if len(Gamma) == 0:
            HR = 1.0
            blag = 0.0
            Gamma = numpy.zeros((M), dtype=numpy.float64)
        else:
            HR = numpy.max(Gamma)
            blag = numpy.argmax(Gamma)

        # Get fundamental frequency:
        f0 = fs / (blag + eps)
        if f0 > 5000:
            f0 = 0.0
        if HR < 0.1:
            f0 = 0.0

    return (HR, f0)


def mfccInitFilterBanks(fs, nfft):
    """
    Computes the triangular filterbank for MFCC computation (used in the stFeatureExtraction function before the stMFCC function call)
    This function is taken from the scikits.talkbox library (MIT Licence):
    https://pypi.python.org/pypi/scikits.talkbox
    """

    # filter bank params:
    lowfreq = 133.33
    linsc = 200/3.
    logsc = 1.0711703
    numLinFiltTotal = 13
    numLogFilt = 27

    if fs < 8000:
        nlogfil = 5

    # Total number of filters
    nFiltTotal = numLinFiltTotal + numLogFilt

    # Compute frequency points of the triangle:
    freqs = numpy.zeros(nFiltTotal+2)
    freqs[:numLinFiltTotal] = lowfreq + numpy.arange(numLinFiltTotal) * linsc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** numpy.arange(1, numLogFilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nFiltTotal, int(nfft)))
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i+1]
        highTrFreq = freqs[i+2]

        lid = numpy.arange(numpy.floor(lowTrFreq * nfft / fs) + 1, numpy.floor(cenTrFreq * nfft / fs) + 1, dtype=numpy.int)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = numpy.arange(numpy.floor(cenTrFreq * nfft / fs) + 1, numpy.floor(highTrFreq * nfft / fs) + 1, dtype=numpy.int)
        rslope = heights[i] / (highTrFreq - cenTrFreq)
        fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
        fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

    return fbank, freqs


def stMFCC(X, fbank, nceps):
    """
    Computes the MFCCs of a frame, given the fft mag

    ARGUMENTS:
        X:        fft magnitude abs(FFT)
        fbank:    filter bank (see mfccInitFilterBanks)
    RETURN
        ceps:     MFCCs (13 element vector)

    Note:    MFCC calculation is, in general, taken from the scikits.talkbox library (MIT Licence),
    #    with a small number of modifications to make it more compact and suitable for the pyAudioAnalysis Lib
    """

    mspec = numpy.log10(numpy.dot(X, fbank.T)+eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
    return ceps

def stMFCCAndMelSpectro(X, fbank, nceps):
    """
    Computes the MFCCs of a frame, given the fft mag

    ARGUMENTS:
        X:        fft magnitude abs(FFT)
        fbank:    filter bank (see mfccInitFilterBanks)
    RETURN
        ceps:     MFCCs (13 element vector)

    Note:    MFCC calculation is, in general, taken from the scikits.talkbox library (MIT Licence),
    #    with a small number of modifications to make it more compact and suitable for the pyAudioAnalysis Lib
    """

    mspec = numpy.log10(numpy.dot(X, fbank.T)+eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
    return ceps, mspec

def stChromaFeaturesInit(nfft, fs):
    """
    This function initializes the chroma matrices used in the calculation of the chroma features
    """
    freqs = numpy.array([((f + 1) * fs) / (2 * nfft) for f in range(nfft)])    
    Cp = 27.50    
    nChroma = numpy.round(12.0 * numpy.log2(freqs / Cp)).astype(int)

    nFreqsPerChroma = numpy.zeros((nChroma.shape[0], ))

    uChroma = numpy.unique(nChroma)
    for u in uChroma:
        idx = numpy.nonzero(nChroma == u)
        nFreqsPerChroma[idx] = idx[0].shape
    
    return nChroma, nFreqsPerChroma


def stChromaFeatures(X, fs, nChroma, nFreqsPerChroma):
    #TODO: 1 complexity
    #TODO: 2 bug with large windows

    chromaNames = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    spec = X**2    
    if nChroma.max()<nChroma.shape[0]:        
        C = numpy.zeros((nChroma.shape[0],))
        C[nChroma] = spec
        C /= nFreqsPerChroma[nChroma]
    else:        
        I = numpy.nonzero(nChroma>nChroma.shape[0])[0][0]        
        C = numpy.zeros((nChroma.shape[0],))
        C[nChroma[0:I-1]] = spec            
        C /= nFreqsPerChroma
    finalC = numpy.zeros((12, 1))
    newD = int(numpy.ceil(C.shape[0] / 12.0) * 12)
    C2 = numpy.zeros((newD, ))
    C2[0:C.shape[0]] = C
    C2 = C2.reshape(int(C2.shape[0]/12), 12)
    #for i in range(12):
    #    finalC[i] = numpy.sum(C[i:C.shape[0]:12])
    finalC = numpy.matrix(numpy.sum(C2, axis=0)).T
    finalC /= spec.sum()

#    ax = plt.gca()
#    plt.hold(False)
#    plt.plot(finalC)
#    ax.set_xticks(range(len(chromaNames)))
#    ax.set_xticklabels(chromaNames)
#    xaxis = numpy.arange(0, 0.02, 0.01);
#    ax.set_yticks(range(len(xaxis)))
#    ax.set_yticklabels(xaxis)
#    plt.show(block=False)
#    plt.draw()

    return chromaNames, finalC


def stChromagram(signal, Fs, Win, Step, PLOT=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a numpy array (nFFT x numOfShortTermWindows)
    ARGUMENTS:
        signal:      the input signal samples
        Fs:          the sampling freq (in Hz)
        Win:         the short-term window size (in samples)
        Step:        the short-term window step (in samples)
        PLOT:        flag, 1 if results are to be ploted
    RETURNS:
    """
    Win = int(Win)
    Step = int(Step)
    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX - DC)

    N = len(signal)        # total number of signals
    curPos = 0
    countFrames = 0
    nfft = int(Win / 2)
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nfft, Fs)
    chromaGram = numpy.array([], dtype=numpy.float64)

    while (curPos + Win - 1 < N):
        countFrames += 1
        x = signal[curPos:curPos + Win]
        curPos = curPos + Step
        X = abs(fft(x))
        X = X[0:nfft]
        X = X / len(X)
        chromaNames, C = stChromaFeatures(X, Fs, nChroma, nFreqsPerChroma)
        C = C[:, 0]
        if countFrames == 1:
            chromaGram = C.T
        else:
            chromaGram = numpy.vstack((chromaGram, C.T))
    FreqAxis = chromaNames
    TimeAxis = [(t * Step) / Fs for t in range(chromaGram.shape[0])]

    if (PLOT):
        fig, ax = plt.subplots()
        chromaGramToPlot = chromaGram.transpose()[::-1, :]
        Ratio = chromaGramToPlot.shape[1] / (3*chromaGramToPlot.shape[0])        
        if Ratio < 1:
            Ratio = 1
        chromaGramToPlot = numpy.repeat(chromaGramToPlot, Ratio, axis=0)
        imgplot = plt.imshow(chromaGramToPlot)
        Fstep = int(nfft / 5.0)
#        FreqTicks = range(0, int(nfft) + Fstep, Fstep)
#        FreqTicksLabels = [str(Fs/2-int((f*Fs) / (2*nfft))) for f in FreqTicks]
        ax.set_yticks(range(Ratio / 2, len(FreqAxis) * Ratio, Ratio))
        ax.set_yticklabels(FreqAxis[::-1])
        TStep = countFrames / 3
        TimeTicks = range(0, countFrames, TStep)
        TimeTicksLabels = ['%.2f' % (float(t * Step) / Fs) for t in TimeTicks]
        ax.set_xticks(TimeTicks)
        ax.set_xticklabels(TimeTicksLabels)
        ax.set_xlabel('time (secs)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()

    return (chromaGram, TimeAxis, FreqAxis)


def phormants(x, Fs):
    N = len(x)
    w = numpy.hamming(N)

    # Apply window and high pass filter.
    x1 = x * w   
    x1 = lfilter([1], [1., 0.63], x1)
    
    # Get LPC.    
    ncoeff = 2 + Fs / 1000
    A, e, k = lpc(x1, ncoeff)    
    #A, e, k = lpc(x1, 8)

    # Get roots.
    rts = numpy.roots(A)
    rts = [r for r in rts if numpy.imag(r) >= 0]

    # Get angles.
    angz = numpy.arctan2(numpy.imag(rts), numpy.real(rts))

    # Get frequencies.    
    frqs = sorted(angz * (Fs / (2 * math.pi)))

    return frqs
def beatExtraction(stFeatures, winSize, PLOT=False):
    """
    This function extracts an estimate of the beat rate for a musical signal.
    ARGUMENTS:
     - stFeatures:     a numpy array (numOfFeatures x numOfShortTermWindows)
     - winSize:        window size in seconds
    RETURNS:
     - BPM:            estimates of beats per minute
     - Ratio:          a confidence measure
    """

    # Features that are related to the beat tracking task:
    toWatch = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    maxBeatTime = int(round(2.0 / winSize))
    HistAll = numpy.zeros((maxBeatTime,))
    for ii, i in enumerate(toWatch):                                        # for each feature
        DifThres = 2.0 * (numpy.abs(stFeatures[i, 0:-1] - stFeatures[i, 1::])).mean()    # dif threshold (3 x Mean of Difs)
        if DifThres<=0:
            DifThres = 0.0000000000000001        
        [pos1, _] = utilities.peakdet(stFeatures[i, :], DifThres)           # detect local maxima
        posDifs = []                                                        # compute histograms of local maxima changes
        for j in range(len(pos1)-1):
            posDifs.append(pos1[j+1]-pos1[j])
        [HistTimes, HistEdges] = numpy.histogram(posDifs, numpy.arange(0.5, maxBeatTime + 1.5))
        HistCenters = (HistEdges[0:-1] + HistEdges[1::]) / 2.0
        HistTimes = HistTimes.astype(float) / stFeatures.shape[1]
        HistAll += HistTimes
        if PLOT:
            plt.subplot(9, 2, ii + 1)
            plt.plot(stFeatures[i, :], 'k')
            for k in pos1:
                plt.plot(k, stFeatures[i, k], 'k*')
            f1 = plt.gca()
            f1.axes.get_xaxis().set_ticks([])
            f1.axes.get_yaxis().set_ticks([])

    if PLOT:
        plt.show(block=False)
        plt.figure()

    # Get beat as the argmax of the agregated histogram:
    I = numpy.argmax(HistAll)
    BPMs = 60 / (HistCenters * winSize)
    BPM = BPMs[I]
    # ... and the beat ratio:
    Ratio = HistAll[I] / HistAll.sum()

    if PLOT:
        # filter out >500 beats from plotting:
        HistAll = HistAll[BPMs < 500]
        BPMs = BPMs[BPMs < 500]

        plt.plot(BPMs, HistAll, 'k')
        plt.xlabel('Beats per minute')
        plt.ylabel('Freq Count')
        plt.show(block=True)

    return BPM, Ratio


def stSpectogram(signal, Fs, Win, Step, PLOT=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a numpy array (nFFT x numOfShortTermWindows)
    ARGUMENTS:
        signal:      the input signal samples
        Fs:          the sampling freq (in Hz)
        Win:         the short-term window size (in samples)
        Step:        the short-term window step (in samples)
        PLOT:        flag, 1 if results are to be ploted
    RETURNS:
    """
    Win = int(Win)
    Step = int(Step)
    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX - DC)

    N = len(signal)        # total number of signals
    curPos = 0
    countFrames = 0
    nfft = int(Win / 2)
    specgram = numpy.array([], dtype=numpy.float64)

    while (curPos + Win - 1 < N):
        countFrames += 1
        x = signal[curPos:curPos+Win]
        curPos = curPos + Step
        X = abs(fft(x))
        X = X[0:nfft]
        X = X / len(X)

        if countFrames == 1:
            specgram = X ** 2
        else:
            specgram = numpy.vstack((specgram, X))

    FreqAxis = [((f + 1) * Fs) / (2 * nfft) for f in range(specgram.shape[1])]
    TimeAxis = [(t * Step) / Fs for t in range(specgram.shape[0])]

    if (PLOT):
        fig, ax = plt.subplots()
        imgplot = plt.imshow(specgram.transpose()[::-1, :])
        Fstep = int(nfft / 5.0)
        FreqTicks = range(0, int(nfft) + Fstep, Fstep)
        FreqTicksLabels = [str(Fs / 2 - int((f * Fs) / (2 * nfft))) for f in FreqTicks]
        ax.set_yticks(FreqTicks)
        ax.set_yticklabels(FreqTicksLabels)
        TStep = countFrames/3
        TimeTicks = range(0, countFrames, TStep)
        TimeTicksLabels = ['%.2f' % (float(t * Step) / Fs) for t in TimeTicks]
        ax.set_xticks(TimeTicks)
        ax.set_xticklabels(TimeTicksLabels)
        ax.set_xlabel('time (secs)')
        ax.set_ylabel('freq (Hz)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()

    return (specgram, TimeAxis, FreqAxis)


""" Windowing and feature extraction """


def feature_extraction(y=None, fs=None, statistics=True, include_mfcc0=True, include_delta=True, include_acceleration=True, mfcc_params=None, delta_params=None, acceleration_params=None):
    # Extract features, Mel Frequency Cepstral Coefficients
    eps = numpy.spacing(1)

    # Windowing function
    if mfcc_params['window'] == 'hamming_asymmetric':
        window = hamming(mfcc_params['n_fft'], sym=False)
    elif mfcc_params['window'] == 'hamming_symmetric':
        window = hamming(mfcc_params['n_fft'], sym=True)
    #elif mfcc_params['window'] == 'hann_asymmetric':
    #    window = hann(mfcc_params['n_fft'], sym=False)
    #elif mfcc_params['window'] == 'hann_symmetric':
    #    window = hann(mfcc_params['n_fft'], sym=True)
    else:
        window = None

    # Calculate Static Coefficients
    magnitude_spectrogram = numpy.abs(librosa.stft(y + eps, n_fft=mfcc_params['n_fft'], win_length=mfcc_params['win_length'], hop_length=mfcc_params['hop_length'], window=window))**2
    mel_basis = librosa.filters.mel(sr=fs, n_fft=mfcc_params['n_fft'], n_mels=mfcc_params['n_mels'], fmin=mfcc_params['fmin'], fmax=mfcc_params['fmax'], htk=mfcc_params['htk'])
    mel_spectrum = numpy.dot(mel_basis, magnitude_spectrogram)
    mfcc = librosa.feature.mfcc(S=librosa.logamplitude(mel_spectrum))

    # Collect the feature matrix
    feature_matrix = mfcc
    if include_delta:
        # Delta coefficients
        mfcc_delta = librosa.feature.delta(mfcc, **delta_params)

        # Add Delta Coefficients to feature matrix
        feature_matrix = numpy.vstack((feature_matrix, mfcc_delta))

    if include_acceleration:
        # Acceleration coefficients (aka delta)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2, **acceleration_params)

        # Add Acceleration Coefficients to feature matrix
        feature_matrix = numpy.vstack((feature_matrix, mfcc_delta2))


    if not include_mfcc0:
        # Omit mfcc0
        feature_matrix = feature_matrix[1:, :]

    feature_matrix = feature_matrix.T

    # Collect into data structure
    if statistics:
        return {
            'feat': feature_matrix,
            'stat': {
                'mean': numpy.mean(feature_matrix, axis=0),
                'std': numpy.std(feature_matrix, axis=0),
                'N': feature_matrix.shape[0],
                'S1': numpy.sum(feature_matrix, axis=0),
                'S2': numpy.sum(feature_matrix ** 2, axis=0),
            }
        }
    else:
        return {
            'feat': feature_matrix}


def stFeatureExtractionForSpeechDetect(signal, Fs, Win, Step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
        Win:          the short-term window size (in samples)
        Step:         the short-term window step (in samples)
    RETURNS
        stFeatures:   a numpy array (numOfFeatures x numOfShortTermWindows)
    """

    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = numpy.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)

    N = len(signal)  # total number of samples
    curPos = 0
    countFrames = 0
    nFFT = Win / 2

    [fbank, freqs] = mfccInitFilterBanks(Fs, nFFT)  # compute the triangular filter banks used in the mfcc calculation
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nFFT, Fs)

    numOfTimeSpectralFeatures = 1
    numOfHarmonicFeatures = 0
    nceps = 13
    numOfChromaFeatures = 0
    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps - 1 + numOfHarmonicFeatures + numOfChromaFeatures
    #    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures

    stFeatures = []
    while (curPos + Win - 1 < N):  # for each short-term window until the end of signal
        countFrames += 1
        x = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position
        X = abs(fft(x))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        #(HR, f0) = stHarmonic(x, Fs)
        #if f0 > 750 or f0 < 50:
        #    f0 = 0

        if countFrames == 1:
            Xprev = X.copy()                             # keep previous fft mag (used in spectral flux)

        curFV = numpy.zeros(totalNumOfFeatures)
        #curFV[0] = stZCR(x)                              # zero crossing rate
        curFV[0] = stEnergy(x)               # short-term energy
        #curFV[1] = f0
        #curFV[2] = stSpectralFlux(X, Xprev)              # spectral flux
        curFV[numOfTimeSpectralFeatures:numOfTimeSpectralFeatures+nceps-1] = stMFCC(X, fbank, nceps).copy()[1:]    # MFCCs

        # end of delta
        Xprev = X.copy()

        stFeatures.append(curFV)

    return stFeatures

def stEnergyExtraction(signal, Fs, Win, Step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
        Win:          the short-term window size (in samples)
        Step:         the short-term window step (in samples)
    RETURNS
        stFeatures:   a numpy array (numOfFeatures x numOfShortTermWindows)
    """

    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = numpy.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)

    N = len(signal)  # total number of samples
    curPos = 0

    stFeatures = []
    while (curPos + Win - 1 < N):  # for each short-term window until the end of signal
        x = signal[curPos:curPos + Win]  # get current window
        curPos = curPos + Step  # update window position
        stFeatures.append(stEnergy(x))

    stFeatures = numpy.array(stFeatures)
    return stFeatures

def loadlibrosaAudio(fileName="", sample_rate=8000):
    return librosa.load(fileName, sr=sample_rate)

def _spectrogram(y=None, S=None, n_fft=2048, hop_length=512, power=1):
    if S is not None:
        # Infer n_fft from spectrogram shape
        n_fft = 2 * (S.shape[0] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        S = numpy.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**power

    return S, n_fft

def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                   power=2.0, **kwargs):

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                            power=power)

    # Build a Mel filter
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=40, **kwargs)
    return numpy.dot(mel_basis, S)

def melSpectrumFromBuffer(fileBuffer=[], sr=8000, nfft=256):
    n_fft = nfft  # 2048
    hop_len = n_fft // 2 #4
    S = librosa.power_to_db(melspectrogram(y=fileBuffer, sr=sr, n_fft=n_fft, hop_length=hop_len))
    return S

def makeFrameFeatureFromSpectrogram(spectrogram=[], frame_size=15):
    feat_len = len(spectrogram) - frame_size + 1
    feat_size = frame_size*len(spectrogram[0])
    feat = numpy.zeros((feat_len, feat_size), dtype=float)

    for idx in range(feat_len):
        feat[idx, :] = numpy.reshape(spectrogram[idx:idx+frame_size], feat_size)
    return feat


def stMelFeatureExtraction(signal, Fs, Win, Step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
        Win:          the short-term window size (in samples)
        Step:         the short-term window step (in samples)
    RETURNS
        stFeatures:   a numpy array (numOfFeatures x numOfShortTermWindows)
    """

    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = numpy.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)

    N = len(signal)                                # total number of samples
    curPos = 0
    countFrames = 0
    nFFT = Win / 2
    #nFFT = Win

    [fbank, freqs] = mfccInitFilterBanks(Fs, nFFT)                # compute the triangular filter banks used in the mfcc calculation

    numOfMelFilterBankFeatures = 40
    nceps = 13
    totalNumOfFeatures = 1+numOfMelFilterBankFeatures + nceps
#    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures

    stFeatures = []
    while (curPos + Win - 1 < N):                        # for each short-term window until the end of signal
        countFrames += 1
        x = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position
        X = abs(fft(x))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        curFV = numpy.zeros((totalNumOfFeatures, 1))
        mfcc, mspec = stMFCCAndMelSpectro(X, fbank, nceps)
        curFV[0] = stEnergy(x)                           # short-term energy
        featCnt = 1
        curFV[featCnt:featCnt+numOfMelFilterBankFeatures, 0] = mspec.copy()    # Mel filter bank energy
        featCnt = featCnt + numOfMelFilterBankFeatures
        curFV[featCnt:featCnt+nceps, 0] = mfcc.copy()    # Mel filter bank energy
        stFeatures.append(curFV)

    stFeatures = numpy.concatenate(stFeatures, 1)
    return stFeatures


def stFeatureExtraction(signal, Fs, Win, Step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
        Win:          the short-term window size (in samples)
        Step:         the short-term window step (in samples)
    RETURNS
        stFeatures:   a numpy array (numOfFeatures x numOfShortTermWindows)
    """

    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = numpy.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)

    #print("Max: {}, Mean: {}, STD: {}".format(MAX, DC, signal.std()))

    N = len(signal)                                # total number of samples
    curPos = 0
    countFrames = 0
    nFFT = int(Win / 2)

    [fbank, freqs] = mfccInitFilterBanks(Fs, nFFT)                # compute the triangular filter banks used in the mfcc calculation
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nFFT, Fs)

    numOfTimeSpectralFeatures = 8
    numOfHarmonicFeatures = 0
    nceps = 13
    numOfChromaFeatures = 13
    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures + numOfChromaFeatures
#    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures

    stFeatures = []
    while (curPos + Win - 1 < N):                        # for each short-term window until the end of signal
        countFrames += 1
        x = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position
        X = abs(fft(x))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        if countFrames == 1:
            Xprev = X.copy()                             # keep previous fft mag (used in spectral flux)
        curFV = numpy.zeros((totalNumOfFeatures, 1))
        curFV[0] = stZCR(x)                              # zero crossing rate
        curFV[1] = stEnergy(x)                           # short-term energy
        curFV[2] = stEnergyEntropy(x)                    # short-term entropy of energy
        [curFV[3], curFV[4]] = stSpectralCentroidAndSpread(X, Fs)    # spectral centroid and spread
        curFV[5] = stSpectralEntropy(X)                  # spectral entropy
        #curFV[6] = stSpectralFlux(X, Xprev)              # spectral flux
        curFV[6] = stIntensity(x)  # intensity
        #curFV[7] = stSpectralRollOff(X, 0.90, Fs)        # spectral rolloff
        curFV[7] = stF0Raw(Fs, x)  # pitch: f0-raw
        curFV[numOfTimeSpectralFeatures:numOfTimeSpectralFeatures+nceps, 0] = stMFCC(X, fbank, nceps).copy()    # MFCCs

        chromaNames, chromaF = stChromaFeatures(X, Fs, nChroma, nFreqsPerChroma)
        curFV[numOfTimeSpectralFeatures + nceps: numOfTimeSpectralFeatures + nceps + numOfChromaFeatures - 1] = chromaF
        curFV[numOfTimeSpectralFeatures + nceps + numOfChromaFeatures - 1] = chromaF.std()

        curFV[8] = curFV[7]
        stFeatures.append(curFV)
        # delta features
        '''
        if countFrames>1:
            delta = curFV - prevFV
            curFVFinal = numpy.concatenate((curFV, delta))            
        else:
            curFVFinal = numpy.concatenate((curFV, curFV))
        prevFV = curFV
        stFeatures.append(curFVFinal)        
        '''
        # end of delta
        Xprev = X.copy()

    stFeatures = numpy.concatenate(stFeatures, 1)
    return stFeatures


def mtFeatureExtraction(signal, Fs, mtWin, mtStep, stWin, stStep):
    """
    Mid-term feature extraction
    """

    mtWinRatio = int(round(mtWin / stStep))
    mtStepRatio = int(round(mtStep / stStep))

    mtFeatures = []

    stFeatures = stFeatureExtraction(signal, Fs, stWin, stStep)
    numOfFeatures = len(stFeatures)
    numOfStatistics = 2

    mtFeatures = []
    #for i in range(numOfStatistics * numOfFeatures + 1):
    for i in range(numOfStatistics * numOfFeatures):
        mtFeatures.append([])

    for i in range(numOfFeatures):        # for each of the short-term features:
        curPos = 0
        N = len(stFeatures[i])
        while (curPos < N):
            N1 = curPos
            N2 = curPos + mtWinRatio
            if N2 > N:
                N2 = N
            curStFeatures = stFeatures[i][N1:N2]

            mtFeatures[i].append(numpy.mean(curStFeatures))
            mtFeatures[i+numOfFeatures].append(numpy.std(curStFeatures))
            #mtFeatures[i+2*numOfFeatures].append(numpy.std(curStFeatures) / (numpy.mean(curStFeatures)+0.00000010))
            curPos += mtStepRatio

    return numpy.array(mtFeatures), stFeatures


# TODO
def stFeatureSpeed(signal, Fs, Win, Step):

    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / MAX
    # print (numpy.abs(signal)).max()

    N = len(signal)        # total number of signals
    curPos = 0
    countFrames = 0

    lowfreq = 133.33
    linsc = 200/3.
    logsc = 1.0711703
    nlinfil = 13
    nlogfil = 27
    nceps = 13
    nfil = nlinfil + nlogfil
    nfft = Win / 2
    if Fs < 8000:
        nlogfil = 5
        nfil = nlinfil + nlogfil
        nfft = Win / 2

    # compute filter banks for mfcc:
    [fbank, freqs] = mfccInitFilterBanks(Fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)

    numOfTimeSpectralFeatures = 8
    numOfHarmonicFeatures = 1
    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures
    #stFeatures = numpy.array([], dtype=numpy.float64)
    stFeatures = []

    while (curPos + Win - 1 < N):
        countFrames += 1
        x = signal[curPos:curPos + Win]
        curPos = curPos + Step
        X = abs(fft(x))
        X = X[0:nfft]
        X = X / len(X)
        Ex = 0.0
        El = 0.0
        X[0:4] = 0
#        M = numpy.round(0.016 * fs) - 1
#        R = numpy.correlate(frame, frame, mode='full')
        stFeatures.append(stHarmonic(x, Fs))
#        for i in range(len(X)):
            #if (i < (len(X) / 8)) and (i > (len(X)/40)):
            #    Ex += X[i]*X[i]
            #El += X[i]*X[i]
#        stFeatures.append(Ex / El)
#        stFeatures.append(numpy.argmax(X))
#        if curFV[numOfTimeSpectralFeatures+nceps+1]>0:
#            print curFV[numOfTimeSpectralFeatures+nceps], curFV[numOfTimeSpectralFeatures+nceps+1]
    return numpy.array(stFeatures)


""" Feature Extraction Wrappers

 - The first two feature extraction wrappers are used to extract long-term averaged
   audio features for a list of WAV files stored in a given category.
   It is important to note that, one single feature is extracted per WAV file (not the whole sequence of feature vectors)

 """


def dirWavFeatureExtraction(dirName, mtWin, mtStep, stWin, stStep, computeBEAT=False):
    """
    This function extracts the mid-term features of the WAVE files of a particular folder.

    The resulting feature vector is extracted by long-term averaging the mid-term features.
    Therefore ONE FEATURE VECTOR is extracted for each WAV file.

    ARGUMENTS:
        - dirName:        the path of the WAVE directory
        - mtWin, mtStep:    mid-term window and step (in seconds)
        - stWin, stStep:    short-term window and step (in seconds)
    """

    allMtFeatures = numpy.array([])
    processingTimes = []

    types = ('*.wav', '*.aif',  '*.aiff', '*.mp3','*.au')
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob(os.path.join(dirName, files)))

    wavFilesList = sorted(wavFilesList)    
    wavFilesList2 = []
    for i, wavFile in enumerate(wavFilesList):        
        print("Analyzing file {0:d} of {1:d}: {2:s}".format(i+1, len(wavFilesList), wavFile.encode('utf-8')))
        if os.stat(wavFile).st_size == 0:
            print("   (EMPTY FILE -- SKIPPING)")
            continue        
        [Fs, x] = audioBasicIO.readAudioFile(wavFile)            # read file    
        if isinstance(x, int):
            continue        

        t1 = time.clock()        
        x = audioBasicIO.stereo2mono(x)                          # convert stereo to mono                
        if x.shape[0]<float(Fs)/10:
            print("  (AUDIO FILE TOO SMALL - SKIPPING)")
            continue
        wavFilesList2.append(wavFile)
        if computeBEAT:                                          # mid-term feature extraction for current file
            [MidTermFeatures, stFeatures] = mtFeatureExtraction(x, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))
            [beat, beatConf] = beatExtraction(stFeatures, stStep)
        else:
            [MidTermFeatures, _] = mtFeatureExtraction(x, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))

        MidTermFeatures = numpy.transpose(MidTermFeatures)
        MidTermFeatures = MidTermFeatures.mean(axis=0)         # long term averaging of mid-term statistics
        if (not numpy.isnan(MidTermFeatures).any()) and (not numpy.isinf(MidTermFeatures).any()):            
            if computeBEAT:
                MidTermFeatures = numpy.append(MidTermFeatures, beat)
                MidTermFeatures = numpy.append(MidTermFeatures, beatConf)
            if len(allMtFeatures) == 0:                              # append feature vector
                allMtFeatures = MidTermFeatures
            else:
                allMtFeatures = numpy.vstack((allMtFeatures, MidTermFeatures))
            t2 = time.clock()
            duration = float(len(x)) / Fs
            processingTimes.append((t2 - t1) / duration)
    if len(processingTimes) > 0:
        print("Feature extraction complexity ratio: {0:.1f} x realtime".format((1.0 / numpy.mean(numpy.array(processingTimes)))))
    return (allMtFeatures, wavFilesList2)


def dirsWavFeatureExtraction(dirNames, mtWin, mtStep, stWin, stStep, computeBEAT=False):
    '''
    Same as dirWavFeatureExtraction, but instead of a single dir it takes a list of paths as input and returns a list of feature matrices.
    EXAMPLE:
    [features, classNames] =
           a.dirsWavFeatureExtraction(['audioData/classSegmentsRec/noise','audioData/classSegmentsRec/speech',
                                       'audioData/classSegmentsRec/brush-teeth','audioData/classSegmentsRec/shower'], 1, 1, 0.02, 0.02);

    It can be used during the training process of a classification model ,
    in order to get feature matrices from various audio classes (each stored in a seperate path)
    '''

    # feature extraction for each class:
    features = []
    classNames = []
    fileNames = []
    for i, d in enumerate(dirNames):
        [f, fn] = dirWavFeatureExtraction(d, mtWin, mtStep, stWin, stStep, computeBEAT=computeBEAT)
        if f.shape[0] > 0:       # if at least one audio file has been found in the provided folder:
            features.append(f)
            fileNames.append(fn)
            if d[-1] == "/":
                classNames.append(d.split(os.sep)[-2])
            else:
                classNames.append(d.split(os.sep)[-1])
    return features, classNames, fileNames


def dirWavFeatureExtractionNoAveraging(dirName, mtWin, mtStep, stWin, stStep):
    """
    This function extracts the mid-term features of the WAVE files of a particular folder without averaging each file.

    ARGUMENTS:
        - dirName:          the path of the WAVE directory
        - mtWin, mtStep:    mid-term window and step (in seconds)
        - stWin, stStep:    short-term window and step (in seconds)
    RETURNS:
        - X:                A feature matrix
        - Y:                A matrix of file labels
        - filenames:
    """

    allMtFeatures = numpy.array([])
    signalIndices = numpy.array([])
    processingTimes = []

    types = ('*.wav', '*.aif',  '*.aiff')
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob(os.path.join(dirName, files)))

    wavFilesList = sorted(wavFilesList)

    for i, wavFile in enumerate(wavFilesList):
        [Fs, x] = audioBasicIO.readAudioFile(wavFile)            # read file
        if isinstance(x, int):
            continue        
        
        x = audioBasicIO.stereo2mono(x)                          # convert stereo to mono
        [MidTermFeatures, _] = mtFeatureExtraction(x, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))  # mid-term feature

        MidTermFeatures = numpy.transpose(MidTermFeatures)
#        MidTermFeatures = MidTermFeatures.mean(axis=0)        # long term averaging of mid-term statistics
        if len(allMtFeatures) == 0:                # append feature vector
            allMtFeatures = MidTermFeatures
            signalIndices = numpy.zeros((MidTermFeatures.shape[0], ))
        else:
            allMtFeatures = numpy.vstack((allMtFeatures, MidTermFeatures))
            signalIndices = numpy.append(signalIndices, i * numpy.ones((MidTermFeatures.shape[0], )))

    return (allMtFeatures, signalIndices, wavFilesList)


# The following two feature extraction wrappers extract features for given audio files, however
# NO LONG-TERM AVERAGING is performed. Therefore, the output for each audio file is NOT A SINGLE FEATURE VECTOR
# but a whole feature matrix.
#
# Also, another difference between the following two wrappers and the previous is that they NO LONG-TERM AVERAGING IS PERFORMED.
# In other words, the WAV files in these functions are not used as uniform samples that need to be averaged but as sequences

def mtFeatureExtractionToFile(fileName, midTermSize, midTermStep, shortTermSize, shortTermStep, outPutFile,
                              storeStFeatures=False, storeToCSV=False, PLOT=False):
    """
    This function is used as a wrapper to:
    a) read the content of a WAV file
    b) perform mid-term feature extraction on that signal
    c) write the mid-term feature sequences to a numpy file
    """
    [Fs, x] = audioBasicIO.readAudioFile(fileName)            # read the wav file
    x = audioBasicIO.stereo2mono(x)                           # convert to MONO if required
    if storeStFeatures:
        [mtF, stF] = mtFeatureExtraction(x, Fs, round(Fs * midTermSize), round(Fs * midTermStep), round(Fs * shortTermSize), round(Fs * shortTermStep))
    else:
        [mtF, _] = mtFeatureExtraction(x, Fs, round(Fs*midTermSize), round(Fs * midTermStep), round(Fs * shortTermSize), round(Fs * shortTermStep))

    numpy.save(outPutFile, mtF)                              # save mt features to numpy file
    if PLOT:
        print("Mid-term numpy file: " + outPutFile + ".npy saved")
    if storeToCSV:
        numpy.savetxt(outPutFile+".csv", mtF.T, delimiter=",")
        if PLOT:
            print("Mid-term CSV file: " + outPutFile + ".csv saved")

    if storeStFeatures:
        numpy.save(outPutFile+"_st", stF)                    # save st features to numpy file
        if PLOT:
            print("Short-term numpy file: " + outPutFile + "_st.npy saved")
        if storeToCSV:
            numpy.savetxt(outPutFile+"_st.csv", stF.T, delimiter=",")    # store st features to CSV file
            if PLOT:
                print("Short-term CSV file: " + outPutFile + "_st.csv saved")


def mtFeatureExtractionToFileDir(dirName, midTermSize, midTermStep, shortTermSize, shortTermStep, storeStFeatures=False, storeToCSV=False, PLOT=False):
    types = (dirName + os.sep + '*.wav', )
    filesToProcess = []
    for files in types:
        filesToProcess.extend(glob.glob(files))
    for f in filesToProcess:
        outPath = f
        mtFeatureExtractionToFile(f, midTermSize, midTermStep, shortTermSize, shortTermStep, outPath, storeStFeatures, storeToCSV, PLOT)
