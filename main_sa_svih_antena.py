# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:57:17 2021

@author: Marta
"""

import pandas as pd
import os
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage
from math import floor
from scipy import stats
from scipy import io

class Parameters:
    RxNum=4
    FSlope=40.8450012207031251e6*1e6
    Fs=3e6
    Bw=3.746303561822511e9
    ChirpsInFrame=2
    ADCSamp=256
    Fc=60.25e9
    C=3e8
    Lambda=C/(Bw/2+Fc)
    RR=C/(2*Bw)
    RMax=(Fs*C)/(2*FSlope)
    ChirpTime=Bw/FSlope
    FFTSize=2**10
    RRFFT=C*Fs/(2*FSlope)
    FPS=20
    Len=FPS*60*5

class Movement:
    def __init__(self,a,b):
        self.MFlag=b
        self.MTotal=a
        
class ParaExtreme:
    def __init__(self,refHR,refBR):
        self.maxhr=np.amax(refHR)/60
        self.minhr=np.amin(refHR)/60
        self.maxbr=np.amax(refBR)/60
        self.minbr=np.amin(refBR)/60
     
def ReadRadarData(dirpath, pnum):
    fpath=dirpath+'\\Rawdata_'+str(pnum)+'.csv'
    with open(fpath) as file:
        data1 = file.read().splitlines()
    file.close()
    for i in range(0,len(data1)):
        data1[i]=data1[i].replace('i','j').split(',') 
    return data1

def ReadGTData(fpath,pnum,GTtype):
    data=pd.read_csv(fpath+'\Ref_'+GTtype+'_'+str(pnum)+'.csv',header=None)
    return data

def ReadGTWaveData(fpath,pnum,GTtype):
    data=pd.read_csv(fpath+'\Ref_'+GTtype+'_Wave_'+str(pnum)+'.csv',header=None)
    return data

def GetFFT(data,para):
    X=np.empty((para.RxNum,para.Len,para.FFTSize),dtype=complex)
    for i in range(0,para.RxNum):
        for j in range(0,para.Len):
            X[i,j,:]=fft(data[i,j,:],para.FFTSize)
    return X

def DetectMovement(A,para):
    B=np.empty((para.RxNum,para.Len,para.FFTSize),dtype=complex)
    C=np.empty((para.RxNum,para.Len,int(para.FFTSize/2)),dtype=complex)
    alpha=0.1
    for i in range(0,para.RxNum):
        for j in range(1,para.Len):
            B[i,j,:]=alpha*A[i,j,:]+(1-alpha)*B[i,j-1,:]
            C[i,j,:]=A[i,j,:int(para.FFTSize/2)]-B[i,j,:int(para.FFTSize/2)]
    return C

def MeasureMovement(X):
    
    avg=np.sum(np.abs(X),axis=2)
    Y=np.abs(avg[:,1:]-avg[:,:-1])
    Z=np.append(np.zeros((4,1)),Y,axis=1)
    
    Total=np.sum(Z,axis=0)
    Total[0:99]=0
    Flag=Total>=850000
    return Movement(Total,Flag)

def GetDistance(A,para,antena):
    # d=np.empty((para.Len,para.RxNum),dtype=float)
    
    # for j in range(0,para.RxNum):
    #     for i in range(0,para.Len):
    #         ind=np.argmax(A[j,i,:int(para.FFTSize/2)])
    #         d[i,j]=ind*para.RRFFT/para.FFTSize
    
    # d=np.empty(para.Len,dtype=float)  
    # for i in range(0,para.Len):
    #     ind=np.argmax(A[antena-1,i,:int(para.FFTSize/2)])
    #     d[i]=ind*para.RRFFT/para.FFTSize
    # return d

    d=np.empty(para.Len,dtype=float)  
    for i in range(0,para.Len):
        ind=np.argmax(np.sum(np.abs(A[:,i,:int(para.FFTSize/2)]),axis=0))
        d[i]=ind*para.RRFFT/para.FFTSize
    return d

def GetDistanceNew(A,para,antena,win):
    d=np.zeros((4,para.Len),dtype=float)
    ind=np.zeros((4,para.Len),dtype=int)
    for j in range(4):
        B=np.abs(A[j,:,:int(para.FFTSize/2)])
        for i in range(win-1,para.Len):
            STDs=np.std(B[i-win+1:i+1,:],axis=0)
            ind[j,i]=np.argmax(STDs)
            d[j,i]=ind[j,i]*para.RRFFT/para.FFTSize
    return d,ind
    
    
def CalcHR(sig, win, step, para, ex):
    
    sos = signal.butter(3, [0.95*ex.minhr, 1.05*ex.maxhr], 'bp', fs=para.FPS, output='sos')
    b,a = signal.butter(3, [0.95*ex.minhr, 1.05*ex.maxhr], 'bp', fs=para.FPS)
    w,h=signal.freqz(b,a,fs=para.FPS)
    plt.figure(10)
    plt.plot(w,np.abs(h),color='dimgray', label='HR opseg')
    plt.title('Frekvencijska karakteristika filtara')
    plt.xlabel('f [Hz]')
    
    sig_filtered = signal.sosfilt(sos, sig)
    # plt.figure()
    # plt.plot(sig_filtered)
    # plt.xlim([0,1000])
    
    # b,a=signal.butter(3, [ex.minhr, ex.maxhr], 'bp', fs=para.FPS)
    w, gd = signal.group_delay((b,a))
    delay=floor(np.amax(gd))
    # plt.figure()
    # plt.title('Digital filter group delay')
    # plt.plot(w, gd)
    # plt.ylabel('Group delay [samples]')
    # plt.xlabel('Frequency [rad/sample]')
    # plt.show()
    
    hr_res=np.zeros(int(para.Len/step))
    f1=np.arange(0,para.FFTSize)*para.FPS/para.FFTSize
    for i in range(6,int(para.Len/step)):
        spekt=np.abs(fft(sig_filtered[(i*step-win-delay):(i*step-1-delay)],para.FFTSize))
        puls=f1[np.argmax(spekt)]*60
        hr_res[i]=puls
    return hr_res

def CalcBR(sig, win, step, para, ex):
    sos = signal.butter(3, [0.95*ex.minbr, 1.05*ex.maxbr], 'bp', fs=para.FPS, output='sos')
    b,a = signal.butter(3, [0.95*ex.minbr, 1.05*ex.maxbr], 'bp', fs=para.FPS)
    w,h=signal.freqz(b,a,fs=para.FPS)
    plt.figure(10)
    plt.plot(w,np.abs(h),label='BR opseg')
    plt.legend()
    
    sig_filtered = signal.sosfilt(sos, sig)
    # b,a=signal.butter(3, [ex.minhr, ex.maxhr], 'bp', fs=para.FPS)
    w, gd = signal.group_delay((b,a))
    delay=floor(np.amax(gd))
    print(delay)
    # plt.figure()
    # plt.title('Digital filter group delay')
    # plt.plot(w, gd)
    # plt.ylabel('Group delay [samples]')
    # plt.xlabel('Frequency [rad/sample]')
    # plt.show()
    br_res=np.zeros(int(para.Len/step))
    f1=np.arange(0,para.FFTSize)*para.FPS/para.FFTSize
    for i in range(6,int(para.Len/step)):
        spekt=np.abs(fft(sig_filtered[(i*step-win-delay):(i*step-1-delay)],para.FFTSize))
        puls=f1[np.argmax(spekt)]*60
        br_res[i]=puls
    return br_res

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
#%% Izbor ispitanika

plt.close('all')

para=Parameters()
participant=0
while (participant<1 or participant>50):
    print('Participant # :')
    participant = int(input())



#%% Ucitavanje raw podataka i GT vrednosti
data=ReadRadarData(os.path.dirname(__file__) + '\Children Dataset\FMCW Radar\Rawdata',participant)

data=np.reshape(data,(para.RxNum,para.Len,para.ADCSamp*para.ChirpsInFrame))
data=data.astype('complex')
refHR=ReadGTData(os.path.dirname(__file__) + '\Children Dataset\\Nihon Kohden\Heart Rate & Breathing Rate',participant,'Heart')
refBR=ReadGTData(os.path.dirname(__file__) + '\Children Dataset\\Nihon Kohden\Heart Rate & Breathing Rate',participant,'Breath')
refHR=refHR.to_numpy(dtype=float)
refBR=refBR.to_numpy(dtype=float)

refHRwave=ReadGTWaveData(os.path.dirname(__file__) + '\Children Dataset\\Nihon Kohden\Heartbeat & Breathing Waveform',participant,'Heart')
refBRwave=ReadGTWaveData(os.path.dirname(__file__) + '\Children Dataset\\Nihon Kohden\Heartbeat & Breathing Waveform',participant,'Breath')
refHRwave=refHRwave.to_numpy()
refBRwave=refBRwave.to_numpy()

extremes=ParaExtreme(refHR, refBR)
#%% Izdvajanje distance detektovanog predmeta

# sos = signal.butter(3, [18,64], 'bp', fs=para.FFTSize, output='sos')
# filtered = signal.sosfilt(sos, data,axis=2)

# #18 do 64
# #5 do 20
# b,a=signal.butter(3, [18,64], 'bp', fs=para.FFTSize)
# w, gd = signal.group_delay((b,a))
# delay=floor(np.amax(gd))


ant=1 #antena koju posmatramo trenutno
frame1=100

time1=1e6*np.arange(para.ChirpsInFrame*para.ADCSamp)*para.ChirpTime/para.ADCSamp
fig,ax=plt.subplots()
ax.plot(time1,np.real(data[ant-1,frame1,:]),color='darkblue',label='Re{IF}')
ax.plot(time1,np.imag(data[ant-1,frame1,:]),color='plum',label='Im{IF}')
plt.title('IF signal frejma iz trenutka t='+str(frame1/para.FPS)+' s\n Ispitanik '+str(participant))
plt.ylabel('Re{IF}, Im{IF}')
plt.xlabel('Vreme [\u03bcs]')
ax.legend()

X=GetFFT(data,para) #1024 odgovara 3Msps
# Xfilt=GetFFT(filtered,para)

f=np.arange(0,para.FFTSize)*para.RRFFT/para.FFTSize #udaljenosti u range FFT


fig1,ax1=plt.subplots()
ax1.plot(f,np.abs(X[0,frame1,:]),color='dimgrey')
plt.title('Anvelopa range-FFT signala u trenutku t='+str(frame1/para.FPS)+' s\n Ispitanik '+str(participant))
plt.xlabel('Distanca [m]')
plt.ylabel('Amplituda')
plt.xlim([0,2.75])
# ax1.plot(f,np.abs(Xfilt[0,frame1,:]),color='plum',label='izdvojen opseg od interesa')
# ax1.legend()

Y=DetectMovement(X,para) #vraca FFT koji je 99posto razlike ovog i prethodnog trenutka
Z=MeasureMovement(Y) #razlika ukupne snage u ovom i prethodnom trenutku


# d1=np.mean(GetDistance(Xfilt, para, 1),axis=1)
# d1=GetDistance(Xfilt, para, ant)


d2,ind2=GetDistanceNew(X, para, ant,100) #radi grubu detekciju udaljenosti predmeta
hr_all=np.zeros((4,300))
br_all=np.zeros((4,300))

for j in range(4):
        faza=np.zeros(para.Len)
    
        for i in range(99,para.Len):
            faza[i]=np.angle(X[ant-1,i,ind2[j,i]])
        
        faza=np.unwrap(faza)
        fazadiff=faza[1:]-faza[:-1]
        
        sig2=np.append([0],fazadiff)
    
    
    
        hr_res=CalcHR(sig2, 100, 20, para, extremes)
    
        hr_unif=ndimage.uniform_filter1d(hr_res, 16)
        hr_all[j,:]=hr_unif
        
        br_res=CalcBR(sig2, 100, 20, para, extremes)
        br_unif=ndimage.uniform_filter1d(br_res, 20)
        br_all[j,:]=br_unif
        
hr_final=np.mean(hr_all,axis=0)
refHR_unif=ndimage.uniform_filter1d(refHR,5)
    
    
br_final=np.mean(br_all,axis=0)

refBR_unif=ndimage.uniform_filter1d(refBR,10)
# faza=np.zeros(para.Len)
# for i in range(99,para.Len):
#     faza[i]=np.angle(X[ant-1,i,ind2[i]])
    
# faza=np.unwrap(faza)
# fazadiff=faza[1:]-faza[:-1]
# sig2=np.append([0],fazadiff)


time6000=np.arange(0,para.Len)/para.FPS
time300=np.arange(0,int(para.Len/para.FPS))


fig2,axs2=plt.subplots(2,1,sharex=True)
axs2[0].plot(time6000, faza, color='dimgrey')
axs2[0].set_xlabel('Vreme [s]')
axs2[0].set_ylabel('\u03c6 [rad]')
axs2[1].plot(time6000,sig2, color='dimgrey')
axs2[1].set_ylabel('\u0394\u03c6 [rad]')
plt.xlim([60,90])
fig2.suptitle('Zavisnost faze signala u vremenu\nIspitanik '+str(participant))


# spektar_faze=fft(fazadiff[100:300],para.FFTSize)



# plt.figure()
# plt.plot(np.abs(spektar_faze))
# plt.title('Amplitudski spektar evolucije faze')
# plt.xlim([0,100])


# fig2,ax2=plt.subplots()
# ax2.plot(np.arange(0,para.Len)/para.FPS,d1,color='darkblue')
# plt.title('Ispitanik '+str(participant)+': Distanca')
# plt.xlabel('Vreme [s]')
# plt.ylabel('Distanca [m]')
# plt.xlim([0,50])

# hr_res=CalcHR(sig2, 100, 20, para, extremes)

# hr_unif=ndimage.uniform_filter1d(hr_res, 16)
# refHR_unif=ndimage.uniform_filter1d(refHR,5)
# # hr_res_ma=running_mean(hr_res, 10)
# # refHR_ma=running_mean(refHR,10)




# fig3,ax3=plt.subplots()
# ax3.plot(time300,np.transpose(refHR_unif),color='darkblue',label='GT')
# ax3.plot(time300[6+16:],hr_unif[16:-6],color='plum',label='radar')
# plt.title('Referentni puls i estimacija, ispitanik '+str(participant))
# plt.xlabel('Vreme [s]')
# plt.ylabel('HR [bpm]')
# ax3.legend()
# plt.show()


# br_res=CalcBR(sig2, 100, 20, para, extremes)
# br_unif=ndimage.uniform_filter1d(br_res, 20)
# refBR_unif=ndimage.uniform_filter1d(refBR,10)
# # br_res_ma=running_mean(br_res, 10)
# # refBR_ma=running_mean(refBR,10)


# fig4,ax4=plt.subplots()
# ax4.plot(time300,np.transpose(refBR_unif),color='darkblue',label='GT')
# ax4.plot(time300[5+20:],br_unif[20:-5],color='plum',label='radar')
# plt.title('Referentni ritam disanja i estimacija')
# plt.xlabel('Vreme [s]')
# plt.ylabel('BR [bpm]')
# ax4.legend()
# plt.show()




fig5, axs=plt.subplots(3,1,sharex=True)
axs[0].plot(time300,np.transpose(refHR_unif),color='darkblue',label='GT')
axs[0].plot(time300[16:],hr_unif[16:],color='plum',label='radar')
axs[2].set_xlabel('Vreme [s]')
axs[0].set_ylabel('HR [bpm]')
axs[0].set_ylim([40, 200])
axs[1].plot(time300,np.transpose(refBR_unif),color='darkblue',label='GT')
axs[1].plot(time300[23:],br_unif[23:],color='plum',label='radar')
axs[1].set_ylabel('BR [bpm]')
axs[1].set_ylim([5, 55])
axs[2].plot(time6000,Z.MTotal,color='dimgrey')
axs[2].set_ylabel('Movement')
axs[0].legend(loc='upper left')
axs[1].legend(loc='upper left')
fig5.suptitle('Ispitanik broj '+str(participant))

#%%

# Analiza HR
y=np.transpose(refHR_unif)[20:-40]
x=np.reshape(hr_unif[20:-40],y.shape)
# 32 -25 i 26 -31
plt.figure()
plt.plot(y)
plt.plot(x)


numsim=np.ones(x.shape,dtype=float)-np.abs(x-y)/(np.abs(x)+np.abs(y))
tsim=np.mean(numsim)

rtsim=(np.sum(numsim**2)/x.size)**0.5

psim=np.sum(1-np.abs(x-y)/(2*np.maximum(np.abs(x),np.abs(y))))/x.size

# rxy=np.sum((x-np.ones(x.shape)*np.mean(x))*(y-np.ones(y.shape)*np.mean(y)))/(np.sum((x-np.ones(x.shape)*np.mean(x))**2)*np.sum((y-np.ones(y.shape)*np.mean(y))**2))**0.5

rxy,p=stats.pearsonr(x.flatten(),y.flatten())

cos=np.sum(x*y)/(np.sum(x**2)*np.sum(y**2))**0.5

plt.figure()
plt.scatter(x,y)
plt.plot([60,160],[60,160])
plt.title('Korelacija HR')
plt.xlabel('FMCW')
plt.ylabel('Clinical sensor')

plt.figure()
bland_altman_plot(x, y)
plt.title('Bland-Altman Plot')
plt.show()

# Analiza BR
y=np.transpose(refBR_unif)[23:-37]
x=np.reshape(br_unif[23:-37],y.shape)
# 35 -25 i 30 -30
plt.figure()
plt.plot(y)
plt.plot(x)

numsim=np.ones(x.shape,dtype=float)-np.abs(x-y)/(np.abs(x)+np.abs(y))
tsim=np.mean(numsim)

rtsim=(np.sum(numsim**2)/x.size)**0.5

psim=np.sum(1-np.abs(x-y)/(2*np.maximum(np.abs(x),np.abs(y))))/x.size

# rxy=np.sum((x-np.ones(x.shape)*np.mean(x))*(y-np.ones(y.shape)*np.mean(y)))/(np.sum((x-np.ones(x.shape)*np.mean(x))**2)*np.sum((y-np.ones(y.shape)*np.mean(y))**2))**0.5

rxy,p=stats.pearsonr(x.flatten(),y.flatten())

cos=np.sum(x*y)/(np.sum(x**2)*np.sum(y**2))**0.5

plt.figure()
plt.scatter(x,y)
plt.plot([18,60],[18,60])
plt.title('Korelacija BR')
plt.xlabel('FMCW')
plt.ylabel('Clinical sensor')

plt.figure()
bland_altman_plot(x, y)
plt.title('Bland-Altman Plot')
plt.show()