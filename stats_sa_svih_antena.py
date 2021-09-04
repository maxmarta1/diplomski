# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:38:11 2021

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
import matplotlib.transforms as transforms
import csv

class Results:
    def __init__(self):
        self.x=[]
        self.y=[]
        self.psim=[]
        self.rtsim=[]
        self.rxy=[]
        self.tsim=[]
        self.cos=[]
    
    def add_stats(self,xnew,ynew,psimnew,tsimnew,rtsimnew,rxynew,cosnew):
        self.x.append(xnew)
        self.y.append(ynew)
        self.psim.append(psimnew)
        self.tsim.append(tsimnew)
        self.rtsim.append(rtsimnew)
        self.rxy.append(rxynew)
        self.cos.append(cosnew)
    

        
        
        
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

def GetFFT(data,para):
    X=np.empty((para.RxNum,para.Len,para.FFTSize),dtype=complex)
    for i in range(0,para.RxNum):
        for j in range(0,para.Len):
            X[i,j,:]=fft(data[i,j,:],para.FFTSize)
    return X


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
    sig_filtered = signal.sosfilt(sos, sig)

    
    b,a=signal.butter(3, [0.95*ex.minhr, 1.05*ex.maxhr], 'bp', fs=para.FPS)
    w, gd = signal.group_delay((b,a))
    delay=floor(np.amax(gd))

    
    hr_res=np.zeros(int(para.Len/step))
    f1=np.arange(0,para.FFTSize)*para.FPS/para.FFTSize
    for i in range(6,int(para.Len/step)):
        spekt=np.abs(fft(sig_filtered[(i*step-win-delay):(i*step-1-delay)],para.FFTSize))
        puls=f1[np.argmax(spekt)]*60
        hr_res[i]=puls
    return hr_res

def CalcBR(sig, win, step, para, ex):
    sos = signal.butter(3, [0.95*ex.minbr, 1.05*ex.maxbr], 'bp', fs=para.FPS, output='sos')
    sig_filtered = signal.sosfilt(sos, sig)
    b,a=signal.butter(3, [0.95*ex.minbr, 1.05*ex.maxbr], 'bp', fs=para.FPS)
    w, gd = signal.group_delay((b,a))
    delay=floor(np.amax(gd))

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

def bland_altman_plot(data1, data2,ax, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    line1=md+1.96*sd
    line2=md-1.96*sd
    ax.scatter(mean, diff, *args, **kwargs)
    ax.set_xlabel('(GT+Radar)/2')
    ax.set_ylabel('Radar-GT')
    ax.axhline(md,           color='gray', linestyle='--')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
    trans = transforms.blended_transform_factory(
    ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(1.12,line1, "%.2f" % line1, color="red", transform=trans, 
        ha="right", va="center")
    ax.text(1.12,md, "{:.2f}".format(md), color="red", transform=trans, 
        ha="right", va="center")
    ax.text(1.12,line2, "%.2f" % line2, color="red", transform=trans, 
        ha="right", va="center")
    # ax[1].hist(diff)
    
def corr_plot(x,y,ax,lim1,lim2):
    
    M=np.hstack((x,y))
    n,k=np.shape(M)


    SStotal = np.var(M)*(n*k - 1);
    MSR = np.var(np.mean(M, 1)) * k;
    MSW = np.sum(np.var(M,1)) / n;
    MSC = np.var(np.mean(M, 0)) * n;
    MSE = (SStotal - MSR *(n - 1) - MSC * (k -1))/ ((n - 1) * (k - 1));

    # r = (MSR - MSW) / (MSR + (k-1)*MSW);
    r1 = (MSR - MSE) / (MSR + (k-1)*MSE);
    r2=(MSR - MSE) /(MSR+(k-1)*MSE+k*(MSC-MSE)/n)
    
    m,b=np.polyfit(x[:,0],y[:,0],1)
    
    
    ax.scatter(x,y)
    ax.plot([lim1,lim2],[lim1,lim2],'--',color='black')
    ax.set_xlabel('FMCW Radar')
    ax.set_ylabel('GT')
    # trans = transforms.blended_transform_factory(
    # ax.get_yticklabels().get_transform(), ax.transData)
    ax.text(lim1,lim2, "ICC1=%.3f  ICC2=%.3f" % (r1,r2), color="blue", 
        ha="left", va="center")
    ax.text(lim1,lim2-5, "y=%.2fx + %.2f" % (m,b), color="red",
        ha="left", va="center")
    ax.plot([lim1,lim2],[m*lim1+b,m*lim2+b],color='red')

#%% Izbor ispitanika
plt.close('all')
BR_Results=Results()
HR_Results=Results()


para=Parameters()

for participant in range(1,51):




#%% Ucitavanje raw podataka i GT vrednosti
    data=ReadRadarData(os.path.dirname(__file__) + '\Children Dataset\FMCW Radar\Rawdata',participant)
    
    data=np.reshape(data,(para.RxNum,para.Len,para.ADCSamp*para.ChirpsInFrame))
    data=data.astype('complex')
    refHR=ReadGTData(os.path.dirname(__file__) + '\Children Dataset\\Nihon Kohden\Heart Rate & Breathing Rate',participant,'Heart')
    refBR=ReadGTData(os.path.dirname(__file__) + '\Children Dataset\\Nihon Kohden\Heart Rate & Breathing Rate',participant,'Breath')
    refHR=refHR.to_numpy(dtype=float)
    refBR=refBR.to_numpy(dtype=float)
    
    extremes=ParaExtreme(refHR, refBR)
    #%% Izdvajanje distance detektovanog predmeta
    
    ant=1 #antena koju posmatramo trenutno
    frame1=100
    
    X=GetFFT(data,para) #1024 odgovara 3Msps
    
    
    f=np.arange(0,para.FFTSize)*para.RRFFT/para.FFTSize #udaljenosti u range FFT
    
    
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
    
    
    #%%
    hr_unif=hr_final
    br_unif=br_final
    # Analiza HR
    y=np.transpose(refHR_unif)[23:-37]
    x=np.reshape(hr_unif[23:-37],y.shape)
    # y=np.transpose(refHR_unif)[17:-40]
    # x=np.reshape(hr_unif[11:-46],y.shape)
    
    
    numsim=np.ones(x.shape,dtype=float)-np.abs(x-y)/(np.abs(x)+np.abs(y))
    tsim=np.mean(numsim)
    
    rtsim=(np.sum(numsim**2)/x.size)**0.5
    
    psim=np.sum(1-np.abs(x-y)/(2*np.maximum(np.abs(x),np.abs(y))))/x.size
    
    
    rxy,p=stats.pearsonr(x.flatten(),y.flatten())
    
    cos=np.sum(x*y)/(np.sum(x**2)*np.sum(y**2))**0.5
    
    HR_Results.add_stats(x,y,psim,tsim,rtsim,rxy,cos)
    
    # Analiza BR
    y=np.transpose(refBR_unif)[23:-37]
    x=np.reshape(br_unif[23:-37],y.shape)
    # y=np.transpose(refBR_unif)[20:-40]
    # x=np.reshape(br_unif[15:-45],y.shape)
    
    numsim=np.ones(x.shape,dtype=float)-np.abs(x-y)/(np.abs(x)+np.abs(y))
    tsim=np.mean(numsim)
    
    rtsim=(np.sum(numsim**2)/x.size)**0.5
    
    psim=np.sum(1-np.abs(x-y)/(2*np.maximum(np.abs(x),np.abs(y))))/x.size
    
    rxy,p=stats.pearsonr(x.flatten(),y.flatten())
    
    cos=np.sum(x*y)/(np.sum(x**2)*np.sum(y**2))**0.5
    
    BR_Results.add_stats(x,y,psim,tsim,rtsim,rxy,cos)
#%%

BRx=np.vstack(BR_Results.x)
BRy=np.vstack(BR_Results.y)


fig1,ax1=plt.subplots()
bland_altman_plot(BRx, BRy,ax1)
plt.title('Bland Altman za BR')

HRx=np.vstack(HR_Results.x)
HRy=np.vstack(HR_Results.y)


fig2,ax2=plt.subplots()
bland_altman_plot(HRx, HRy,ax2)
plt.title('Bland Altman za HR')

# fig3,axs=plt.subplots(3,1,sharex=(True),sharey=True)
# fig3.suptitle('BR mere slicnosti')
# axs[0].stem(BR_Results.tsim,label='tsim')
# axs[1].stem(BR_Results.psim,label='psim')
# axs[2].stem(BR_Results.rtsim, label='rtsim')
# axs[2].set_xlabel('redni broj ispitanika')
# axs[0].set_ylim([0.9,1])

# fig4,axs=plt.subplots(3,1,sharex=(True),sharey=(True))
# fig4.suptitle('HR mere slicnosti')
# axs[0].stem(HR_Results.tsim,label='tsim')
# axs[1].stem(HR_Results.psim,label='psim')
# axs[2].stem(HR_Results.rtsim, label='rtsim')
# axs[2].set_xlabel('redni broj ispitanika')
# axs[0].set_ylim([0.95,1])

# fig5,ax5=plt.subplots()
# fig5.suptitle('Koeficijenti korelacije')
# ax5.plot(HR_Results.rxy,color='darkblue',label='HR')
# ax5.plot(BR_Results.rxy,color='plum',label='BR')
# ax5.set_xlabel('redni broj ispitanika')
# ax5.legend()

#%%

  
# # field names 
# fields = range(1,51)
    
# # data rows of csv file 
# rows = [ BR_Results.psim, 
#          BR_Results.tsim, 
#          BR_Results.rtsim, 
#          BR_Results.cos, 
#          BR_Results.rxy] 
  
# with open('br_results.csv', 'w+') as f:
      
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
      
#     write.writerow(fields)
#     write.writerows(rows)
# f.close()    
# # field names 
# fields = range(1,51)
    
# # data rows of csv file 
# rows = [ HR_Results.psim, 
#          HR_Results.tsim, 
#          HR_Results.rtsim, 
#          HR_Results.cos, 
#          HR_Results.rxy] 
  
# with open('hr_results.csv', 'w+') as f:
      
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
      
#     write.writerow(fields)
#     write.writerows(rows)
# f.close()    
    
#%%

fig6,ax6=plt.subplots()
corr_plot(BRx,BRy,ax6,10,60)
fig6.suptitle('BR korelacija')

fig7,ax7=plt.subplots()
corr_plot(HRx,HRy,ax7,60,160)
fig7.suptitle('HR korelacija')

#%%

age=pd.read_excel(os.path.dirname(__file__) +'\Children Dataset\Participant\Human Data\HumanData.xlsx',usecols=[2])
age=age.to_numpy(dtype=int)
ages=np.empty((0,1))
for i in range(50):
    temp=np.ones((12,1))*age[i]
    ages=np.vstack((ages,temp))
    
BR_set=BRx.reshape((600,20))
HR_set=HRx.reshape((600,20))

bins=np.array([0,72,156])
x=HR_set[:,np.newaxis,:]
y=np.digitize(ages,bins)


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head
from sktime.utils.slope_and_trend import _slope 
 

X_train, X_test, y_train, y_test = train_test_split(x, y)

from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
steps = [
      (
          "extract",
          RandomIntervalFeatureExtractor(
              n_intervals="sqrt", features=[np.mean, np.std, _slope]
          ),
      ),
      ("clf", DecisionTreeClassifier()),
]
time_series_tree = Pipeline(steps)
 
time_series_tree.fit(X_train, y_train)
print(time_series_tree.score(X_test, y_test))
 
tsf = TimeSeriesForestClassifier( 
      n_estimators=1000,
      random_state=1,
      n_jobs=-1,
) 
 
tsf.fit(X_train, y_train)

y_pred=tsf.predict(X_test)

print(tsf.score(X_test,y_test))


from sktime.classification.interval_based import RandomIntervalSpectralForest

rise = RandomIntervalSpectralForest(n_estimators=10)
rise.fit(X_train, y_train)
print(rise.score(X_test, y_test))

