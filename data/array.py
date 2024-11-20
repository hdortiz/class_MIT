import numpy as np
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt
import math

#to = UTCDateTime(2019,9,17,2,10,53)
#dur = 4 #seconds


to = UTCDateTime(2019,9,17,4,21,20)
dur = 30 #seconds
st = read('6M.CON.0?.HDF.2019.260.mseed',starttime=to,endtime=to+dur)

#convert to physical units
st[0].data = st[0].data*(3.814697*10**-9)/(46*10**-6)
st[1].data = st[1].data*(3.814697*10**-9)/(46*10**-6)
st[2].data = st[2].data*(3.814697*10**-9)/(46*10**-6)

st.plot()


#%%filter
lf = 7
hf = 15.

st.detrend('linear')
st.filter('bandpass',freqmin=lf,freqmax=hf)
st.plot()
#%%


pos = np.genfromtxt('cone.csv', delimiter=',')

x = pos[:,1]*111.19*1000
y = pos[:,0]*111.19*1000

G = np.array([
    [x[0] - x[1], y[0] - y[1]],
    [x[0] - x[2], y[0] - y[2]],
    [x[1] - x[2], y[1] - y[2]]
])

    
ndata = len(st[0].data) - 1
    
cc01 = np.correlate(st[0].data, st[1].data,'full')
cc02 = np.correlate(st[0].data, st[2].data,'full')
cc12 = np.correlate(st[1].data, st[2].data,'full')
    
lag01 = cc01.argmax() - ndata
lag02 = cc02.argmax() - ndata
lag12 = cc12.argmax() - ndata
    
con = lag01-lag02+lag12
    
d = np.array([lag01,lag02,lag12])
m = np.linalg.inv(G.T @ G) @ (G.T @ d)

    
#%%
def bkaz(Sx, Sy):
    n = len(Sx)
    theta = np.full(n, np.nan)
    
    for i in range(n):
        if np.isfinite(Sx[i]):
            if Sx[i] > 0 and Sy[i] > 0:
                theta[i] = np.pi + np.arctan(Sx[i] / Sy[i])
            elif Sx[i] > 0 and Sy[i] < 0:
                theta[i] = 2 * np.pi + np.arctan(Sx[i] / Sy[i])
            elif Sx[i] < 0 and Sy[i] < 0:
                theta[i] = np.arctan(Sx[i] / Sy[i])
            elif Sx[i] < 0 and Sy[i] > 0:
                theta[i] = np.pi + np.arctan(Sx[i] / Sy[i])
    
    # Uncomment the line below to convert theta from radians to degrees
    theta = theta * 180 / np.pi
    return theta
#%%
s = np.zeros([1,2])
s[0,0] = m[0]
s[0,1] = m[1]


theta = np.arctan(m[1]/ m[0])
c = np.cos(theta)/m[0]

bka = bkaz(s[:,0],s[:,1])



