# import the Packages
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the data
df = yf.Ticker('MSFT').history('15y')
cc_rets = np.log(df.Close/df.Close.shift(1))[1:] # Close-to-Close daily returns
hl_rets = np.log(df.High/df.Low) # High-to-Low daily returns
co_rets = np.log(df.Close/df.Open) # Close-to-Open daily returns
oc_rets = np.log(df.Open/df.Close.shift(1))[1:] # Open-to-Close daily returns
# input Parameters
F = 1 # no of samples/day. e.g 5min sampling will be F=12*24 (12 5mins in an hour and 24hrs/day)
N = 21 # window size

#============================= close-to-close volatility ==================================#

sq_rets = cc_rets**2
var = sq_rets.rolling(N).mean().dropna()
cc_vol = np.sqrt(var * F * 252)

#=================================== parkinson volatility =================================#

sq_rets = hl_rets**2
var = sq_rets.rolling(N).mean().dropna()
park_vol = np.sqrt(var * F * 252 / (4 * np.log(2)))
        
#================================== Garman Klass Volatility ===============================#

hl_sq_rets = hl_rets**2
co_sq_rets = co_rets**2
var1 = 0.5 * hl_sq_rets.rolling(N).mean().dropna()
var2 = (2*np.log(2) - 1) * co_sq_rets.rolling(N).mean().dropna()
var = var1 - var2
gk_vol = np.sqrt(var * F * 252)
   
#=========================== Garman Klass Yang Zhang Volatility ============================#

oc_sq_rets = oc_rets**2
hl_sq_rets = hl_rets**2
co_sq_rets = co_rets**2
var1 = 0.5 * oc_sq_rets.rolling(N).mean().dropna()
var2 = 0.5 * hl_sq_rets.rolling(N).mean().dropna()
var3 = (2*np.log(2) - 1) * co_sq_rets.rolling(N).mean().dropna()
var = (var1 + var2 - var3).dropna()
gkyz_vol = np.sqrt(var * F * 252)

#==========================================================================================#
# Merging and Plotting
  
data_vol = 100 * pd.concat([cc_vol, park_vol, gk_vol, gkyz_vol], axis=1)
data_vol = data_vol.dropna()
data_vol.columns = ['Close-Close', 'Parkinson', 'Garman-Klass', 'GKYZ']
data_vol.plot()
plt.ylabel('% volatility')
plt.show()

# comparisons
data_vol[['Close-Close', 'Parkinson']].plot(ylabel='% volatility')
data_vol[['Close-Close', 'Garman-Klass']].plot(ylabel='% volatility')
data_vol[['Close-Close', 'GKYZ']].plot(ylabel='% volatility')
