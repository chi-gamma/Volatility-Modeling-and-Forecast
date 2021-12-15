#=============================== import the relevant packages =============================#

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as RFR

#==========================================================================================#
#===================================== download the data ==================================#

start = '2003-12-01'
end = '2021-12-01'
tic = 'NG' # Natural Gas
df = yf.Ticker(tic).history(start=start, end=end)[['Close']]

#==========================================================================================#
#======================================= calculations =====================================#

rets = 100 * ( np.log(df.Close/df.Close.shift(1))[1:] ) 
real_vol = rets.rolling(5).std()
real_vol = real_vol.dropna()

#==========================================================================================#
#========================================= Regression =====================================#

rets_rf = rets ** 2
n = 252 # number of days to forecast
X = pd.concat([real_vol, rets_rf], axis=1).dropna()
X.reset_index(drop=True, inplace=True)
X.columns = ['real_vol', 'rets_sq']
y = real_vol.iloc[1:-(n-1)].values
regr = RFR(n_estimators = 300, max_depth=5, random_state=0)
regr.fit(X.iloc[:-n].values, y)
predict_rf= regr.predict(X[-n:]) # make prediction on the last 252 days
predict_rf = pd.DataFrame(predict_rf)
predict_rf.index = rets.iloc[-n:].index

#==========================================================================================#
#======================== calculate the RMSE of the predictions ===========================#

rmse_rf = np.sqrt(mean_squared_error(real_vol.iloc[-n:] / 100, predict_rf / 100))
print('The RMSE value of random forest is {:.6f}'.format(rmse_rf))

#==========================================================================================#
#======================================== plotting ========================================#

real_vol.index = rets.iloc[len(rets)-len(real_vol):].index 
plt.figure(figsize=(10, 6))
plt.plot(real_vol / 100, label= 'Realized Volatility') 
plt.plot(predict_rf / 100, label= 'Volatility Prediction-RF-GARCH')
plt.title('Volatility Prediction with Random Forest-GARCH', fontsize=12)
plt.gca().yaxis.grid(True)
plt.legend()
plt.show()