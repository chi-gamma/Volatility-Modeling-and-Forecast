import numpy as np
from sklearn.metrics import mean_squared_error
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
#=========================================================================================#
# Load the data
start = '2000-01-01'
end = '2021-10-20'
tic = 'SPY'
df = yf.Ticker(tic).history(start=start, end=end)[['Close']]
rets = 100 * ( np.log(df.Close/df.Close.shift(1))[1:] ) 
real_vol = rets.rolling(50).std().dropna()
real_vol.reset_index(drop=True, inplace=True)

#=========================================================================================#
# Plot the series to see what it looks like
fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
ax.plot(real_vol.index, real_vol, label='Realized Volatility')
ax.legend()
ax.yaxis.grid()
#=========================================================================================#
# Custom class to estimate the local level model
class MLELocalLevel(sm.tsa.statespace.MLEModel):
    start_params = [1.0, 1.0]
    param_names = ['obs.var', 'level.var']

    def __init__(self, endog):
        super(MLELocalLevel, self).__init__(endog, k_states=1)

        self['design', 0, 0] = 1.0
        self['transition', 0, 0] = 1.0
        self['selection', 0, 0] = 1.0

        self.initialize_approximate_diffuse()
        self.loglikelihood_burn = 1

    def transform_params(self, params):
        return params**2

    def untransform_params(self, params):
        return params**0.5

    def update(self, params, **kwargs):
        # Transform the parameters if they are not yet transformed
        params = super(MLELocalLevel, self).update(params, **kwargs)

        self['obs_cov', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[1]
#=========================================================================================#
# Maximum Likelihood Estimation

real_vol_model = MLELocalLevel(real_vol)
real_vol_results = real_vol_model.fit()
print(real_vol_results.summary())  
real_vol_results.plot_diagnostics(figsize=(13, 5));
plt.show()
#=========================================================================================#
# Construct the predictions / forecasts
n = 252
start = len(real_vol) - n
end = len(real_vol) - 1
real_vol_forecast = real_vol_results.get_prediction(start=start, end=end)
forecast = real_vol_forecast.predicted_mean

rmse = np.sqrt(mean_squared_error(forecast, real_vol[start:]))
print('The RMSE value of local level state space is {:.6f}'.format(rmse))

# Plotting
plt.plot(real_vol/100, label='realized volatility')
plt.plot(forecast/100, label='predicted')
plt.legend()
plt.show()
