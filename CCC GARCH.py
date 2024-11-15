import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt


stocks = ['AAPL', 'MSFT', 'TSLA']

prices = yf.Tickers(stocks).history('10Y')['Close']
returns = 100.0 * (prices / prices.shift(1)).apply(np.log).dropna()


# Apply the univariate GARCH Model to each of the returns series
results = {}
coefficients = pd.DataFrame()
std_residuals = pd.DataFrame()

for stock in stocks:
    res = arch_model(returns[stock], dist='t').fit(disp='off')
    results[stock]  = res
    coefficients[stock] = res.params
    std_residuals[stock] = res.std_resid
    # Plot the residuals and annualized conditional volatility
    res.plot(annualize='D')   
    plt.suptitle(stock)
    plt.show()

    
    
# Calculate the Bollerslev's Constant Conditional Correlation Estimator
T, N = std_residuals.shape
R = std_residuals.T.dot(std_residuals) / T
# rescaling
temp = np.tile(np.diag(R), (N,1))
R = R / (np.sqrt(temp.T * temp))
    

# Forecast one-step-ahead conditional covariance matrix
n = len(stocks)  
D = np.zeros((n, n))
diag = []
for stock in stocks:
    diag.append(results[stock].forecast(horizon=1).variance.values[-1][0])
diag = np.array(diag)**0.5
np.fill_diagonal(D, diag)
H = D @ R @ D
    
