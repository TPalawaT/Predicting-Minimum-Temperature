import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score

df = pd.read_csv('daily-minimum-temperatures-in-me.csv')

df.columns = ['date', 'temperature']  #Setting Colummn name
df['date'] = pd.to_datetime(df['date'], infer_datetime_format = True)
df['temperature'] = pd.to_numeric(df['temperature'])

print(df.head())
print(df.dtypes)
print(df.info())

df.set_index(['date'], inplace=True)  #Setting index for dataframe as date

#Setting up graph style
plt.rcParams['figure.figsize'] = (30, 10)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')

df.plot()  #Plotting the dataframe
plt.title('Data visualization')
plt.xlabel('date')
plt.ylabel('Temperature')

#Setting width of line here
leg  = plt.legend()
for line in leg.get_lines():
	line.set_linewidth(7)
plt.show()
#The graph shows it's stationary i.e. no variance or covariance

pd.plotting.autocorrelation_plot(df)
plt.title('ACF Graph')
plt.show()
#This graph follows seasonality i.e after a certain time the autocorrelation value goes negative.

#Dickey Fuller test for stationarity
#p-valueu is less than 0.05 and hence we can say the series is stationary
X = df.iloc[:,0].values
print(X)
result = adfuller(X)
print("ADF Statistic: %f" %result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#To check Arima model and plot relevant graphs for the residual values
'''
for i in range(1,3):
	for j in range(1,3):
		for k in range(1,3):
			model = ARIMA(df, order = (1,0,0))
			model_fit = model.fit(disp=0)
			print("For i=",i,"j=",j,"k=",k,"AIC=",model_fit.aic,"BIC=",model_fit.bic)
			residuals = pd.DataFrame(model_fit.resid) #The change in original value vs predicted value is residual
			residuals.plot()
			plt.show()
			residuals.plot(kind='kde')
			plt.show()


print("Residual data")
print(residuals.describe())
'''

#We iterate over test values to predict values
size = int(len(X)*0.8)
train, test = X[0: size], X[size:]

train = [x for x in train]
pred = list()
error = list()
for i in range(len(test)):
	model = ARIMA(train, order=(1,0,3))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	obs = test[i]
	pred.append(yhat)
	train.append(test[i])
	err = abs(yhat - obs)/obs
	error.append(err)
	print("Predicted=%f 	Expected=%f 	Error=%f"%(yhat,obs,err))

tot_err = sum(error)/len(error)
print("Error in dataset: ", tot_err)
print("Root Mean Square Error: ", r2_score(test, pred))