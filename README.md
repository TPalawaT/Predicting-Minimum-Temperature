# Predicting-Minimum-Temperature
## Overview
The python file uses time series to predict minimum temperature in Melbourne, Australia.</br>
The model uses ARIMA with (p,d,q) as (1,0,3)</br>
It has an error in prediction of 0.2° C.

## Code Explanation
We import several libraries namely numpy, pandas, r2_score from sklearn and ARIMA, adfuller from statsmodels.</br>
ARIMA stands for Auto Regressive Integrated Moving Average and is used to train our data and predict values.</br>
It takes values p,d,q for Auto regression, Sationarity, Moving average respectively.</br>
adfuller test is Dickey Fuller test for stationarity.</br>
The code commented from line 54 to 70 is a test for AIC value for different pdq values. This also plotted various graphs. It was commented due to the interruptions graphs were causing due to repeated looping.</br>

## Further Developements
I am now trying to include not just Melbourne but any place in the world. It will not just tell the data for one day. It will predict minimum temperature value for the next seven days for any city in thw world using API. </br>
Next I'll try to include everything a weather forecast needs to include, be it humidity, maximum temperature, rainfall etc and that will the third stage for the project. A full fledged weather predictor for tomorrow.

##  Contribution
Your contributions are always welcome.<br/>
Feel free to improve existing code, documentation or implement new algorithm.<br/>

###### Thanks for reading.
###### TPT
