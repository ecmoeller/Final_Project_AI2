# 5512-Final_Project 
Using Time Series Algorithms for Predicting Global Land Temperatures for Major Cities

- Emily Moeller, [ecmoeller]
Developed in Python 3.7 (It will still run with Python 3.6 but the print statements will look ugly)

I primarily developed this program in jupyter notebooks, so to run a jupyter notebook you need to download Anaconda 
from this link https://www.anaconda.com/distribution/

Then, you simply run all cells within the notebook.

Alternatively, you also can run 

$ pip install statsmodels
$ python majorcitytemps.py

Introduction

  The Earth’s surface temperature record is one of the many pieces of evidence indicating that Earth is being affected by climate change. Researchers at Berkeley Earth, which is affiliated with Lawrence Berkeley National Laboratory, have proved this fact by studying the Earth’s temperature while dispelling the concerns that global warming skeptics had identified [1]. They assembled a variety of datasets that track the global land temperatures of cities, states and countries. Understanding the trend of global land temperatures is important in gaining greater insight of global warming as a whole and being able to adequately prepare for the way it will impact human life. For this reason, I will be using the data on global land temperatures in 100 major cities to develop models to predict the average temperatures for future dates. To accomplish this task, I first need to gain a background on the basic principles of time series algorithms. In particular, I learned about the different techniques that are used to satisfy the stationarity condition. After learning about the requirements for a time series algorithm, I reviewed a variety of time series algorithms such a Simple Moving Average, Exponential Moving Average, Auto-Regressive Integrated Moving Averages, and Seasonal Autoregressive Integrated Moving-Average (SARIMA). Because order is an important feature in time series algorithms, it is also impossible to shuffle the data and perform typical cross-validation techniques. As a result, I examined two different strategies to solve this problem. After all this research, I was able to apply what I learned to the original task of predicting land temperatures for 100 major cities around the globe. First, I give an overview of what the data looks like and what preprocessing steps are involved before applying a time series algorithm. After understanding the qualities of the data, I was able to choose the time series algorithm that I thought would work best. I discuss my implementation of this algorithm and evaluate my results, while reflecting on areas of improvement. 

For the rest of the report, see the FinalReport word doc or pdf.
