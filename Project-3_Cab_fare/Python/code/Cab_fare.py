##########################################################################################################################
#################################################   Cab fare Prediction    ############################################
##########################################################################################################################

############importing Libraries##########
import knnimpute

import os
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform
import datetime as dt
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from  matplotlib import pyplot

############working directory#################
os.chdir("C:/Users/ASUS/Desktop/Edwisor Training/Project-3_Cab_fare/Python")

###########loading file####################
cab_data = pd.read_csv("train_cab.csv")

test_data = pd.read_csv("test.csv")



############exploratory data analysis#######################
####Type Conversion#####
cab_data.columns
cab_data['fare_amount'].describe()
cab_data.dtypes
cab_data.shape
cab_data.head(5)

########Splitting daytime to day month year########
##Training data##
#Day
d1=cab_data['pickup_datetime'].copy()
for i in range (0,d1.shape[0]):
    d1[i]=dt.datetime.strptime(d1[i], "%Y-%m-%d  %H:%M:%S UTC").strftime('%d')

cab_data['day']=d1

#Month
d1=cab_data['pickup_datetime'].copy()
for i in range (0,d1.shape[0]):
    d1[i]=dt.datetime.strptime(d1[i], "%Y-%m-%d  %H:%M:%S UTC").strftime('%m')


cab_data['month']=d1

#Year
d1=cab_data['pickup_datetime'].copy()
for i in range (0,d1.shape[0]):
    d1[i]=dt.datetime.strptime(d1[i], "%Y-%m-%d  %H:%M:%S UTC").strftime('%Y')


cab_data['year']=d1

#hour
d1=cab_data['pickup_datetime'].copy()
for i in range (0,d1.shape[0]):
    d1[i]=dt.datetime.strptime(d1[i], "%Y-%m-%d  %H:%M:%S UTC").strftime('%H')


cab_data['hour']=d1

#Minutes
d1=cab_data['pickup_datetime'].copy()
for i in range (0,d1.shape[0]):
    d1[i]=dt.datetime.strptime(d1[i], "%Y-%m-%d  %H:%M:%S UTC").strftime('%M')


cab_data['minutes']=d1

#Seconds
d1=cab_data['pickup_datetime'].copy()
for i in range (0,d1.shape[0]):
    d1[i]=dt.datetime.strptime(d1[i], "%Y-%m-%d  %H:%M:%S UTC").strftime('%S')


cab_data['Seconds']=d1


cab_data.dtypes

#Date columns data type conversion
cab_data['day'] = cab_data['day'].astype('int')

cab_data['month'] = cab_data['month'].astype('int')

cab_data['year'] = cab_data['year'].astype('int')

#Time columns data type conversion
cab_data['hour'] = cab_data['hour'].astype('int')

cab_data['minutes'] = cab_data['minutes'].astype('int')

cab_data['Seconds'] = cab_data['Seconds'].astype('int')

cab_data.dtypes

#Drop datetime as it is of no use now
cab_data = cab_data.drop(['pickup_datetime'], axis=1)

##Test data##
#Day

d1=test_data['pickup_datetime'].copy()
for i in range (0,d1.shape[0]):
    d1[i]=dt.datetime.strptime(d1[i], "%Y-%m-%d  %H:%M:%S UTC").strftime('%d')

test_data['day']=d1

#Month
d1=test_data['pickup_datetime'].copy()
for i in range (0,d1.shape[0]):
    d1[i]=dt.datetime.strptime(d1[i], "%Y-%m-%d  %H:%M:%S UTC").strftime('%m')


test_data['month']=d1

#Year
d1=test_data['pickup_datetime'].copy()
for i in range (0,d1.shape[0]):
    d1[i]=dt.datetime.strptime(d1[i], "%Y-%m-%d  %H:%M:%S UTC").strftime('%Y')


test_data['year']=d1

#hour
d1=test_data['pickup_datetime'].copy()
for i in range (0,d1.shape[0]):
    d1[i]=dt.datetime.strptime(d1[i], "%Y-%m-%d  %H:%M:%S UTC").strftime('%H')


test_data['hour']=d1

#Minutes
d1=test_data['pickup_datetime'].copy()
for i in range (0,d1.shape[0]):
    d1[i]=dt.datetime.strptime(d1[i], "%Y-%m-%d  %H:%M:%S UTC").strftime('%M')


test_data['minutes']=d1

#Seconds
d1=test_data['pickup_datetime'].copy()
for i in range (0,d1.shape[0]):
    d1[i]=dt.datetime.strptime(d1[i], "%Y-%m-%d  %H:%M:%S UTC").strftime('%S')


test_data['Seconds']=d1


test_data.dtypes

#Date columns data type conversion
test_data['day'] = test_data['day'].astype('int')

test_data['month'] = test_data['month'].astype('int')

test_data['year'] = test_data['year'].astype('int')

#Time columns data type conversion
test_data['hour'] = test_data['hour'].astype('int')

test_data['minutes'] = test_data['minutes'].astype('int')

test_data['Seconds'] = test_data['Seconds'].astype('int')

test_data.dtypes

#Drop datetime as it is of no use now
test_data = test_data.drop(['pickup_datetime'], axis=1)




#######################################################################################

############Missing value analysis#################

##Checking Null values#



resp = cab_data.isnull().values.any()

print("Missing values in training data :",resp)

#Yes there are missing values in train data

resp = test_data.isnull().values.any()

print("Missing values in test data :",resp)

#There are no missing values in test data

#Let's go ahead with training dta
missing_val = pd.DataFrame(cab_data.isnull().sum())
missing_val


#reset index
missing_val=missing_val.reset_index()

missing_val

#Rename varibale
missing_val=missing_val.rename(columns = {'index':'variables',0: 'Missing_percentage'})

missing_val
#calculate percentage
missing_val['Missing_percentage']=(missing_val['Missing_percentage']/len(cab_data))*100

missing_val

#save output
missing_val.to_csv("missingInTrain.csv",index=False)

#Missing value treatment
#Removing all missing values
cab_data = cab_data.drop(cab_data[cab_data.isnull().any(1)].index, axis = 0)

resp = test_data.isnull().values.any()

print("Missing values in test data :",resp)

#No missing values now


##############Outlier Analysis########
cab_data.dtypes
plt.boxplot(cab_data['fare_amount'])
plt.show()

#we can see outliers here

plt.boxplot(cab_data['pickup_longitude'])
plt.show()


#we have already plotted in R now let us handle it here with code
#storing numerical columns
cname = cab_data.columns
#cname = cname.drop('pickup_datetime')
#cname = cname.drop('time')


for i in cname:
  q75,q25 = np.percentile(cab_data.loc[:,i],[75,25])
  
  iqr = q75 - q25
  
  min = q25 - (iqr*1.5)
  max = q75 + (iqr*1.5)
  
  cab_data=cab_data.drop(cab_data[cab_data.loc[:,i] < min].index)

  cab_data=cab_data.drop(cab_data[cab_data.loc[:,i] > max].index)

#Have removed all outliers


########Feature Selection########

##Correlation Analysis##
cname = cab_data.columns

#correlation plot
df_corr = cab_data.loc[:,cname]

#set width and height of plot
f, ax = plt.subplots(figsize=(7,5))

#Generate correlation matrix

corr = df_corr.corr()

#PLot using seaborn


sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool), cmap=sns.diverging_palette(220,10,as_cmap=True),
            square=True,ax=ax)
plt.show()


#########Modelling##########


#divide data into train and test

cname

train, test = train_test_split(cab_data, test_size = 0.2)

#Dicision tree for regression
#max depth = 2 means we are limiting depth of model 
#maximum node to a leaf is 2
fit = DecisionTreeRegressor().fit(train.iloc[:,1:12], train.iloc[:,0])

#Apply model on test data
#fit
predict_DT = fit.predict(test.iloc[:,1:12])
#predict_DT


#Calculate MAPE
def MAPE(y_true, y_pred):
  mape = np.mean(np.abs((y_true - y_pred) / y_true))
  return mape

MAPE(test.iloc[:,0], predict_DT)  

#0.33011

#Random forest regressor
RFmodel = RandomForestRegressor(n_estimators = 1000).fit(train.iloc[:,1:12], train.iloc[:,0])
RF_Predictions = RFmodel.predict(test.iloc[:,1:12])


MAPE(test.iloc[:,0], RF_Predictions)  

#MAPE = 0.24


#Linear Regression
model = sm.OLS(train.iloc[:,0], train.iloc[:,1:12]).fit()
predictions_LR = model.predict(test.iloc[:,1:12])

MAPE(test.iloc[:,0], predictions_LR)  

#MAPE = 0.53

#Least is 0.24 for RF
#We will make final prediction by RF model

#test_data.dtypes

result=pd.DataFrame(test_data.iloc[:,0:12])
RF_Predictions = RFmodel.predict(test_data.iloc[:,0:12])


result['Predicted_Fare'] = (RF_Predictions)

result.to_csv("Predicted_fare_RF.csv",index=False)

print("*****File has been created****")



