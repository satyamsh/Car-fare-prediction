####Clear objects####
rm(list=ls(all=T))

####set working directory#####

setwd("C:/Users/ASUS/Desktop/Edwisor Training/Project-3_Cab_fare/R")

###importing packages####
library(ggplot2)
#library(tidyverse) 

####Read file####
cab_data = read.csv("train_cab.csv")
cab_test_data = read.csv("test.csv")

####Glance at data ########
head(cab_data)
dim(cab_data)

#We have 16067 observation and 7 features in our train data##

head(cab_test_data)
dim(cab_test_data)

#We have 9914 observation and 6 features in test data##

####Checking structure of both data files#####
str(cab_data)

#we have first two features(fare_amount and pickup_datetime) as factor

str(cab_test_data)
# here also pickup_datetime is factor

######converting data types#####
#copy data ##

df1 = cab_data
testDF = cab_test_data

df1$fare_amount = as.numeric(df1$fare_amount)   
#testDF$fare_amount = as.numeric(testDF$fare_amount)

#We will separate date time values in pickup_datetime feature for ease in analysis
#and model

#Separating date from pickup_datetime to dteday

df1$date=as.Date(df1$pickup_datetime,format="%Y-%m-%d")

testDF$date=as.Date(testDF$pickup_datetime,format="%Y-%m-%d")

str(testDF)

#check type of dteday
str(df1$date)
str(testDF$date)

df1$day=format(as.Date(df1$date,format="%Y-%m-%d"), "%d")
df1$month=format(as.Date(df1$date,format="%Y-%m-%d"), "%m")
df1$year=format(as.Date(df1$date,format="%Y-%m-%d"), "%Y")


testDF$day=format(as.Date(testDF$date,format="%Y-%m-%d"), "%d")
testDF$month=format(as.Date(testDF$date,format="%Y-%m-%d"), "%m")
testDF$year=format(as.Date(testDF$date,format="%Y-%m-%d"), "%Y")

#Drop date variable as it is of no use
#str(df1)
df1 = df1[,-8]
testDF = testDF[,-7]

#Convert all to numeric for ease of feeding
df1$day=as.numeric(df1$day)
df1$month=as.numeric(df1$month)
df1$year=as.numeric(df1$year)

testDF$day=as.numeric(testDF$day)
testDF$month=as.numeric(testDF$month)
testDF$year=as.numeric(testDF$year)


#str(df1)
#str(testDF)



#same for time
df1$time = strptime(df1$pickup_datetime,"%Y-%m-%d %H:%M:%S")
testDF$time = strptime(testDF$pickup_datetime,"%Y-%m-%d %H:%M:%S")

library(data.table)
df1$time = as.ITime(df1$time) 
testDF$time = as.ITime(testDF$time) 

#df1=df1[,-12]
#str(df1)
#str(testDF)

#check type
#str(df1$daytime)
#str(testDF$daytime)

#Type is POSIXlt now
#Hence now we can feed this to as.ITime function which is part of "data.table" Library to fetch only time from pickup_datetime feature
#df2=df1


#Check type now
#str(df1$daytime)
#str(testDF$daytime)

#It is integer time format now

head(df1)

#drop daytime feature as we have splitted this in separate columns
#str(testDF[,1])
#str(df1[,2])

#df1 = df1[,-2]
#testDF = testDF[,-1]
df2=df1
df2$day = as.factor(df2$day)
#mean(df1$month)
#mean(df2$day)

#check the structure
str(df1)

###########Missing value analysis######
#Checking total missing values 
cat("Total missing value=",sum(is.na(df1)))

#Checking columns having missing values
colSums(sapply(df1, is.na))

#######Below is the explanation of function used below####
##data.frame = used to create a dataframe##
##apply = used for loop (as loop are slow we used apply fro faster processing##
##df1 = our dataset##
##function(x)=is the function we are creating in the line itself ##
##followed by {} which will count null values in each column##
##X = the count which we are getting from sunction##

missing_val = data.frame(apply(df1,2,function(x){sum(is.na(x))}))

missing_val2 = data.frame(apply(testDF,2,function(x){sum(is.na(x))}))

missing_val

missing_val2
##Test data does not have  missing values

####converting row names into columns###
missing_val$Columns=row.names(missing_val)
#missing_val2$Columns=row.names(missing_val2)

##remove index = remove frist variable##
row.names(missing_val) = NULL
#row.names(missing_val2) = NULL


##row names has been removed
##Rename the variable name of missing values##

names(missing_val)[1] = "Missing_Percentage" 
names(missing_val2)[1] = "Missing_Percentage" 


missing_val
missing_val2

##column name has been changed##

##calculating percentage of missing values###
missing_val$Missing_Percentage = (missing_val$Missing_Percentage/nrow(df1))*100
missing_val2$Missing_Percentage = (missing_val2$Missing_Percentage/nrow(testDF))*100


###Arrange in descending order to show highest percentage on top##
missing_val = missing_val[order(-missing_val$Missing_Percentage),]
missing_val2 = missing_val2[order(-missing_val2$Missing_Percentage),]

missing_val
missing_val2


##Missing Value observation##
#Test datadoe not have missing values
#Passenger_count has 0.3423 % of missing values 
#dteday and daytime has minor 0.00622 % of missing value
#Passenger count has 55 missing values and dteday & daytime has 1 missing value##

##Rearranging columns of missing value dataframe##
missing_val = missing_val[,c(2,1)]
#missing_val2 = missing_val2[,c(2,1)]

missing_val
#missing_val2

#2nd column has now became first column###

####Writing  the results into disk###
write.csv(missing_val,"missing_perce.csv",row.names = FALSE)
write.csv(missing_val2,"missing_perc_test_data.csv",row.names = FALSE)


###Plot graph for missing values###
ggplot(data=missing_val[1:5,],aes(x=reorder(Columns,-Missing_Percentage),y= Missing_Percentage))+
  geom_bar(stat = "identity",fill="grey")+ xlab("Parameters")+
  ggtitle("Missing data percentage(Cab Train)")+theme_bw()

#str(missing_val2)
#ggplot(data=missing_val2[1:3,],aes(x=reorder(Columns,-Missing_Percentage),y= Missing_Percentage))+
#  geom_bar(stat = "identity",fill="grey")+ xlab("Parameters")+
#  ggtitle("Missing data percentage(Cab Test)")+theme_bw()

####As per the rule if missing values are greater then 30% then we are not
####going to deal with that instead we will ignore that###

#####Missing value treatment ######
#Mean Method###
#df2=df1

#str(df2)
#df1$dteday=as.numeric(df1$dteday)
#df1$daytime=as.numeric(df1$daytime)
#testDF$dteday=as.numeric(testDF1$dteday)
#testDF$daytime=as.numeric(testDF$daytime)

#head(df1$dteday)

#we will apply this on copy of our data just to test##
#df2$passenger_count[is.na(df2$passenger_count)] = mean(df2$passenger_count,na.rm = T)

#sum(is.na(df2$daytime))
#have replaced nul values with mean of the column

#KNN imputation##
library(DMwR)
str(df1)
#df2=df2[,-1]
#colnames(df2)
#df2=df2[,-6]

#df2=df2[,-6]
#colnames(df2)

df1=knnImputation(df1,k=5)
#testDF=knnImputation(testDF,k=5)

##there are no missing values now###

#######Outlier Analysis########
##We can apply outlier analysis only on numerical varibales##
###Passenger count could have been converted to factor but we are 
##keeping it as numeric for ease of analysis and model feeding##

##Excluding fare amount as that is our target varible#
cnames = colnames(df1[,-1])
cnames

#Loop to plot boxplot for each column#


#Explanation of below code:
#as outlier can be performed on numeric data only we are 
#storing numeric features only
#storing required paramters to be passed in for graph
numeric_index = sapply(df1,is.numeric) 
numeric_data = df1[,numeric_index]
cnames = colnames(numeric_data)
cnames

#Assign - will assign name to a parameter passed to it
#Ex : assign("fg",data.frame(missing_val))
#will assign fg to dataframe

#inside assign we are giving random name
#so paste0 will join "gn" and i iteratively
#hence paste0 argument will become the name of reast of the object
#after ,


#In the below code we are storing each graph in name = gn"i"

for (i in 1:length(cnames))
  
{
  
  assign(paste0("Graph",i), ggplot(aes_string(y = (cnames[i]), x = "fare_amount"), data = subset(df1))+ 
           
           stat_boxplot(geom = "errorbar", width = 0.5) +
           
           geom_boxplot(outlier.colour="red", fill = "blue" ,outlier.shape=18,
                        
                        outlier.size=1, notch=FALSE) +
           
           theme(legend.position="bottom")+
           
           labs(y=cnames[i],x="Fare amount")+
           
           ggtitle(paste("Box plot of Fare_amount for",cnames[i])))
  
}

#Now we will plot all graphs together 
#gridextra is library name for grid.arrange function
#grid.arrange function is arranging plots side by side
#Didn't include Graph1 because it is for fare amount

gridExtra::grid.arrange(Graph2,Graph3,ncol=2)

gridExtra::grid.arrange(Graph4,Graph5,ncol=2)

gridExtra::grid.arrange(Graph6,Graph7,ncol=2)

gridExtra::grid.arrange(Graph8,Graph9,Graph10,ncol=3)


#Graphs are against fare_amount at x axis
#Graphs are representing ooutliers in red color
#Except day time month year we have outliers in each column

###Outlier treatment##
#Remove outliers##

#Explanation of below code##
#Example:
# val = df1$pickup_longitude[df1$pickup_longitude %in% boxplot.stats(df1$pickup_longitude)$out]
# In above code we are extracting outliers from graph
#%in% is used to search 
#out function is to detect outlier
# After detecting indexes of outliers we have removed them
for (i in cnames) {
  print(i)
  val = df1[,i][df1[,i] %in% boxplot.stats(df1[,i])$out]
  print(length(val))
  df1 = df1[which(!df1[,i] %in% val),]
  
}



##Second method to remove outliers is replace with NA 
#and use knnimputation to replace them
#Below is the code for same

#for (i in cnames) {
#  print(i)
#  val = df1[,i][df1[,i] %in% boxplot.stats(df1[,i])$out]
#  print(length(val))
#  df1[,i][df1[,i] %in% val] = NA
#  
#}
#df1 = knnImputation(df1,k=3)
#we can use mean method as well if knn gives error because
#of neighbours

##############Feature Selection########
####Correlation Analysis########
library(corrgram)
corrgram(df1[,numeric_index],order = F,
         upper.panel = panel.pie,text.panel = panel.txt,
         main="Correlation Plot")
str(df1)
#Blue color represents that variables are positively correlated
##Understanding
#pickup_longitude and pickup_latitude are highly correlated
#dropoff_longitude and dropoff_latitude are highly correlated

#We can drop one of above two observationss but we are keeping them for 
#distance because distance is very important factor
##Chi square test will be applied for categorical variables 
##We are not having that's why leaving it


####Dimension Reduction##
#df1 = subset(df1,select = -c(pickup_latitude,dropoff_latitude))
#str(df1)


######Feature Scaling#####
##Normalisation check##

##pickup_longitude
library(qqplotr)
qqnorm(df1$pickup_longitude)
hist(df1$pickup_longitude)

#dropoff_longitude
qqnorm(df1$dropoff_longitude)
hist(df1$dropoff_longitude)

#passenger_count
qqnorm(df1$passenger_count)
hist(df1$passenger_count)

#dteday
qqnorm(df1$day)
hist(df1$day)

#daytime
qqnorm(df1$time)
hist(df1$time)

#Normalisation

#cnames=colnames(df1[-1])
#cnames
#for (i in cnames) {
#  print(i)
#  df1[,i] = (df1[,i] - min(df1[,i])) /
#    (max(df1[,i]) - min(df1[,i]))
#}

#summary(df1$pickup_longitude)


############Modelling####################
##Sampling###
##Drop datetime from df1
df1 = df1[,-2]
str(df1)
train_index = sample(1:nrow(df1), 0.8 * nrow(df1))

train = df1[train_index,]

test = df1[-train_index,]

#######Dicision Tree #######

#library(C50)

##Explanation of code##
#c5.0 is function for dicision tree 
#fare amount will be independent variable 
#~. tells that consider other variables as dependent variable
#we can give ~Variable name as well
#trials is telling that design 100 trees and select best fit
#train is training data
#rules - extract the bussiness rules from model
#This is for categorical target
#c50_model = C5.0(fare_amount ~.,train, trials = 100, rules = TRUE)

# We will use rpart for regression
library(rpart)
fit = rpart(fare_amount ~ ., data = train, method = "anova")

str(test)

predict_DT = predict(fit, test[,-1])



##########Random forest#######

library(randomForest)

#Importance is to tell algorithm that show me the important variables
#and their calculation
#str(test)
model_RF = randomForest(fare_amount ~., train, importance = TRUE, ntree = 300)

predict_RF = predict(model_RF, test[,-1])


#########Linear regression###########
#check multicollinearity
library(usdm)

#finding correlation
vif(train[,-1])

#th - > setting threshhold
vifcor(train[,-1],th=0.9)

#Run regression model
library(e1071)
lm_model = lm(fare_amount ~., data = train) 

summary(lm_model)

predict_LM = predict(lm_model, test[,-1])

##Explanation of summary of model
#Residuals means Errors


###Accuracy ##
#Dicision tree
regr.eval(test[,1],predict_DT,stats = c('mae','rmse','mape','mse'))

#Random forest
regr.eval(test[,1],predict_RF,stats = c('mae','rmse','mape','mse'))

#Linear regression
regr.eval(test[,1],predict_LM,stats = c('mae','rmse','mape','mse'))

#Random forest has least MAPE so use that to predict

#########Model Tuning########
#check by increasing number of trees
#Tree = 500
#model_RF2 = randomForest(fare_amount ~., train, importance = TRUE, ntree = 500)

#predict_RF2 = predict(model_RF2, test[,-1])


#regr.eval(test[,1],predict_RF2,stats = c('mae','rmse','mape','mse'))

#3.13 error %
#Another attempt
#Tree = 700
#model_RF3 = randomForest(fare_amount ~., train, importance = TRUE, ntree = 700)

#predict_RF3 = predict(model_RF3, test[,-1])

#regr.eval(test[,1],predict_RF3,stats = c('mae','rmse','mape','mse'))

#Result is same
#Another attempt
#Number of trees = 1000
model_RF4 = randomForest(fare_amount ~., train, importance = TRUE, ntree = 1000)

predict_RF4 = predict(model_RF4, test[,-1])

regr.eval(test[,1],predict_RF4,stats = c('mae','rmse','mape','mse'))

#Result = Minor change in error
#We will go with last attempt i.e. tree = 1000 commenting rest all

#####Confusion Mtrix########
#library(caret)
#u = union(test$fare_amount,predict_RF)
#Confmatrix_rf = table(factor(test$fare_amount,u), factor(predict_RF,u))
#confusionMatrix(Confmatrix_rf)
#testDF$dteday = as.numeric(testDF$dteday)
#testDF = testDF[,-2]
#str(testDF)
#testDF = testDF[,-3]
str(testDF)
testDF = testDF[,-1]
final_predict_RF = predict(model_RF4, testDF)
testDF$Predicted_Fare = final_predict_RF
str(testDF)

#write file
write.csv(testDF,"Prediction.csv",row.names = FALSE)
rm(list=ls(all=T))
