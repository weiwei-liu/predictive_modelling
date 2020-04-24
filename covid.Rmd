---
title: "Project of Data 583 Advanced Predictive Modelling"
author: "Bruno Santos, Harry Sun, Weiwei Liu"
date: "16/04/2020"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

This project aims to explore the Coronavirus dataset from Kaggle and forecast on the number of cases for the next 30 days. We will explore the data to find trends and forecast using ARIMA model. The dataset consists of around 4,935 rows of records of daily cases from January 22 to March 11. The data is mainly sourced out from Johns Hopkins University for education and academic research purposes, where they aggregated data from variou globalresources such as, World Health Organization, US Center for Disease Control, Canadian Government, China CDC, Italy Ministry of Health and others. 

In this project, we will try to model the number of confirmed infected cases of COVID-19 in Canada. The data was downloaded from Kaggle, which was provided by John Hopskin Hospital team. The data available are from 22th Jan. to 15th Apr.

We tried three different models `GLM`, `GAM` and `ARIMA`, and evaluated the goodness of fit of each model, then compared the advantage and disadvantage of these models and drew some conclusions. 



### 0.1 read the data

Read in Canada COVID-19 data

```{r}
train<-read.csv('train_canada.csv')
head(train,2)
```

### 0.2 Data visualization and evaluation of data quality

*Plot `log(count)~day` to get a sense of the data*

```{r}
count.total<-diff(train$ConfirmedCase)
day.total<-1:length(count.total)
plot(log(count.total)~day.total) 
```

From this plot, it can be seen that in the first twenty days of the data, the number of cases changed very slowly, this could be a concern in the modelling in terms of including or excluding this periods.

*Plot `British Columbia` data *

```{r}
train1<-read.csv('train.csv')
train1<-train1[which(train1$Country_Region=='Canada'),]
train1$province=droplevels(train1$Province_State)

count.total<-diff(train1[which(train1['province']=='British Columbia'),]$ConfirmedCases)
day.total<-1:length(count.total)
plot(log(count.total+1)~day.total)    # use `+1` to avoid log(0)
```

This plot shows that in day 40-80, there are several days, the daily increase is 0, which indicates the data was not updated daily, so this could cause some trouble in the modelling of this data.

And we also plot the data of different province, several provinces showed the same problem of updating data not daily. Another finding is that, different province shows different cases increasing patterns. This should also be beard in mind in the evaluation of the models, since we take the data of the whole country as ONE.


## 1.General Linear Model

### 1.1 Data Processing

In this analysis, we mainly focused on the `ConfirmedCases` modelling and prediction. The data was splited into two subsets. The first set of 75 data records were used to fit some models and the second subset is used to test the models.

```{r}
count.total<-diff(train$ConfirmedCase)
day<-1:length(count.total)
count<-count.total[1:75]
count.test<-count.total[76:83]
day<-day[1:75]
day.test<-day[76:83]
plot(log(count)~day)
```

### 1.2 Fit the GLM model

```{r}
covid.glm<-glm(count~day,family=poisson(link='log'))
summary(covid.glm)
```

```{r}
suppressMessages(library(MPV))
plot(day,log(count))
quadline(covid.glm)
```


### 1.3 Run Diagnostic check

```{r}
par(mfrow=c(2,2))
plot(covid.glm)
```

There are some concerns from this plot, the major one is that there are some outliers in the data strongly influenced the model from the `Cook's distance` plot, and the reponse data is not normally distributed from the `QQ` plot. In the `Scale-Location` plot, there is an upward trend which is concerning.

### 1.4 Conclusion

Overall, this model is **not** a good model, since the the reponse of the data after `log()` is not linear, so it is hard to use a linear model to fit the data. So nonlinear model could be a possible solution.

## 2. Generalized Additive Model

### 2.1 Fit the model

```{r}
suppressMessages(library(mgcv))
covid.gam<-gam(count~s(day),family=poisson(link='log'),method="REML")
summary(covid.gam)
```

Obtain plot of the estimate of the spline function, together with its confidence band.

```{r}
plot(covid.gam)
```

Check the residual deviance

```{r}
print(paste("Null deviance: ",covid.gam$null.deviance))

print(paste("GLM deviance: ",covid.glm$deviance))

print(paste("GAM deviance: ",covid.gam$deviance))
```

Compare with the `GLM` model in terms of residual deviance, this model improved largely. It would be a better model in this sense.

### 2.2 Prediction

Predict the test set and evaluate the model

```{r}
pre.gam=predict.gam(covid.gam,newdata=data.frame(day=seq(76,83)),type="response",se.fit = T)
pre.gam
```

```{r}
(pre.gam$fit-count.test)/pre.gam$se.fit
```

### 2.3 Conclusion

First conclusion: based on the difference of the fitted value and true value, most of the differnces are around 6~7 standard errors, which are really big. So the `GAM` model is **not** a good model in this sense.

*Residual plot*

```{r}
acf(residuals(covid.gam))
```

Second conclusion: based on this residual autocorrelation check, the model is autocorrelated. So, `ARIMA` model could be another possible solution.

```{r,include=F}
x <- seq(0, 100)
plot(x, predict(covid.gam, newdata = data.frame(day=x)),ylab="Expected Output", xlab="day", type="l") 
points(log(count)~day)
```

```{r,include=F}
suppressMessages(library(dplyr))

normalize <- function(x){
  (x - min(x))/(max(x)-min(x))
} 


train1<-read.csv('train.csv')
train1<-train1[which(train1$Country_Region=='Canada'),]
train1$province=droplevels(train1$Province_State)

train1=train1 %>%
  group_by(province) %>%
  mutate(count_std = normalize(ConfirmedCases)) 


train1[which(train1['province']=='Ontario'),]

count1<-diff(train1[which(train1['province']=='British Columbia'),]$count_std)+0.001
day<-1:length(count1)
plot(day,log(count1))
```

## 3. ARIMA

### 3.1 Fit the model

Determine if the `daily case counts` are autocorrelated, and if so, fit an appropriate ARMA
model to them. 
```{r}
acf(count)
```

From the above plot, it can be seen that the data is autocorrelated in some way. So we fitted a arima model using `auto.arima()`.

```{r}
suppressMessages(library(forecast))
covid.arima <- auto.arima(count)
covid.arima
```

### 3.2 Precitons

Predict the next 8 days number of cases.

```{r}
fc<-forecast(count,model=covid.arima,h=8)
```

```{r}
plot(fc)
lines ((length(count)+1):(length(count)+length(count.test)), count.test, col="red", lwd=3)
```

Calculate the standard error of the `point forecast`.

```{r}
fsd <- (fc$upper[,1] - fc$lower[,1]) / (2 * qnorm(.5 + fc$level[1] / 200))
fsd
```

Then, compare the predicted values with the actual count by subtracting the actual values from the predictions and dividing by the estimated standard errors.

```{r}
(fc$mean -  count.test)/fsd
```


### 3.3 Conclusions

Based on this calculation, most of differences between the fitted value and the actual daily case count is smaller than 2 standard errors. So, we accepted the null hypothesis the predicted and the acutal value are the same, and the model is a **proper** model.


## 4. Comments on models

### 4.1 Comparison of models

```{r}
print(paste("ARIMA AIC: ",AIC(covid.arima)))

print(paste("GLM AIC: ",AIC(covid.glm)))

print(paste("GAM AIC: ",AIC(covid.gam)))
```

In terms of AIC, clearly `ARIMA` model wins.

### 4.2 Thoughts

In this project, since the data is a time series model, so the `ARIMA` model shows its advantages. For `GLM` and `GAM` model, it could be possible to fit the data at large, but the prediction of the next 5 or 10 days could be dangerous, since these models are not suitable for extrapolation beyond the data range.

From the first part of this report, it showed that the data is not updated daily or the daily increase stays at 0 at the first phase of the transmission. So this could bring some extra noise to the data.

One thing can be improved for these models is that: since we use aggregated values from all the province, the variance of the data could be reduced if we construct different models for different provinces and then aggregate the predicted value.





