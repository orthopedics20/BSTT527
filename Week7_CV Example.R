####Cross validation####
install.packages("boot")
install.packages("IRkernel")
library(boot)
IRkernel::installspec()      

### Objective 1: using LOOCV/CV on linear model with different levels of flexibility.
#(a) Generate a simulated data set as follows:
X <- rnorm (100) # generate 100 values from standard normal distribution
Y <- X - 2 * X^2 + rnorm (100) # generate Y from the model.

#(b) Create a scatterplot of X against Y . 
plot(X, Y)

#(c) Set a random seed, and then compute the LOOCV errors that result from 
#fitting the following four models:
set.seed(1)
dat <- data.frame(X, Y)  
glm.fit1 <- glm(Y~X, data = dat) 
cv.err1 <- cv.glm(data = dat, glm.fit1) # perform LOOCV for the model Y=b0+b1*X+epsilon
cv.err1$delta  # LOOCV error for the model Y=b0+b1*X+epsilon

glm.fit2 <- glm(Y~poly(X, 2), data = dat) 
cv.err2 <- cv.glm(data = dat, glm.fit2) # perform LOOCV for Y=b0+b1*X+b2*X^2+epsilon
cv.err2$delta  # LOOCV error for Y=b0+b1*X+b2*X^2+epsilon

glm.fit3 <- glm(Y~poly(X, 3), data = dat) 
cv.err3 <- cv.glm(data = dat, glm.fit3) # LOOCV for Y=b0+b1*X+b2*X^2+b3*X^3+epsilon
cv.err3$delta # LOOCV error for Y=b0+b1*X+b2*X^2+b3*X^3+epsilon

glm.fit4 <- glm(Y~poly(X, 4), data = dat) 
cv.err4 <- cv.glm(data = dat, glm.fit4) # LOOCV for Y=b0+b1*X+b2*X^2+b3*X^3+b4*X^4+epsilon
cv.err4$delta # LOOCV error for Y=b0+b1*X+b2*X^2+b3*X^3+b4*X^4+epsilon

###################################################################################################
#More explaination about cv.glm$delta
#The first component of delta is the average mean-squared error that you obtain from 
#doing K-fold CV.
#The second component of delta is the average mean-squared error that you obtain from doing  
#K-fold CV, but with a bias correction. How this is achieved is, initially, the residual 
#sum of squares (RSS) is computed based on the GLM predicted values and the actual response 
#values for the entire data set. As you're going through the K-folds, you generate a 
#training model, and then you compute the RSS between the entire data set of y-values 
#(not just the training set) and the predicted values from the training model. 
#These resulting RSS values are then subtracted from the initial RSS. After you're done going 
#through your K folds, you will have subtracted K values from the initial RSS.
#This is related second component of delta.
###################################################################################################

# write a loop for polynomial from 1 to 4
cv.error <- rep(0, 4)
for (i in 1:4) {
  glm.fit <- glm(Y~poly(X, i), data = dat)
  cv.error[i] <- cv.glm(dat, glm.fit)$delta[1]
  }
cv.error
plot(cv.error, xlab="polynomial", ylab="Loocv error")
lines(cv.error,col="red")

#(d) Repeat (c) using another random seed, try set.seed(100).
set.seed(100)
cv.error <- rep(0, 4)
for (i in 1:4) {
  glm.fit <- glm(Y~poly(X, i), data = dat)
  cv.error[i] <- cv.glm(dat, glm.fit)$delta[1]
}
cv.error

#(f) perming 5-folds CV instead of LOOCV
glm.fit1 <- glm(Y~X, data = dat) 
cv_5.err1 <- cv.glm(data = dat, glm.fit1, K=5) # perform 5-folds CV for Y=b0+b1*X+epsilon
cv_5.err1$delta  # 5-folds CV error for Y=b0+b1*X+epsilon

cv_5.error <- rep(0, 4)
for (i in 1:4) {
  glm.fit <- glm(Y~poly(X, i), data = dat)
  cv_5.error[i] <- cv.glm(dat, glm.fit, K=5)$delta[1]
}
cv_5.error
plot(cv_5.error, xlab="polynomial", ylab="5-folds cv error")
lines(cv_5.error,col="green")


### Objective 2: using CV on Logistic regression with different predictors.
####################################################################################
setwd("/Users/meidaw/Desktop/pvalue/courses/527/2025 Fall/HW/HW2")
SAheart <- read.table("SAheart.data.txt", header = T, sep = ",")

################################################################################## 
# logistic regression ---> 10-folds CV error
log.fit1=glm(chd~., family=binomial, data=SAheart)
log.cv10_err1 <- cv.glm(data = SAheart, log.fit1, K=10)
log.cv10_err1$delta[1]

log.fit2=glm(chd~sbp+adiposity, family=binomial,data=SAheart)
log.cv10_err2 <- cv.glm(data = SAheart, log.fit2, K=10)
log.cv10_err2$delta[1]
###################################################################################





