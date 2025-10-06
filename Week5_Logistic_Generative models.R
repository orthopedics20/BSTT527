#######################################################
#   Logistic Regression, LDA, QDA, Naive Bayes        #                              
#######################################################

###################################################################################
# A sample of 2010 medicare data
# Variables:
# id = patient id
# readmit= flag for readmission within 30 days: 1 for readmission and 0 for not
# sex=1, male; sex=2, female (lower numbered or first alphabet is reference in R)
# race=1, white; race=2, black; race=3, others; race=5, hispanic
# los=length of stay in days during admission
# age=age of patient
# state=state of residence
# risk= HCC risk Score
# drg= admission diagnosis related group (medical, surgical, ungroup)
# er= number of ER visits in previous 12 months
###################################################################################

### load() for downloading Rdata file ############################################
setwd("/Users/meidaw/Desktop/pvalue/courses/527/2025 Fall/Week 3")
load("medicare2010Adm.Rdata")
x <- medicare10
head(x)
################################################################################## 

# Evaluate performance of classification problem (logistic regression)
log.fit <- glm(readmit~.,family=binomial,data=x[,-c(1,7)])
prob <- predict(log.fit,type=c("response"))
#prob <- log.fit$fit #Predictive probabilities same as fitted values in logistic regression

############ Objective 1: Confusion Matrix #######################################
table(round(prob),x$readmit) # x$readmit are true values, prob are predicted probs,
                             # round prob to 0 or 1 with threshold 0.5, get predicted response

# The confusionMatrix function in caret package produces the table and associated statistics
library(caret)
pred <-as.factor(round(prob))
confusionMatrix(pred, x$readmit, positive='1')

# Creating indicator variables with different threshold values 
pred.1 <- as.numeric(log.fit$fit > 0.5) # predicted response with threshold=0.5
head(pred.1)
table(pred.1, x$readmit) # sensitivity=6/(6+776)
confusionMatrix(as.factor(pred.1), x$readmit, positive='1')

# with threshold value 0.1
pred.2 <- as.numeric(log.fit$fit> 0.1)
head(pred.2)
table(pred.2, x$readmit) # sensitivity=223/(223+559)
confusionMatrix(as.factor(pred.2), x$readmit, positive='1')

# with threshold value 0.05
pred.3 <- as.numeric(log.fit$fit> 0.05)
head(pred.3)
table(pred.3, x$readmit) # sensitivity=518/(264+518)
confusionMatrix(as.factor(pred.3), x$readmit, positive='1')


############### Objective 2: ROC-AUC curve #########################################
# Receiver Operating Characteristic (ROC), or ROC curve, is a graphical plot that  #
# illustrates the performance of a binary classifier system with threshold varies. #
####################################################################################
#install.packages("pROC")
library(pROC)            
rocCurve <- roc(readmit ~ log.fit$fit, data=x) # response Y ~ predicted probs.
plot.roc(rocCurve, print.thres="best", legacy.axes = TRUE)
# "best" gives best threshold(s) with "largest" sensitivitity+specificitity
auc(rocCurve) # from this object, we can produce the area under the ROC curve
#plot.roc(rocCurve, print.thres="best", print.auc=TRUE, legacy.axes = TRUE) # plot with AUC.

# Area under ROC curve of 1 represents a perfect model fit; 
# an area of .5 represents a worthless test. 
# A rough guide for classifying the accuracy of a diagnostic fit/test is 
#.90-1 = excellent (A) 
#.80-.90 = good (B) 
#.70-.80 = fair (C) 
#.60-.70 = poor (D) 
#.50-.60 = fail (F) => ROC curve 45 degrees straight line


############ Objective 3: Linear Discriminant analysis ###################
# Note: In this medicare dataset, Some predictors are categorical.       #
# So LDA and QDA are not really applicable methods.                      #
##########################################################################
library(MASS)
data("iris") # Load data
head(iris)
summarysd<-function(x) { 
  c(summary(x),sd=sd(x)) 
}
par(mfrow=c(2,2))
for (i in 1:4){
  boxplot(iris[,i]~iris[,5])
  title(print(names(iris)[i]))
  print(i)
  print(tapply(iris[,i], iris[,5], summarysd))
}

? lda
lda.fit <- lda(Species~., data = iris, prior = c(1,1,1)/3)
# 50/150,50/150,50/150 is default prior in the example. not really needed!
names(lda.fit)
# Scatter plot using the first two discriminant components 
par(mfrow=c(1,1))
plot(lda.fit) 
# Better scatter plot
plot(lda.fit, panel = function(x, y, ...) points(x, y, ...),
     col = as.integer(iris$Species), pch = 20)
lda.pred <- predict(lda.fit, iris)
names(lda.pred)
lda.pred$class	# predicted outcome classes
lda.pred$post	  # posterior probabilities

# Accuracy of predictions/classification
con.matrix<-table(lda.pred$class,iris$Species)
con.matrix
sum(diag(con.matrix))/nrow(iris) # classification accuracy
1-sum(diag(con.matrix))/nrow(iris) # misclassification error

# Multiclass AUC is a mean of several AUC and cannot be plotted. 
library(pROC)
mul.roc <- multiclass.roc(iris$Species ~ lda.pred$post)
mul.roc$auc

#######################################################################################
# lda output includes 
# i. the prior probability of each class, 
# ii. counts for each class in the data, 
# iii. prior class-specific means for each feature, 
# iv. linear combination coefficients (scaling) for each linear discriminant. 
#     In 3 classes/categories we have at most two(3-1) linear discriminants. 
# v. the singular values (svd) that gives the ratio of the between- and 
#    within-group standard deviations on the linear discriminant variables.
# vi. LD1-LD2 are different from our linear discriminant (delta) function from the Bayes Rule.
#     They are (K-1) orthogonal Fisher LD functions vectors to separate observations into classes.
#     They can be used as dimensional reduction but they are derived using the features and also class labels. 
#     (So they are different from PCs of PCA, the standard dimensional reduction method, which consider only the features.) 
#     The coefficients of LD determine the Fisher's linear discriminants LD1, LD2. 
#     LD1 is a linear function that achieves the maximal separation between classes. 
#     LD2 is a linear function, orthogonal to LD1, that achieves the maximal separation 
#     among all linear functions orthogonal to LD1, etc. 
#     These functions are linear combinations of our linear discriminant functions.
#     Here, LD1 captures 99% of differences  between the classes, 
#           and LD2 captures remaining 1%.
########################################################################################


############ Objective 4: Quadratic Discriminant analysis ########################
qda.fit <- qda(formula = Species ~ ., data = iris, prior = c(1,1,1)/3)
qda.pred <- predict(qda.fit, iris)
con.matrix2 <- table(qda.pred$class, iris$Species) # confusion matrix
con.matrix2
sum(diag(con.matrix2))/nrow(iris) # classification accuracy
1-sum(diag(con.matrix2))/nrow(iris)  # miclassification error
# Multiclass AUC is a mean of several AUC and cannot be plotted. 
mul.roc2 <- multiclass.roc(iris$Species ~ qda.pred$post)
mul.roc2$auc


############ Objective 5: Naive Bayes Classifier #####################################
# Naive Bayes Classifier in R using naiveBayes() in library(e1071)                   #
# If Some predictors are categorical, LDA and QDA are not really applicable methods. #
# You may use logistic but requires quite bit work for large datasets                #
# Naive Bayes is quick and dirty method                                              #
######################################################################################
#install.packages("e1071")
library(e1071)
naive.fit <- naiveBayes(Species~ ., data = iris)
naive.pred <- predict(naive.fit, iris)
con.matrix3 <- table(naive.pred, iris$Species)
con.matrix3
sum(diag(con.matrix3))/nrow(iris) # classification accuracy
1-sum(diag(con.matrix3))/nrow(iris)  # miclassification error





