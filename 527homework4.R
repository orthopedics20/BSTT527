# Import packages
library(tree)
library(randomForest)
library(gbm)
library(MASS)      # for LDA

### Question 1

## 1

# Train Test Split
set.seed(10)
diabetes$Outcome <- as.factor(diabetes$Outcome)

index <- sort(sample(1:nrow(diabetes), 500))
train <- diabetes[index, ]
test  <- diabetes[-index, ]

## 2

# Fit classification tree
tree.fit <- tree(Outcome ~ ., data = train)

# Summary of tree
summary(tree.fit)

# Training predictions & training error rate
train.pred.tree <- predict(tree.fit, newdata = train, type = "class")
train.err.tree  <- mean(train.pred.tree != train$Outcome)
train.err.tree


## 3

# Plot tree
plot(tree.fit)
text(tree.fit, pretty = 0)

# Predict on test data
test.pred.tree <- predict(tree.fit, newdata = test, type = "class")

# Confusion matrix & test error rate
tree.cm <- table(True = test$Outcome, Pred = test.pred.tree)
tree.cm

test.err.tree <- mean(test.pred.tree != test$Outcome)
test.err.tree

## 4

set.seed(10)
cv.tr <- cv.tree(tree.fit, FUN = prune.misclass)

# Look at CV results
cv.tr$size
cv.tr$dev

# Plot CV error vs tree size
plot(cv.tr$size, cv.tr$dev, type = "b", col = "green",
     xlab = "Tree size (number of terminal nodes)",
     ylab = "CV misclassification error (dev)",
     main = "Cross-validated error vs tree size")

min.dev <- which.min(cv.tr$dev)
points(cv.tr$size[min.dev], cv.tr$dev[min.dev],
       col = "red", cex = 2, pch = 20)

best.size <- cv.tr$size[min.dev]
best.size

## 5

prune.fit <- prune.misclass(tree.fit, best = best.size)

summary(prune.fit)  # for number of terminal nodes etc.

# Plot pruned tree
plot(prune.fit)
text(prune.fit, pretty = 0)

## 6

## Unpruned tree errors (already calculated)
train.err.tree
test.err.tree

## Pruned tree errors
train.pred.prune <- predict(prune.fit, newdata = train, type = "class")
train.err.prune  <- mean(train.pred.prune != train$Outcome)

test.pred.prune  <- predict(prune.fit, newdata = test, type = "class")
test.err.prune   <- mean(test.pred.prune != test$Outcome)

train.err.prune
test.err.prune

##7

set.seed(10)
p <- ncol(train) - 1   # number of predictors (exclude Outcome)

bag.fit <- randomForest(Outcome ~ ., data = train,
                        mtry = p,
                        importance = TRUE)

bag.fit   # brief summary

# Predict on test
bag.pred <- predict(bag.fit, newdata = test, type = "class")
bag.cm   <- table(True = test$Outcome, Pred = bag.pred)
bag.cm

test.err.bag <- mean(bag.pred != test$Outcome)
test.err.bag

# Variable importance plot
varImpPlot(bag.fit,
           main = "Variable importance: Bagging model")

## 8
set.seed(10)
rf.fit <- randomForest(Outcome ~ ., data = train,
                       importance = TRUE)

rf.fit

rf.pred <- predict(rf.fit, newdata = test, type = "class")
rf.cm   <- table(True = test$Outcome, Pred = rf.pred)
rf.cm

test.err.rf <- mean(rf.pred != test$Outcome)
test.err.rf

varImpPlot(rf.fit,
           main = "Variable importance: Random forest")

## 9

# Make numeric copy for boosting
train.boost <- train
test.boost  <- test

train.boost$Outcome <- as.numeric(as.character(train.boost$Outcome))
test.boost$Outcome  <- as.numeric(as.character(test.boost$Outcome))

set.seed(10)
boost.fit <- gbm(Outcome ~ .,
                 data = train.boost,
                 distribution = "bernoulli",
                 n.trees = 2000,
                 interaction.depth = 2,
                 shrinkage = 0.01,
                 bag.fraction = 0.5,
                 verbose = FALSE)

summary(boost.fit)  # relative influence plot

# Get training predictions and error
boost.train.prob <- predict(boost.fit, 
                            newdata = train.boost, 
                            n.trees = 2000, 
                            type = "response")

boost.train.pred <- ifelse(boost.train.prob > 0.5, 1, 0)
boost.train.pred <- factor(boost.train.pred, levels = c(0,1))
train.outcome.factor <- factor(train.boost$Outcome, levels = c(0,1))

# Training confusion matrix
boost.train.cm <- table(True = train.outcome.factor, Pred = boost.train.pred)
boost.train.cm

# Training error
train.err.boost <- mean(boost.train.pred != train.outcome.factor)
train.err.boost

# Predict probabilities on test set
boost.prob <- predict(boost.fit,
                      newdata = test.boost,
                      n.trees = 2000,
                      type = "response")

# Classify using 0.5 cutoff
boost.pred <- ifelse(boost.prob > 0.5, 1, 0)
boost.pred <- factor(boost.pred, levels = c(0,1))
test.outcome.factor <- factor(test.boost$Outcome, levels = c(0,1))

boost.cm <- table(True = test.outcome.factor, Pred = boost.pred)
boost.cm

test.err.boost <- mean(boost.pred != test.outcome.factor)
test.err.boost


## 10

lda.fit <- lda(Outcome ~ ., data = train)

lda.fit

# Get training predictions
lda.train.pred <- predict(lda.fit, newdata = train)
lda.train.class <- lda.train.pred$class

# Training confusion matrix
lda.train.cm <- table(True = train$Outcome, Pred = lda.train.class)
lda.train.cm

# Training error
train.err.lda <- mean(lda.train.class != train$Outcome)
train.err.lda

lda.pred <- predict(lda.fit, newdata = test)
lda.class <- lda.pred$class

lda.cm <- table(True = test$Outcome, Pred = lda.class)
lda.cm

test.err.lda <- mean(lda.class != test$Outcome)
test.err.lda

## 11

library(pROC)

# Class predictions
tree.pred.test <- predict(tree.fit, newdata = test, type = "class")

# Probabilities (needed for AUC)
tree.prob.test <- predict(tree.fit, newdata = test, type = "vector")[,2]

prune.pred.test <- predict(prune.fit, newdata = test, type = "class")

prune.prob.test <- predict(prune.fit, newdata = test, type = "vector")[,2]


bag.pred <- predict(bag.fit, newdata = test, type = "class")

bag.prob <- predict(bag.fit, newdata = test, type = "prob")   # two columns


bag_prob_1 <- bag.prob[,2]


rf.pred <- predict(rf.fit, newdata = test, type = "class")

rf.prob <- predict(rf.fit, newdata = test, type = "prob")
rf_prob_1 <- rf.prob[,2]


boost.prob   # numeric probabilities
boost.pred   # factor(0/1)

boost.pred <- factor(boost.pred, levels = c(0,1))
lda.pred <- predict(lda.fit, newdata = test)
lda.class <- lda.pred$class

lda.prob <- lda.pred$posterior    # two columns: [,1] and [,2]
lda_prob_1 <- lda.prob[,2]



compute_metrics <- function(true, pred_class, pred_prob) {
  true <- factor(true, levels = c(0, 1))
  pred_class <- factor(pred_class, levels = c(0, 1))
  
  cm <- table(true, pred_class)
  
  TN <- cm[1,1]
  FP <- cm[1,2]
  FN <- cm[2,1]
  TP <- cm[2,2]
  
  accuracy  <- (TP + TN) / (TP + TN + FP + FN)
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  f1 <- 2 * TP / (2*TP + FP + FN)
  
  auc <- as.numeric(auc(true, pred_prob))
  
  return(c(Accuracy = accuracy,
           Sensitivity = sensitivity,
           Specificity = specificity,
           F1 = f1,
           AUC = auc))
}

results <- rbind(
  Unpruned_Tree = compute_metrics(test$Outcome, tree.pred.test,  tree.prob.test),
  Pruned_Tree   = compute_metrics(test$Outcome, prune.pred.test, prune.prob.test),
  Bagging       = compute_metrics(test$Outcome, bag.pred,        bag.prob[,2]),
  RandomForest  = compute_metrics(test$Outcome, rf.pred,         rf.prob[,2]),
  Boosting      = compute_metrics(test$Outcome, boost.pred,      boost.prob),
  LDA           = compute_metrics(test$Outcome, lda.class,       lda.prob[,2])
)

round(results, 3)


### Question 2

## 1

set.seed(1)
n <- 150
x1 <- runif(n, -2, 2)
x2 <- runif(n, -2, 2)

# Nonlinear boundary: circle of radius ~1.2
y  <- ifelse(x1^2 + x2^2 > 1.2^2, 1, 0)
y  <- factor(y)

simdat <- data.frame(x1 = x1, x2 = x2, y = y)

# Quick plot
plot(simdat$x1, simdat$x2, col = as.numeric(simdat$y) + 1,
     pch = 19, xlab = "x1", ylab = "x2",
     main = "Simulated two-class data with nonlinear separation")
legend("topright", legend = levels(y), col = 2:3, pch = 19)

# Train/test split (2/3 vs 1/3)
set.seed(2)
train.idx <- sample(1:n, size = floor(2*n/3))
train2    <- simdat[train.idx, ]
test2     <- simdat[-train.idx, ]

## 2

library(e1071)

# Linear SVM
svm.lin <- svm(y ~ ., data = train2,
               kernel = "linear",
               cost = 1, scale = TRUE)

# Plot decision boundary on training data
plot(svm.lin, train2,
     main = "Linear SVM on nonlinear data")

# Predictions and error
pred.lin.train <- predict(svm.lin, newdata = train2)
pred.lin.test  <- predict(svm.lin, newdata = test2)

train.err.lin <- mean(pred.lin.train != train2$y)
test.err.lin  <- mean(pred.lin.test != test2$y)

train.err.lin
test.err.lin

## 2.5 linear tuning

library(e1071)

set.seed(123)

tune.lin <- tune(
  svm,
  y ~ .,
  data = train2,
  kernel = "linear",
  ranges = list(
    cost = c(0.01, 0.1, 1, 5, 10, 50, 100)
  )
)

summary(tune.lin)

best.lin <- tune.lin$best.model
best.lin

# Training predictions
lin.train.pred <- predict(best.lin, train2)
train.err.lin.tuned <- mean(lin.train.pred != train2$y)

# Test predictions
lin.test.pred <- predict(best.lin, test2)
test.err.lin.tuned <- mean(lin.test.pred != test2$y)

train.err.lin.tuned
test.err.lin.tuned

plot(best.lin, train2, main = "Tuned Linear SVM")



## 3

# Polynomial kernel (degree > 1)
svm.poly <- svm(y ~ ., data = train2,
                kernel = "polynomial",
                degree = 3,
                cost = 1,
                coef0 = 1,
                scale = TRUE)

plot(svm.poly, train2,
     main = "Polynomial (degree 3) SVM")

pred.poly.train <- predict(svm.poly, newdata = train2)
pred.poly.test  <- predict(svm.poly, newdata = test2)

train.err.poly <- mean(pred.poly.train != train2$y)
test.err.poly  <- mean(pred.poly.test != test2$y)

train.err.poly
test.err.poly

##3.5 Linear tuned

set.seed(123)

tune.poly <- tune(
  svm,
  y ~ .,
  data = train2,
  kernel = "polynomial",
  ranges = list(
    cost = c(0.1, 1, 5, 10),
    degree = c(2, 3, 4),
    gamma = c(0.01, 0.1, 0.5),
    coef0 = c(0, 1, 2)
  )
)

summary(tune.poly)
best.poly <- tune.poly$best.model
best.poly

# Training predictions
poly.train.pred <- predict(best.poly, train2)
train.err.poly.tuned <- mean(poly.train.pred != train2$y)

# Test predictions
poly.test.pred <- predict(best.poly, test2)
test.err.poly.tuned <- mean(poly.test.pred != test2$y)

train.err.poly.tuned
test.err.poly.tuned

plot(best.poly, train2, main = "Tuned Poly SVM")


## 4

# Radial (RBF) kernel SVM
svm.rad <- svm(y ~ ., data = train2,
               kernel = "radial",
               gamma = 1,
               cost = 1,
               scale = TRUE)

plot(svm.rad, train2,
     main = "Radial kernel SVM")

pred.rad.train <- predict(svm.rad, newdata = train2)
pred.rad.test  <- predict(svm.rad, newdata = test2)

train.err.rad <- mean(pred.rad.train != train2$y)
test.err.rad  <- mean(pred.rad.test != test2$y)

train.err.rad
test.err.rad

## 4 with tuning

library(e1071)

set.seed(123)

tune.rad <- tune(
  svm,
  y ~ .,
  data = train2,
  kernel = "radial",
  ranges = list(
    cost = c(0.1, 1, 5, 10, 50, 100),
    gamma = c(0.01, 0.05, 0.1, 0.5, 1, 2)
  )
)

summary(tune.rad)

best.rad <- tune.rad$best.model
best.rad

# Training predictions
rad.train.pred <- predict(best.rad, train2)
train.err.rad.tuned <- mean(rad.train.pred != train2$y)

# Test predictions
rad.test.pred <- predict(best.rad, test2)
test.err.rad.tuned <- mean(rad.test.pred != test2$y)

train.err.rad.tuned
test.err.rad.tuned

plot(best.rad, train2, main = "Tuned Radial SVM")


## 5

train.err.lin;  test.err.lin
train.err.poly; test.err.poly
train.err.rad;  test.err.rad

## 5.5 

lin.def.class <- predict(svm.lin, test2)
lin.def.prob  <- attr(predict(svm.lin, test2, decision.values = TRUE),
                      "decision.values")

lin.tuned.class <- predict(best.lin, test2)
lin.tuned.prob  <- attr(predict(best.lin, test2, decision.values = TRUE),
                        "decision.values")


poly.def.class <- predict(svm.poly, test2)
poly.def.prob  <- attr(predict(svm.poly, test2, decision.values = TRUE),
                       "decision.values")

poly.tuned.class <- predict(best.poly, test2)
poly.tuned.prob  <- attr(predict(best.poly, test2, decision.values = TRUE),
                         "decision.values")


rad.def.class <- predict(svm.rad, test2)
rad.def.prob  <- attr(predict(svm.rad, test2, decision.values = TRUE),
                      "decision.values")

rad.tuned.class <- predict(best.rad, test2)
rad.tuned.prob  <- attr(predict(best.rad, test2, decision.values = TRUE),
                        "decision.values")

# class predictions
lin.test.class <- predict(best.lin, test2)

# probability predictions
lin.test.prob <- attr(predict(best.lin, test2, decision.values = TRUE), 
                      "decision.values")


poly.test.class <- predict(best.poly, test2)

poly.test.prob <- attr(predict(best.poly, test2, decision.values = TRUE),
                       "decision.values")


rad.test.class <- predict(best.rad, test2)

rad.test.prob <- attr(predict(best.rad, test2, decision.values = TRUE),
                      "decision.values")

lin.def.metrics   <- compute_metrics(test2$y, lin.def.class,   lin.def.prob)
lin.tuned.metrics <- compute_metrics(test2$y, lin.tuned.class, lin.tuned.prob)

poly.def.metrics   <- compute_metrics(test2$y, poly.def.class,   poly.def.prob)
poly.tuned.metrics <- compute_metrics(test2$y, poly.tuned.class, poly.tuned.prob)

rad.def.metrics   <- compute_metrics(test2$y, rad.def.class,   rad.def.prob)
rad.tuned.metrics <- compute_metrics(test2$y, rad.tuned.class, rad.tuned.prob)



results <- rbind(
  Linear_Default    = lin.def.metrics,
  Linear_Tuned      = lin.tuned.metrics,
  Poly_Default      = poly.def.metrics,
  Poly_Tuned        = poly.tuned.metrics,
  Radial_Default    = rad.def.metrics,
  Radial_Tuned      = rad.tuned.metrics
)

round(results, 3)








