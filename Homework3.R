# 1 — Diabetes dataset: classification & comparisons
#-------------------------------------------------------

# ------------- setup -------------
library(caret)    # confusionMatrix
library(pROC)     # ROC, AUC
library(MASS)     # lda, qda
library(e1071)    # naiveBayes
library(boot)     # cv.glm, boot
library(leaps)    # regsubsets (used later)
library(ggplot2)

# set working directory appropriately or provide full path to files


# Inspect:
str(diabetes)
# Ensure Outcome is factor for classification where needed:
diabetes$Outcome <- as.factor(diabetes$Outcome)  # 0/1 as factor

# 1.1 Logistic Regression

log.fit <- glm(Outcome ~ ., data = diabetes, family = binomial)
print(log.fit)
prob <- predict(log.fit, type = "response")

metrics_for_threshold <- function(probs, truth_factor, theta) {
  pred <- ifelse(probs >= theta, "1", "0")
  pred <- factor(pred, levels = levels(truth_factor))
  cm <- confusionMatrix(pred, truth_factor, positive = "1")
  # extract desired metrics
  list(
    threshold = theta,
    confusion = cm$table,
    accuracy = cm$overall["Accuracy"],
    precision = cm$byClass["Pos Pred Value"],  # PPV
    sensitivity = cm$byClass["Sensitivity"],    # recall
    cm_obj = cm
  )
}

thetas <- c(0.2, 0.5, 0.7)
results_t <- lapply(thetas, function(t) metrics_for_threshold(prob, diabetes$Outcome, t))

summary_df <- do.call(rbind, lapply(results_t, function(x) {
  data.frame(theta = x$threshold,
             Accuracy = as.numeric(x$accuracy),
             Precision = as.numeric(x$precision),
             Sensitivity = as.numeric(x$sensitivity))
}))
print(summary_df)

for (res in results_t) {
  cat("\n\n==== Threshold: ", res$threshold, " ====\n")
  print(res$confusion)
  print(res$cm_obj$byClass[c("Sensitivity","Specificity","Pos Pred Value","Neg Pred Value")])
}


# 1.2 ROC Curve

# ROC & AUC using pROC
roc_obj <- roc(response = diabetes$Outcome, predictor = prob, levels = c("0","1"), direction = "<")
plot(roc_obj, main = "ROC curve: Logistic regression (full model)")
auc_val <- auc(roc_obj)
cat("AUC:", auc_val, "\n")

# Choose best threshold by Youden's index
coords_df <- coords(roc_obj, "best", best.method = "youden", ret = c("threshold","sensitivity","specificity","accuracy"), transpose = FALSE)
print(coords_df)
# coords_df$threshold is the chosen threshold
best_threshold <- coords_df[["threshold"]]
cat("Best threshold by Youden index:", best_threshold, "\n")


# 1.3 LQA QDA Naive Bayes

# prepare predictors/response
X <- diabetes[, setdiff(names(diabetes), "Outcome")]
Y <- diabetes$Outcome

# LDA
lda.fit <- lda(Outcome ~ ., data = diabetes)
lda.pred <- predict(lda.fit, diabetes)$class
cm.lda <- confusionMatrix(lda.pred, Y, positive = "1")
print(cm.lda)

# QDA
# QDA can fail if covariance singular; wrap in tryCatch
qda.fit <- tryCatch(qda(Outcome ~ ., data = diabetes), error = function(e) e)
if (inherits(qda.fit, "error")) {
  cat("QDA failed:", qda.fit, "\n")
  cm.qda <- NULL
} else {
  qda.pred <- predict(qda.fit, diabetes)$class
  cm.qda <- confusionMatrix(qda.pred, Y, positive = "1")
  print(cm.qda)
}

# Naive Bayes
nb.fit <- naiveBayes(Outcome ~ ., data = diabetes)
nb.pred <- predict(nb.fit, diabetes)
cm.nb <- confusionMatrix(nb.pred, Y, positive = "1")
print(cm.nb)

# For logistic classifier, confusion matrix at theta=0.5
log.pred.05 <- factor(ifelse(prob >= 0.5, "1", "0"), levels = levels(Y))
cm.log05 <- confusionMatrix(log.pred.05, Y, positive = "1")
print(cm.log05)

# Make a small summary table (Accuracy / Precision / Sensitivity)
models <- list(Logistic = cm.log05, LDA = cm.lda, QDA = cm.qda, NaiveBayes = cm.nb)
summary_models <- do.call(rbind, lapply(names(models), function(nm) {
  cmobj <- models[[nm]]
  if (is.null(cmobj)) {
    return(data.frame(Model = nm, Accuracy = NA, Precision = NA, Sensitivity = NA))
  }
  data.frame(Model = nm,
             Accuracy = as.numeric(cmobj$overall["Accuracy"]),
             Precision = as.numeric(cmobj$byClass["Pos Pred Value"]),
             Sensitivity = as.numeric(cmobj$byClass["Sensitivity"]))
}))
print(summary_models)


# 1.4 Choose the classifier that gives the best balance of sensitivity/precision depending on your goal. If the priority is minimizing missed diabetics (maximize sensitivity), pick the classifier with highest sensitivity. If minimizing false positives is priority, pick highest precision. Based on the summary table above, choose the model and justify.

# Discriminant analysis (LDA/QDA) can be used because predictors are numeric (continuous) and LDA/QDA assume multivariate normal predictors within classes and equal (LDA) or class-specific (QDA) covariance matrices. The diabetes dataset predictors are continuous measurements, making discriminant analysis appropriate (subject to checking assumptions).

# 2 MODEL SELECTIon
#--------------------------------------------------------------

# Define the three models
model1 <- glm(Outcome ~ BMI, data = diabetes, family = binomial)
model2 <- glm(Outcome ~ BMI + Glucose, data = diabetes, family = binomial)
model3 <- glm(Outcome ~ BMI + Glucose + SkinThickness, data = diabetes, family = binomial)

# LOOCV using cv.glm (default is leave-one-out when K not given or K = n)
set.seed(123) 
loocv1 <- cv.glm(diabetes, model1, K = nrow(diabetes))
loocv2 <- cv.glm(diabetes, model2, K = nrow(diabetes))
loocv3 <- cv.glm(diabetes, model3, K = nrow(diabetes))

loocv_res <- data.frame(
  Model = c("M1_BMI", "M2_BMI+glucose", "M3_BMI+glucose+Skin"),
  LOOCV_delta = c(loocv1$delta[1], loocv2$delta[1], loocv3$delta[1])  # raw estimated CV error
)
print(loocv_res)

# Which has smallest LOOCV error?
loocv_res[which.min(loocv_res$LOOCV_delta), ]

# Print chosen model coefficients
best_loocv_model <- list(model1, model2, model3)[[which.min(loocv_res$LOOCV_delta)]]
summary(best_loocv_model)
coef(best_loocv_model)


#2.1 

# 5-fold CV using cv.glm with K=5 (this randomizes partitioning)
set.seed(123)
cv5_1 <- cv.glm(diabetes, model1, K = 5)
cv5_2 <- cv.glm(diabetes, model2, K = 5)
cv5_3 <- cv.glm(diabetes, model3, K = 5)

cv5_res <- data.frame(
  Model = c("M1_BMI", "M2_BMI+glucose", "M3_BMI+glucose+Skin"),
  CV5_delta = c(cv5_1$delta[1], cv5_2$delta[1], cv5_3$delta[1])
)
print(cv5_res)
cv5_res[which.min(cv5_res$CV5_delta), ]

# Print final chosen model coefficients for 5-fold best
best_cv5_model <- list(model1, model2, model3)[[which.min(cv5_res$CV5_delta)]]
summary(best_cv5_model)
coef(best_cv5_model)


#2.2 

# AIC comparison
aic_res <- data.frame(
  Model = c("M1_BMI", "M2_BMI+glucose", "M3_BMI+glucose+Skin"),
  AIC = c(AIC(model1), AIC(model2), AIC(model3))
)
print(aic_res)
aic_res[which.min(aic_res$AIC), ]

# Print coefficients of model with smallest AIC
best_aic_model <- list(model1, model2, model3)[[which.min(aic_res$AIC)]]
summary(best_aic_model)
coef(best_aic_model)


# 3 MEDICAL COST DATASET

medicost <- medical_cost[, -1]  # drop id
str(medicost)

# Response: charges; predictors: the other 5 variables (confirm there are 5 predictors)
names(medicost)
# Best subset selection using regsubsets
regfit.full <- regsubsets(charges ~ ., data = medicost, nvmax = 5, method = "exhaustive")
reg.summary <- summary(regfit.full)
reg.summary

# 3.1 Plot RSS, Adjusted R2, Cp, BIC and highlight best models
par(mfrow = c(2,2))
# RSS
plot(reg.summary$rss, xlab = "Number of predictors", ylab = "RSS", type = "b",
     main = "RSS vs #predictors")
# adjusted R^2
plot(reg.summary$adjr2, xlab = "Number of predictors", ylab = "Adjusted R2", type = "b",
     main = "Adjusted R2 vs #predictors")
best_adjr2 <- which.max(reg.summary$adjr2)
points(best_adjr2, reg.summary$adjr2[best_adjr2], col = "red", pch = 19)

# Cp
plot(reg.summary$cp, xlab = "Number of predictors", ylab = "Cp", type = "b",
     main = "Cp vs #predictors")
best_cp <- which.min(reg.summary$cp)
points(best_cp, reg.summary$cp[best_cp], col = "red", pch = 19)

# BIC
plot(reg.summary$bic, xlab = "Number of predictors", ylab = "BIC", type = "b",
     main = "BIC vs #predictors")
best_bic <- which.min(reg.summary$bic)
points(best_bic, reg.summary$bic[best_bic], col = "red", pch = 19)

# Show indices
cat("best by adjR2:", best_adjr2, "  best by Cp:", best_cp, "  best by BIC:", best_bic, "\n")


# Fit the 4-predictor model
best_model4 <- lm(charges ~ age + bmi + children + smokeryes, data = medicost)

# Print the coefficients
coef(best_model4)


# 3.1 

# 3.2 Print coefficients of the overall best model per chosen criterion
# Suppose we choose the model chosen by BIC (common)
best_model_bic <- regsubsets(charges ~ ., data = medicost, nvmax = 5)
# get coefficients for model of size best_bic
coefs_bic <- coef(regfit.full, best_bic)
print(coefs_bic)


# 3.3 Forward stepwise selection
regfit.fwd <- regsubsets(charges ~ ., data = medicost, nvmax = 5, method = "forward")
reg.fwd.sum <- summary(regfit.fwd)

par(mfrow = c(2,2))
plot(reg.fwd.sum$rss, type = "b", xlab = "Number predictors", ylab = "RSS", main = "RSS (forward)")
plot(reg.fwd.sum$adjr2, type = "b", xlab = "Number predictors", ylab = "Adj R2", main = "Adj R2 (forward)")
best_adjr2_fwd <- which.max(reg.fwd.sum$adjr2)
points(best_adjr2_fwd, reg.fwd.sum$adjr2[best_adjr2_fwd], col = "red", pch = 19)
plot(reg.fwd.sum$cp, type = "b", xlab = "Number predictors", ylab = "Cp", main = "Cp (forward)")
best_cp_fwd <- which.min(reg.fwd.sum$cp)
points(best_cp_fwd, reg.fwd.sum$cp[best_cp_fwd], col = "red", pch = 19)
plot(reg.fwd.sum$bic, type = "b", xlab = "Number predictors", ylab = "BIC", main = "BIC (forward)")
best_bic_fwd <- which.min(reg.fwd.sum$bic)
points(best_bic_fwd, reg.fwd.sum$bic[best_bic_fwd], col = "red", pch = 19)

cat("Forward best by adjR2:", best_adjr2_fwd, 
    " Cp:", best_cp_fwd, " BIC:", best_bic_fwd, "\n")

# Compare coefficients for the BIC-selected model from exhaustive and forward
cat("Coefficients (exhaustive, size = best_bic):\n")
print(coef(regfit.full, best_bic))

cat("Coefficients (forward, size = best_bic_fwd):\n")
print(coef(regfit.fwd, best_bic_fwd))

# 4 Bootstrap for X Values

# 4: read Xvalues
xvalues <- Xvalues
x <- as.numeric(xvalues$V1)

# 4.1 Bootstrap estimate of mean and bootstrap SE using R = 100
library(boot)
stat_mean <- function(data, indices) {
  d <- data[indices]
  return(mean(d))
}
set.seed(123)
boot_out <- boot(data = x, statistic = stat_mean, R = 100)
print(boot_out)
# bootstrap standard error:
boot_se <- sd(boot_out$t)
cat("Bootstrap standard error (R=100):", boot_se, "\n")

# 4.2 95% CI using boot.ci (use percentile or basic)
boot_ci <- boot.ci(boot_out, type = c("perc", "basic", "bca"))
print(boot_ci)

# 4.3 Why bootstrap is not ideal for estimating testing error:
# We'll provide explanation as text below — code not required.
