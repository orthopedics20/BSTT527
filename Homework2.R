# PROBLEM 1 #
#-------------#

# Load prostate dataset
prostate <- prostate.data

# 1. Create X and Y
X <- prostate[, 1:8]
Y <- prostate$lpsa

# 2. Correlation matrix
cor_matrix <- cor(X)
print(cor_matrix)

# 3. Multiple regression
full <- lm(lpsa ~ ., data = prostate)
summary(full)

mse <- summary(full)$sigma^2
mse


# Diagnostic plots
par(mfrow = c(2, 2)) # sets up 2x2 plotting area
plot(full)           # produces residuals vs fitted, QQ plot, scale-location, Cook’s distance
par(mfrow = c(1, 1)) # reset


# PROBLEM 2 #
#-------------#

# Load SAheart dataset
SAheart <- SAheart.data

# 1. Numerical + graphical summaries
summary(SAheart)
hist(SAheart$age, main="Histogram of Age", xlab="Age (years)")
hist(SAheart$tobacco, main="Histogram of Tobacco", xlab="Tobacco (kg)")
hist(SAheart$alcohol, main="Histogram of Alcohol", xlab="Current Alcohol Consumption")
boxplot(SAheart$ldl ~ SAheart$chd, main="LDL by CHD Status", xlab="CHD (0=No, 1=Yes)", ylab="LDL")

# 2. Logistic regression
full_logit <- glm(chd ~ ., data = SAheart, family = binomial)
summary(full_logit)

# Predicted probabilities
probs <- predict(full_logit, type = "response")

# Predicted class (threshold 0.5)
pred_class <- ifelse(probs >= 0.5, 1, 0)

# Actual class
actual <- SAheart$chd

# Confusion matrix
table(predicted = pred_class, actual = actual)

# Precision, Recall, F1 using caret
library(caret)
conf_mat <- confusionMatrix(as.factor(pred_class), as.factor(actual), positive = "1")
conf_mat$byClass[c("Precision", "Recall", "F1")]


# 3. Model adequacy check
# Pseudo R² (McFadden)
library(pscl)
pR2(full_logit)

# Hosmer-Lemeshow test
library(ResourceSelection)
hoslem.test(SAheart$chd, fitted(full_logit))

# ROC curve
library(pROC)
roc_curve <- roc(SAheart$chd, fitted(full_logit))
plot(roc_curve, main="ROC Curve for Logistic Regression")
auc(roc_curve)


# Problem 3
# ------------------

# Construct 2x2 table
table <- matrix(c(22, 378, 8, 592), nrow=2, byrow=TRUE)
colnames(table) <- c("Diseased", "Healthy")
rownames(table) <- c("Exposed", "Not_Exposed")
table <- as.table(table)
table

# 1. Odds and odds ratio
odds_exposed <- 22 / 378
odds_not_exposed <- 8 / 592
odds_ratio <- odds_exposed / odds_not_exposed
odds_exposed; odds_not_exposed; odds_ratio

# 2. Hypothesis test (Chi-square or Fisher’s exact)
chisq.test(table)
fisher.test(table)

# 3. Confidence interval for odds ratio
library(epitools)
oddsratio(table, method="wald")

