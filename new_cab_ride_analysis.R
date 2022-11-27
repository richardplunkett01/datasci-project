
set.seed(123)
cab_rides <- read.csv("rideshare_kaggle.csv")

# Check correlation between columns
cor <- data.frame(cor(cab_rides[,sapply(cab_rides,is.numeric)],use = "complete.obs", method = "pearson"))

# Remove highly correlated columns
cab_rides <- cab_rides[,!(names(cab_rides) %in% c(
  "id",
  "timestamp",
  "latitude",
  "longitude",
  "apparentTemperature",
  "precipProbability",
  "windGust",
  "windGustTime",
  "temperatureHigh",
  "temperatureHighTime",
  "temperatureLow",
  "temperatureLowTime",
  "apparentTemperatureHigh",
  "apparentTemperatureHighTime",
  "apparentTemperatureLow",
  "apparentTemperatureLowTime",
  "dewPoint",
  "visibility.1",
  "sunriseTime",
  "sunsetTime",
  "uvIndexTime",
  "temperatureMin",
  "temperatureMinTime",
  "temperatureMax",
  "temperatureMaxTime",
  "apparentTemperatureMin",
  "apparentTemperatureMinTime",
  "apparentTemperatureMax",
  "apparentTemperatureMaxTime",
  # Also removed
  "datetime",
  "timezone",
  "product_id"))]

# Remove any rows with NAs (55,041 removed)
cab_rides <- cab_rides[complete.cases(cab_rides),]

# Check for correlation among remaining variables
cor(cab_rides[,sapply(cab_rides,is.numeric)],use = "complete.obs", method = "pearson")

# Histogram of price
hist(cab_rides$price, 
     main = "# of Rides by Price",
     xlab = "Price",
     ylab = "# of Rides"
)
abline(v = mean(cab_rides$price), col = "red")
abline(v = median(cab_rides$price), col = "blue")

# Split into Training and Test sets
cab_rides_train_idx <- sample(1:nrow(cab_rides), size = nrow(cab_rides) * 0.70)
cab_rides_train <- cab_rides[cab_rides_train_idx, ]
cab_rides_test <- cab_rides[-cab_rides_train_idx, ]

# Linear Regression with all variables 
cab_rides_lm <- lm(price ~ ., data = cab_rides_train)
summary(cab_rides_lm)

# Several variables have a low p value and should be removed
# Only the following are significant: source, destination, cab_type, name, distance, surge_multiplier, short_summary
get_mse <- function(actual, predicted) {mean((actual - predicted)^2)}


# LM Training MSE = 6.221184
predicted <- predict(cab_rides_lm, cab_rides_train)
get_mse(cab_rides_train$price, predicted)

# LM Test MSE = 6.173095
predicted <- predict(cab_rides_lm, cab_rides_test)
get_mse(cab_rides_test$price, predicted)

# Redo with only important variables
cab_rides_lm <- lm(price ~ source + destination + cab_type + name + distance + surge_multiplier + short_summary, 
                     data = cab_rides_train)
summary(cab_rides_lm)

# LM Training MSE = 6.221564
predicted <- predict(cab_rides_lm, cab_rides_train)
get_mse(cab_rides_train$price, predicted)

# LM Test MSE = 6.172724
predicted <- predict(cab_rides_lm, cab_rides_test)
get_mse(cab_rides_test$price, predicted)
# Same accuracy with fewer variables

# Regression Tree
library(tree)

cab_rides_tree <- tree(price ~ ., data = cab_rides_train)
plot(cab_rides_tree)
text(cab_rides_tree, pretty = 0)

# Tree Training MSE = 73.90861 (oversimplified model)
get_mse(cab_rides_train$price, predict(cab_rides_tree, cab_rides_train))

# Bagging
library(randomForest)

# Very slow - reduced to one tree
cab_rides_bag <- randomForest(price ~ source + destination + cab_type + name + distance + surge_multiplier + short_summary, 
                        data = cab_rides_train,
                        importance = TRUE, 
                        ntree = 1, 
                        mtry = 7)

# Bagging Training MSE = 2.50273
get_mse(cab_rides_train$price, predict(cab_rides_bag, cab_rides_train))

# Bagging Test MSE = 3.372221
get_mse(cab_rides_test$price, predict(cab_rides_bag, cab_rides_test))

# Random Forest
cab_rides_rf <- randomForest(price ~ source + destination + cab_type + name + distance + surge_multiplier + short_summary, 
                              data = cab_rides_train,
                              importance = TRUE, 
                              ntree = 10, 
                              mtry = 2)

# Bagging Training MSE = 4.01846
get_mse(cab_rides_train$price, predict(cab_rides_rf, cab_rides_train))

# Bagging Test MSE = 4.150062
get_mse(cab_rides_test$price, predict(cab_rides_rf, cab_rides_test))

# SVM - better with small datasets with many predictors
# Taking a 5% sample from the training data
library(e1071)

# Using a radial kernel
cab_rides_svm <- svm(price ~ source + destination + cab_type + name + distance + surge_multiplier + short_summary
                     ,cab_rides_train[sample(1:nrow(cab_rides_train), size = nrow(cab_rides_train) * 0.05),]
                     ,kernel = "radial")

# SVM Training MSE = 5.743677
get_mse(cab_rides_train$price, predict(cab_rides_svm, cab_rides_train))

# SVM Test MSE = 5.727293
get_mse(cab_rides_test$price, predict(cab_rides_svm, cab_rides_test))

# Using a linear kernel
cab_rides_svm <- svm(price ~ source + destination + cab_type + name + distance + surge_multiplier + short_summary
                     ,cab_rides_train[sample(1:nrow(cab_rides_train), size = nrow(cab_rides_train) * 0.05),]
                     ,kernel = "linear")

# SVM Training MSE = 6.384661
get_mse(cab_rides_train$price, predict(cab_rides_svm, cab_rides_train))

# SVM Test MSE = 6.329907
get_mse(cab_rides_test$price, predict(cab_rides_svm, cab_rides_test))

# Light GBM
library(lightgbm)

train_x = as.matrix(cab_rides_train[c("price","source","destination","cab_type","name","distance","surge_multiplier","short_summary")])
train_y = as.matrix(cab_rides_train$price)

test_x = as.matrix(cab_rides_test[c("price","source","destination","cab_type","name","distance","surge_multiplier","short_summary")])
test_y = as.matrix(cab_rides_test$price)

dtrain = lgb.Dataset(train_x, label = train_y)
dtest = lgb.Dataset.create.valid(dtrain, test_x, label = test_y)

# define parameters
params = list(
  objective = "regression"
  , metric = "l2"
  , min_data = 1L
  , learning_rate = .3
)

# validataion data
valids = list(test = dtest)

# train model 
model = lgb.train(
  params = params
  , data = dtrain
  , nrounds = 5L
  , valids = valids
)

# LightGBM Training Error = 2.474381
pred_y = predict(model, train_x)
get_mse(train_y, pred_y)

# LightGBM Test Error = 2.469823
pred_y = predict(model, test_x)
get_mse(test_y, pred_y)

# Ridge Regression
library(glmnet)

cab_ride_model_matrix <- model.matrix(price ~ ., cab_rides_train)[, -1]
cab_rides_price <- cab_rides_train$price

cab_ridges_ridge <- glmnet(cab_ride_model_matrix, cab_rides_price, alpha = 0)
cab_rides_cv_ridge <- cv.glmnet(cab_ride_model_matrix, cab_rides_price, alpha = 0)
plot(cab_rides_cv_ridge)

cab_rides_cv_ridge_best_l <- cab_rides_cv_ridge$lambda.min

cab_rides_ridge_pred <- predict(cab_rides_cv_ridge, 
                      s=cab_rides_cv_ridge_best_l, 
                      newx= cab_ride_model_matrix)

# Ridge training MSE = 6.532637
get_mse(cab_rides_train$price, cab_rides_ridge_pred)

# Ridge test MSE = 6.482745
get_mse(cab_rides_test$price, 
        predict(cab_rides_cv_ridge, 
                s = cab_rides_cv_ridge_best_l, 
                newx = model.matrix(price ~ ., cab_rides_test)[, -1]))

# LASSO
x <- model.matrix(price ~ ., cab_rides_train)[, -1]
y <- cab_rides_train$price
lasso.fit <- glmnet(x, y, alpha=1)
cv.lasso.fit <- cv.glmnet(x, y, alpha=1)
plot(cv.lasso.fit)

bestlam.lasso <- cv.lasso.fit$lambda.min
lasso.pred <- predict(lasso.fit, s=bestlam.lasso, newx=x)

# LASSO Training MSE = 6.225786
get_mse(cab_rides_train$price, lasso.pred)

# LASSO Test MSE = 6.175447
x <- model.matrix(price ~ ., cab_rides_test)[, -1]
lasso.pred <- predict(lasso.fit, s=bestlam.lasso, newx=x)
get_mse(cab_rides_test$price, lasso.pred)

# Redo LASSO with fewer predictors
x <- model.matrix(price ~ source + destination + cab_type + name + distance + surge_multiplier + short_summary, cab_rides_train)[, -1]
y <- cab_rides_train$price
lasso.fit <- glmnet(x, y, alpha=1)
cv.lasso.fit <- cv.glmnet(x, y, alpha=1)
plot(cv.lasso.fit)

bestlam.lasso <- cv.lasso.fit$lambda.min
lasso.pred <- predict(lasso.fit, s=bestlam.lasso, newx=x)

# LASSO Training MSE = 6.225813
get_mse(cab_rides_train$price, lasso.pred)

# LASSO Test MSE = 6.175407
x <- model.matrix(price ~ source + destination + cab_type + name + distance + surge_multiplier + short_summary, cab_rides_test)[, -1]
lasso.pred <- predict(lasso.fit, s=bestlam.lasso, newx=x)
get_mse(cab_rides_test$price, lasso.pred)

# PCR
library(pls)
pcr.fit <- pcr(price ~ source + destination + cab_type + name + distance + surge_multiplier + short_summary, 
               data = cab_rides, 
               subset = cab_rides_train_idx, 
               scale = TRUE, 
               validation ="CV")
validationplot(pcr.fit, val.type = "MSEP")

x <- model.matrix(price ~  source + destination + cab_type + name + distance + surge_multiplier + short_summary, cab_rides_train)[, -1]
y <- cab_rides_train$price

pcr.pred <- predict(pcr.fit, x, ncomp = 44)

# PCR Training MSE = 6.221574
get_mse(y, pcr.pred)

# PCR Test MSE = 6.172625
x <- model.matrix(price ~  source + destination + cab_type + name + distance + surge_multiplier + short_summary, cab_rides_test)[, -1]
get_mse(cab_rides_test$price, pcr.pred <- predict(pcr.fit, x, ncomp = 44))