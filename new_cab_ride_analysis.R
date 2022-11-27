
set.seed(123)
cab_rides <- read.csv("rideshare_kaggle.csv")


# Check correlation between columns
cor <- cor(cab_rides[,sapply(cab_rides,is.numeric)],use = "complete.obs", method = "pearson")

# Remove highly correlated columns


# Exploratory Analysis
head(cab_rides, 5)
summary(cab_rides)

# Some variables are not useful for our purpose and can be removed
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
  "datetime",
  "timezone"))]

# Remove any rows with NAs
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
# Training MSE = 6.22
cab_rides_lm_train <- data.frame(prediction = predict(cab_rides_lm, cab_rides_train), 
                                 actual = cab_rides_train$price)
cab_rides_lm_train <- cab_rides_lm_train[!is.na(cab_rides_lm_train$prediction),]
mean((cab_rides_lm_train$actual - cab_rides_lm_train$prediction)^2)

# Test MSE = 6.17
cab_rides_lm_test <- data.frame(prediction = predict(cab_rides_lm, cab_rides_test), 
                                actual = cab_rides_test$price)
cab_rides_lm_test <- cab_rides_lm_test[!is.na(cab_rides_lm_test$prediction),]
mean((cab_rides_lm_test$actual - cab_rides_lm_test$prediction)^2)

# Redo with only important variables
cab_rides_lm_2 <- lm(price ~ source + destination + cab_type + name + distance + surge_multiplier, 
                     data = cab_rides_train)
summary(cab_rides_lm_2)

# Training MSE = 6.22
cab_rides_lm_train <- data.frame(prediction = predict(cab_rides_lm_2, cab_rides_train), 
                                 actual = cab_rides_train$price)
cab_rides_lm_train <- cab_rides_lm_train[!is.na(cab_rides_lm_train$prediction),]
mean((cab_rides_lm_train$actual - cab_rides_lm_train$prediction)^2)

# Test MSE = 6.17
cab_rides_lm_test <- data.frame(prediction = predict(cab_rides_lm_2, cab_rides_test), 
                                actual = cab_rides_test$price)
cab_rides_lm_test <- cab_rides_lm_test[!is.na(cab_rides_lm_test$prediction),]
mean((cab_rides_lm_test$actual - cab_rides_lm_test$prediction)^2)
# Same accuracy with fewer variables

# Try to reduce dimensionality
# Ridge
library(glmnet)

x <- model.matrix(price ~ ., cab_rides_train)[, -1]
y <- cab_rides_train$price

ridge.fit <- glmnet(x, y, alpha = 0)
cv.ridge.fit <- cv.glmnet(x, y, alpha = 0)
plot(cv.ridge.fit)

bestlam.ridge <- cv.ridge.fit$lambda.min

x <- model.matrix(price ~ ., cab_rides_test)[, -1]
y <- cab_rides_test$price

# MSE = 6.256 - no improvement
ridge.pred <- predict(ridge.fit, s=bestlam.ridge, newx=x)
ridge.error <- mean((ridge.pred-y)^2)
ridge.error

# LASSO
x <- model.matrix(price ~ ., cab_rides_train)[, -1]
y <- cab_rides_train$price
lasso.fit <- glmnet(x, y, alpha=1)
cv.lasso.fit <- cv.glmnet(x, y, alpha=1)
plot(cv.lasso.fit)

x <- model.matrix(price ~ ., cab_rides_test)[, -1]
y <- cab_rides_test$price

# MSE = 6.18 - no improvement
bestlam.lasso <- cv.lasso.fit$lambda.min
lasso.pred <- predict(lasso.fit, s=bestlam.lasso, newx=x)
lasso.error <- mean((lasso.pred-y)^2)
lasso.error

# PCR
library(pls)
# pcr.fit <- pcr(price ~ ., data = cab_rides, subset=cab_rides_train_idx, scale=TRUE, validation ="CV")
# validationplot(pcr.fit, val.type = "MSEP")

# x <- model.matrix(price ~ ., cab_rides_test)[, -1]
# y <- cab_rides_test$price

# pcr.pred <- predict(pcr.fit, x, ncomp = 13)
# pcr.error <- mean((pcr.pred - y)^2)
# pcr.error

# Regression Tree
library(tree)

tree_model <- tree(price ~ ., data = cab_rides_train)
plot(tree_model)
text(tree_model, pretty = 0)
tree_model
# Oversimplified, need bagging

# Random Forest very slow
library(randomForest)
bag.mod <- randomForest(price ~ ., data=cab_rides_train, importance = TRUE, ntree = 1, mtry=5)
plot(bag.mod)
text(bag.mod, pretty = 0)

# MSE 6.88 - no improvement
mean((predict(bag.mod, cab_rides_test) - cab_rides_test$price)^2)


hist(cab_rides$price)