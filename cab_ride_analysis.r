
cab_rides <- read.csv("cab_rides.csv")
weather <- read.csv("weather.csv")

# Preprocessing
# cab_rides has 693,017 observations
# Remove 55,095 where Price is NA
# 637,976 observations remaining
cab_rides <- cab_rides[!is.na(cab_rides$price),]

# Convert 13 digit time_stamp (milliseconds) to POSIX datetime
cab_rides$time_stamp <- as.POSIXct(cab_rides$time_stamp/1000, origin = "1970-01-01")

# Convert 10 digit time_stamp (seconds) to POSIX datetime
weather$time_stamp <- as.POSIXct(weather$time_stamp, origin = "1970-01-01")

# Join cab_rides to weather on cab_rides.source = weather.location
# Also want to join on time_stamp, but they are down to seconds, so will not match exactly
# Convert time stamps to the nearest hour
# For weather, take the average of all other columns over the hour
cab_rides$location <- cab_rides$source
cab_rides$rounded_timestamp <- format(round(cab_rides$time_stamp, units = "hours"))
weather$rounded_timestamp <- format(round(weather$time_stamp, units = "hours"))

weather <- aggregate(list(weather$temp, 
                          weather$clouds, 
                          weather$pressure, 
                          weather$rain,
                          weather$humidity, 
                          weather$wind),
          list(weather$location, weather$rounded_timestamp), 
          FUN=mean) 
colnames(weather) <- c("location", "rounded_timestamp", "temp", 
                       "clouds", "pressure","rain", "humidity", "wind")

cab_rides_new <- merge(cab_rides, weather, by = c("location", "rounded_timestamp"), all.x = T)

# Check correlation between columns
# cor(cab_rides_new[,sapply(cab_rides_new,is.numeric)],use = "complete.obs", method = "pearson")

# Changing a few more variables
# time_stamp is too granular (yyyy-mm-dd hh:mm:ss)
# Instead, extract time_of_day (hour) and day_of_week
# Remove id column
# product_id and name represent the same thing, only keep name
# if rain is NA, make it 0
library(lubridate)
cab_rides_new$day_of_week <- wday(cab_rides_new$time_stamp)
cab_rides_new$time_of_day <- hour(cab_rides_new$time_stamp)
cab_rides_new <- cab_rides_new[,!(names(cab_rides_new) %in% c(
                                                      "location", 
                                                      "rounded_timestamp",
                                                      "id", 
                                                      "product_id",
                                                      "time_stamp"))]
cab_rides_new$rain <- ifelse(is.na(cab_rides_new$rain), 0, cab_rides_new$rain)

# Histogram of price
hist(cab_rides_new$price, 
     main = "# of Rides by Price",
     xlab = "Price",
     ylab = "# of Rides"
     )
abline(v = mean(cab_rides$price), col = "red")
abline(v = median(cab_rides$price), col = "blue")

# Take a sample of 2.5% of the data for scatter/box plots
cab_rides_subset <- cab_rides_new[sample(nrow(cab_rides_new), size = nrow(cab_rides_new) * 0.025),]

par(mfrow = c(3, 2))
plot(cab_rides_subset$distance, cab_rides_subset$price)
boxplot(price ~ cab_type, data = cab_rides_subset)
boxplot(price ~ destination, data = cab_rides_subset)
boxplot(price ~ source, data = cab_rides_subset)
boxplot(price ~ surge_multiplier, data = cab_rides_subset)
boxplot(price ~ name, data = cab_rides_subset)

par(mfrow = c(3, 2))
plot(cab_rides_subset$temp, cab_rides_subset$price)
plot(cab_rides_subset$clouds, cab_rides_subset$price)
plot(cab_rides_subset$pressure, cab_rides_subset$price)
plot(cab_rides_subset$rain, cab_rides_subset$price)
plot(cab_rides_subset$humidity, cab_rides_subset$price)
plot(cab_rides_subset$wind, cab_rides_subset$price)

par(mfrow = c(2,1))
boxplot(price ~ day_of_week, data = cab_rides_subset)
boxplot(price ~ time_of_day, data = cab_rides_subset)

# Notable relationships:
# distance, source, destination, surge_multiplier, and name seem to impact price
# cab_type, day_of_week, time_of_day, and the weather predictors do not seem important

# Maybe we can also predict demand (# of rides) and the surge_multiplier

# Split into training and test set
cab_rides_train_idx <- sample(1:nrow(cab_rides_new), size = nrow(cab_rides_new) * 0.70)
cab_rides_train <- cab_rides_new[cab_rides_train_idx, ]
cab_rides_test <- cab_rides_new[-cab_rides_train_idx, ]

# Linear regression with all variables
cab_rides_lm <- lm(price ~ ., data = cab_rides_train)
summary(cab_rides_lm)

# Training MSE ~ 6.22
cab_rides_lm_train <- data.frame(prediction = predict(cab_rides_lm, cab_rides_train), 
                                   actual = cab_rides_train$price)
cab_rides_lm_train <- cab_rides_lm_train[!is.na(cab_rides_lm_train$prediction),]
mean((cab_rides_lm_train$actual - cab_rides_lm_train$prediction)^2)

# Test MSE ~ 6.15
cab_rides_lm_test <- data.frame(prediction = predict(cab_rides_lm, cab_rides_test), 
                                 actual = cab_rides_test$price)
cab_rides_lm_test <- cab_rides_lm_test[!is.na(cab_rides_lm_test$prediction),]
mean((cab_rides_lm_test$actual - cab_rides_lm_test$prediction)^2)

# Some variables are not important
# Use Lasso to try to remove these
library(glmnet)

x <- model.matrix(price ~ ., cab_rides_train)
y <- cab_rides_train[complete.cases(cab_rides_train), ]$price

# 10-fold cross-validation to find optimal lambda value
cv_model <- cv.glmnet(x, y, alpha = 1, kfolds = 10)

# Find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda

# Plot MSE by Lambda
plot(cv_model)

############################
best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
coef(best_model)

t <- cab_rides_test[complete.cases(cab_rides_test), ]
test <- model.matrix(price ~., t)
p <- predict(best_model, s = best_lambda, newx = test)
a <- t$price

# Lasso did not reduce the MSE by very much
mean((p - a)^2)