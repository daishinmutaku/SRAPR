x <- log(cars$speed)
y <- log(cars$dist)
result <- lm(y~x)
result