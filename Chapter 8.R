# Ex. 8.1

library(mlbench)
set.seed(200)
simulated <- mlbench.friedman1(200, sd = 1)
simulated <- cbind(simulated$x, simulated$y)
simulated <- as.data.frame(simulated)
colnames(simulated)[ncol(simulated)] <- "y"

# a
library(randomForest)
library(caret)
model1 <- randomForest(y ~ ., data = simulated,
                       importance = T,
                       ntree = 1000)
rfImp1 <- varImp(model1, scale = F)
varImpPlot(model1)

# b

simulated$duplicate1 <- simulated$V1 + rnorm(200) * 0.1
cor(simulated$duplicate1, simulated$V1)

model2 <- randomForest(y ~ ., data = simulated,
                       importance = T,
                       ntree = 1000)
rfImp2 <- varImp(model2, scale = F)
varImpPlot(model2)

# Now importance of V1 is dulluted by duplicate1


