# 10.4
library(AppliedPredictiveModeling)
library(caret)
library(Hmisc)
library(plyr)
library(ggplot2)
library(magrittr)

data(concrete)

str(concrete)
str(mixtures)

featurePlot(x = concrete[, -9],
            y = concrete$CompressiveStrength,
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth"))

describe(concrete)

averaged <- ddply(mixtures,
                  .(Cement, BlastFurnaceSlag, FlyAsh, Water,
                    Superplasticizer, CoarseAggregate, FineAggregate, Age),
                  function(x) c(CompressiveStrength = mean(x$CompressiveStrength)))

# the same thing dplyr way
library(dplyr)

averaged2 <- mixtures %>% 
    group_by(Cement, BlastFurnaceSlag, FlyAsh, Water,Superplasticizer, CoarseAggregate, FineAggregate, Age) %>%
    summarise(CompressiveStrength = mean(CompressiveStrength))

set.seed(975)
forTraining <- createDataPartition(averaged$CompressiveStrength, p = 3/4)[[1]]
trainingSet <- averaged[forTraining, ]
testSet <- averaged[-forTraining, ]

modFormula <- paste("CompressiveStrength ~ (.)^2 + I(Cement^2) +",
                    "I(BlastFurnaceSlag^2) +I(FlyAsh^2) +I(Water^2) +I(Superplasticizer^2) +",
                    "I(CoarseAggregate^2) +I(FineAggregate^2) +I(Age^2)")
modFormula <- as.formula(modFormula)

controlObject <- trainControl(method = "repeatedcv",
                              repeats = 5,
                              number = 10)

set.seed(669)
linearReg <- train(modFormula,
                    data = trainingSet,
                    method = "lm",
                    trControl = controlObject)
linearReg
linearReg$finalModel

# Partial least squares
set.seed(669)
plsModel <- train(modFormula,
                  data = trainingSet,
                  method = "pls",
                  preProc = c("center", "scale"),
                  tuneLength = 15,
                  trControl = controlObject)

plsModel
plsModel$finalModel

# Elastic net
enetGrid <- expand.grid(.lambda = c(0, 0.001, 0.01, 0.1),
                        .fraction = seq(0.05, 1, length.out = 20))
set.seed(669)
enetModel <- train(modFormula,
                  data = trainingSet,
                  method = "enet",
                  preProc = c("center", "scale"),
                  tuneGrid = enetGrid,
                  trControl = controlObject)

enetModel
enetModel$finalModel

# MARS
set.seed(669)
earthModel <- train(CompressiveStrength ~ .,
                   data = trainingSet,
                   method = "earth",
                   tuneGrid = expand.grid(.degree = 1,
                                          .nprune = 2:25),
                   trControl = controlObject)

earthModel
earthModel$finalModel

# SVM
set.seed(669)
svmModel <- train(CompressiveStrength ~ .,
                  data = trainingSet,
                  method = "svmRadial",
                  preProc = c("center", "scale"),
                  tuneLength = 15,
                  trControl = controlObject)

svmModel
svmModel$finalModel

# Neural network
nnetGrid <- expand.grid(.decay = c(0.001, 0.01, 0.1),
                        .size = seq(1, 27, by = 2),
                        .bag = F)

set.seed(669)
nnetModel <- train(CompressiveStrength ~ .,
                  data = trainingSet,
                  method = "avNNet",
                  preProc = c("center", "scale"),
                  tuneGrid = nnetGrid,
                  linout = T,
                  trace = F,
                  maxit = 1000,
                  trControl = controlObject)

nnetModel
nnetModel$finalModel

# CART
set.seed(669)
rpartModel <- train(CompressiveStrength ~ .,
                   data = trainingSet,
                   method = "rpart",
                   tuneLength = 30,
                   trControl = controlObject)

rpartModel
rpartModel$finalModel

# Conditional Inference Trees
set.seed(669)
ctreeModel <- train(CompressiveStrength ~ .,
                    data = trainingSet,
                    method = "ctree",
                    tuneLength = 10,
                    trControl = controlObject)

ctreeModel
ctreeModel$finalModel

# Model trees
set.seed(669)
mtModel <- train(CompressiveStrength ~ .,
                 data = trainingSet,
                 method = "M5",
                 trControl = controlObject)

mtModel
mtModel$finalModel

# Bagged trees
set.seed(669)
treebagModel <- train(CompressiveStrength ~ .,
                 data = trainingSet,
                 method = "treebag",
                 trControl = controlObject)

treebagModel
treebagModel$finalModel

# Random Forest
set.seed(669)
rfModel <- train(CompressiveStrength ~ .,
                 data = trainingSet,
                 method = "rf",
                 tuneLength = 10,
                 ntrees = 1000,
                 importance = T,
                 trControl = controlObject)

rfModel
rfModel$finalModel

# Gradient boosted trees
gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, by = 2),
                       .n.trees = seq(100, 1000, by = 50),
                       .shrinkage = c(0.01, 0.1),
                       .n.minobsinnode = c(5, 10, 20, 30))
set.seed(669)
gbmModel <- train(CompressiveStrength ~ .,
                 data = trainingSet,
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 verbose = F,
                 trControl = controlObject)

gbmModel
gbmModel$finalModel

# Cubist
cubistGrid <- expand.grid(.committees = c(1, 5, 10, 50, 75, 100),
                          .neighbors = c(0, 1, 3, 5, 7, 9))

set.seed(669)
cbModel <- train(CompressiveStrength ~ .,
                  data = trainingSet,
                  method = "cubist",
                  tuneGrid = cubistGrid,
                  trControl = controlObject)

cbModel
cbModel$finalModel

allResamples <- resamples(list("Linear Reg" = linearReg,
                          "PLS" = plsModel,
                          "Elastic Net" = enetModel,
                          "MARS" = earthModel,
                          "SVM" = svmModel,
                          "Neural Networks" = nnetModel,
                          "CART" = rpartModel,
                          "Cond Inf Trees" = ctreeModel,
                          "Bagged Tree" = treebagModel,
                          "Boosted Tree" = gbmModel,
                          "Random Forest" = rfModel,
                          "Cubist" = cbModel))
    
parallelplot(allResamples, metric = "RMSE")
parallelplot(allResamples, metric = "Rsquared")

nnetPredictions <- predict(nnetModel, testSet)
gbmPredictions <- predict(gbmModel, testSet)
cbPerdictions <- predict(cbModel, testSet)

age28Data <- subset(trainingSet, Age == 28)
# Remove Age and compressive strength, then center and scale predictors
pp1 <- preProcess(age28Data[, - (8:9)], c("center", "scale"))

scaledTrain <- predict(pp1, age28Data[,c(1:7)])
set.seed(91)
startMixture <- sample(1:nrow(age28Data), 1)
starters <- scaledTrain[startMixture, 1:7]

pool <- scaledTrain
index <- maxDissim(starters, pool, 14) # select additional 14 samples maximally dissimilar to the starter
startPoints <- c(startMixture, index)  # start mixture together with the 14 maximally dissimilar comprise starting points
                                       # for the search of maximum compressive strength mixtures
starters <- age28Data[startPoints, 1:7]

# Remove water
startingValues <- starters[, -4]

# Search models for maximum compression strenght

# The input to the function is a vector of six mixture proportions
# (argument 'x') and the model used for prediction ('mod')

modelPrediction <- function(x, mod) {
    # Check to make sure the mixture proportions are in the correct range
    
    if(x[1] < 0 | x[1] > 1) return(10^38)
    if(x[2] < 0 | x[2] > 1) return(10^38)
    if(x[3] < 0 | x[3] > 1) return(10^38)
    if(x[4] < 0 | x[4] > 1) return(10^38)
    if(x[5] < 0 | x[5] > 1) return(10^38)
    if(x[6] < 0 | x[6] > 1) return(10^38)
    
    # determine the proportion of water
    x <- c(x, 1 - sum(x))
    
    # check water range
    if(x[7] < 0.05) return(10^38)
    
    # Convert the vector to a data frame, assign names
    # and fix age at 28 days
    tmp <- as.data.frame(t(x))
    names(tmp) <- c('Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Superplasticizer',
                    'CoarseAggregate', 'FineAggregate', 'Water')
    tmp$Age <- 28
    # Get model prediction, square them to get back to the original units
    # then return the negative of the result
    -predict(mod, tmp)
}

cbResults <- startingValues
cbResults$Water <- NA
cbResults$Prediction <- NA

for(i in 1:nrow(cbResults)){
    results <- optim(unlist(cbResults[i, 1:6]),
                     modelPrediction,
                     method = "Nelder-Mead",
                     # use method = "SANN" for simulated annealing
                     control = list(maxit = 5000),
                     mod = cbModel)
    cbResults$Prediction <- -results$value
    cbResults[i, 1:6] <- results$par
}

cbResults$Water <- 1 - apply(cbResults[, 1:6], 1, sum)
cbResults <- cbResults[order(cbResults$Prediction), ][1:3,]
cbResults$Model <- "Cubist"

nnetResults <- startingValues
nnetResults$Water <- NA
nnetResults$Prediction <- NA

for(i in 1:nrow(nnetResults)){
    results <- optim(unlist(nnetResults[i, 1:6]),
                     modelPrediction,
                     method = "Nelder-Mead",
                     # use method = "SANN" for simulated annealing
                     control = list(maxit = 5000),
                     mod = nnetModel)
    nnetResults$Prediction <- -results$value
    nnetResults[i, 1:6] <- results$par
}

nnetResults$Water <- 1 - apply(nnetResults[, 1:6], 1, sum)
nnetResults <- nnetResults[order(nnetResults$Prediction), ][1:3,]
nnetResults$Model <- "NNet"

# Run PCA on the data at 28 days
pp2 <- preProcess(age28Data[, 1:7], "pca")
# Get the components for these mixtures
pca1 <- predict(pp2, age28Data[, 1:7])
pca1$Data <- "Training Set"
# Label data points used to start the searches
pca1$Data[startPoints] <- "Starting Values"

# Project new mixtures in the same way (making sure tp re-order columns
# to match the order of the age28Data object)
pca3 <- predict(pp2, cbResults[, names(age28Data[, 1:7])])
pca3$Data <- "Cubist"
pca4 <- predict(pp2, nnetResults[, names(age28Data[, 1:7])])
pca4$Data <- "Neural Network"
# Combine the data, determine the axes range and plot
pcaData <- rbind(pca1, pca3, pca4)
pcaData$Data <- factor(pcaData$Data, levels = c("Training Set", "Starting Values", "Cubist", "Neural Network"))

lim <- extendrange(pcaData[, 1:2])
xyplot(PC2 ~ PC1, data = pcaData, groups = Data,
       auto.key = list(columns = 2),
       xlim = lim, ylim = lim,
       type = c("g", "p"))

# and ggplot version

pcaData %>% 
    ggplot(aes(PC1, PC2, color = Data)) +
    geom_point(aes(shape = Data))
