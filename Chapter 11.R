library(AppliedPredictiveModeling)
library(caret)
library(klaR)
library(MASS)
library(pROC)
library(randomForest)

set.seed(975)
simulatedTrain <- quadBoundaryFunc(500)
simulatedTest <- quadBoundaryFunc(1000)
head(simulatedTrain)

rfModel <- randomForest(class ~ X1 + X2,
                        data = simulatedTrain,
                        ntree = 2000)

qdaModel <- qda(class ~ X1 + X2,
                data = simulatedTrain)

qdaTrainPred <- predict(qdaModel, simulatedTrain)
names(qdaTrainPred)
head(qdaTrainPred$posterior)

qdaTestPred <- predict(qdaModel, simulatedTest)
simulatedTrain$QDAprob <- qdaTrainPred$posterior[, "Class1"]
simulatedTest$QDAprob <- qdaTestPred$posterior[, "Class1"]

rfTestPred <- predict(rfModel, simulatedTest, type = "prob")
head(rfTestPred)
simulatedTest$RFprob <- rfTestPred[, "Class1"]
simulatedTest$RFclass <- predict(rfModel, simulatedTest)

# Sensitivity and specificity
#
# Class1 will be used as the event of interest
sensitivity(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            positive = "Class1")

specificity(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            negative = "Class2")

posPredValue(data = simulatedTest$RFclass,
             reference = simulatedTest$class,
             positive = "Class1")

negPredValue(data = simulatedTest$RFclass,
             reference = simulatedTest$class,
             negative = "Class2")

# Change the prevalence manually
posPredValue(data = simulatedTest$RFclass,
             reference = simulatedTest$class,
             positive = "Class1",
             prevalence = .9)

# Confusion Matrix
#
confusionMatrix(data = simulatedTest$RFclass,
                reference = simulatedTest$class,
                positive = "Class1")
# ROC
#
rocCurve <- roc(response = simulatedTest$class,
                predictor = simulatedTest$RFprob,
                # This function assumes that the second class
                # is the event of interest, so we must
                # reverse lables
                levels = rev(levels(simulatedTest$class)))

auc(rocCurve)
ci(rocCurve)    # not ci.roc amymore

plot(rocCurve, legacy.axes = T)

# Lift charts
#
labs <- c(RFprob = "Random Forest",
          QDAprob = "Quadratic Discriminant Analysis")
liftCurve <- lift(class ~ RFprob + QDAprob, data = simulatedTest,
                  labels = labs)
liftCurve

xyplot(liftCurve,
       auto.key = list(columns = 2,
                       lines = T,
                       points = F))

# Calibrating probabilities
#
calCurve <- calibration(class ~ RFprob + QDAprob, data = simulatedTest)
calCurve

xyplot(calCurve, auto.key = list(columns = 2))

## The glm() function models the probability of the second factor
## level, so the function relevel() is used to temporarily reverse
## the factors levels
##
sigmoidalCal <- glm(relevel(class, ref = "Class2") ~ QDAprob,
                    data = simulatedTrain,
                    family = "binomial")
coef(summary(sigmoidalCal))

sigmoidProbs <- predict(sigmoidalCal,
                        newdata = simulatedTest[, "QDAprob", drop = F],
                        type = "response")
simulatedTest$QDAsigmoid <- sigmoidProbs

BayesCal <- NaiveBayes(class ~ QDAprob, data = simulatedTrain,
                       usekernel = T)
## Like qda(), the predict function for this model creates
## both the classes and the probabilities
BayesProbs <- predict(BayesCal, newdata = simulatedTest[, "QDAprob", drop = F])
simulatedTest$QDABayes <- BayesProbs$posterior[, "Class1"]

# Probability values before and after calibration
head(simulatedTest[, c(5:6, 8, 9)])

calCurve2 <- calibration(class ~ QDAprob + QDABayes + QDAsigmoid,
                         data = simulatedTest)
xyplot(calCurve2, auto.key = list(columns = 3))
