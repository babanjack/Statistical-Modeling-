# install.packages(c('caret', 'mlbench', 'kernlab','corrplot'))
# load packages
library(caret)
library(mlbench)
library(corrplot)
library(kernlab)
# attach the BostonHousing dataset
data("BostonHousing")

# Split out validation dataset
# create a list of 80% of the rows in the original dataset we can use for training
set.seed(2)
validationIndex <- createDataPartition(BostonHousing$medv,p = 0.8,list = F)
# select 20% of the data for validation
validation <- BostonHousing[-validationIndex,]
# use the remaining 80% of data to training and testing the models
dataset <- BostonHousing[validationIndex,]
# dimensions of dataset
dim(dataset)
#list types for each attribute
sapply(dataset,class)

# take a peek at the first 5 rows of the data
head(dataset,25)

# summarize attribute distributions
summary(dataset)

# convert the chas to numeric attribute
dataset[,4] <- as.numeric(as.character(dataset[,4]))

# correlation between the numeric attribute
cor(dataset[,1:13])

# histograms each attribute
par(mfrow =c(2,7))
for(i in 1:13) {
  hist(dataset[,i], main=names(dataset)[i])
}
# density plot for each attribute
par(mfrow=c(2,7))
for(i in 1:13) {
  plot(density(dataset[,i]), main=names(dataset)[i])
}
# boxplots for each attribute
par(mfrow=c(2,7))
for(i in 1:13) {
  boxplot(dataset[,i], main=names(dataset)[i])
}
dim(validation)
# scatterplot matrix
pairs(dataset[,1:13])
# correlation plot
correlations <- cor(dataset[,1:13])
corrplot(correlations, method="circle")


## Run algorithms using 10-fold cross validation
trainControl <- trainControl(method = "repeatedcv",number = 10,repeats = 3)
metric <- "RMSE"

# LM
set.seed(2)
fit.lm <- train(medv~., data=dataset, method="lm", metric=metric, preProc=c("center","scale"), trControl=trainControl)

# GLM
set.seed(2)
fit.glm <- train(medv~., data=dataset, method="glm", metric=metric, preProc=c("center","scale"), trControl=trainControl)


# GLMNET
set.seed(2)
fit.glmnet <- train(medv~., data=dataset, method="glmnet", metric=metric,
                    preProc=c("center", "scale"), trControl=trainControl)

# SVM
set.seed(2)
fit.svm <- train(medv~., data=dataset, method="svmRadial", metric=metric,
                 preProc=c("center", "scale"), trControl=trainControl)
# CART
set.seed(2)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(medv~., data=dataset, method="rpart", metric=metric, tuneGrid=grid,
                  preProc=c("center", "scale"), trControl=trainControl)

# KNN
set.seed(2)
fit.knn <- train(medv~., data=dataset, method="knn", metric=metric, preProc=c("center",
                                                                              "scale"), trControl=trainControl)
# Compare algorithms
results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm,
                          CART=fit.cart, KNN=fit.knn))
summary(results)
dotplot(results)


# remove correlated attributes
# find attributes that are highly corrected
set.seed(2)
cutoff <- 0.70
correlations <- cor(dataset[,1:13])
highlyCorrelated <- findCorrelation(correlations, cutoff=cutoff)
for (value in highlyCorrelated) {
  print(names(dataset)[value])
}

#create a new dataset without highly corrected features
datasetFeatures <- dataset[,-highlyCorrelated]
dim(datasetFeatures)


# Run algorithms using 10-fold cross validation
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# lm
set.seed(2)
fit.lm <- train(medv~., data=datasetFeatures, method="lm", metric=metric,
                preProc=c("center", "scale"), trControl=trainControl)
# GLM
set.seed(2)
fit.glm <- train(medv~., data=datasetFeatures, method="glm", metric=metric,
                 preProc=c("center", "scale"), trControl=trainControl)
# GLMNET
set.seed(2)
fit.glmnet <- train(medv~., data=datasetFeatures, method="glmnet", metric=metric,
                    preProc=c("center", "scale"), trControl=trainControl)
# SVM
set.seed(2)
fit.svm <- train(medv~., data=datasetFeatures, method="svmRadial", metric=metric,
                 preProc=c("center", "scale"), trControl=trainControl)
# CART
set.seed(2)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(medv~., data=datasetFeatures, method="rpart", metric=metric,
                  tuneGrid=grid, preProc=c("center", "scale"), trControl=trainControl)
# KNN
set.seed(2)
fit.knn <- train(medv~., data=datasetFeatures, method="knn", metric=metric,
                 preProc=c("center", "scale"), trControl=trainControl)
# Compare algorithms
feature_results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm,
                                  CART=fit.cart, KNN=fit.knn))
summary(feature_results)
dotplot(feature_results)


# Run algorithms using 10-fold cross validation
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# lm
set.seed(2)
fit.lm <- train(medv~., data=dataset, method="lm", metric=metric, preProc=c("center",
                                                                            "scale", "BoxCox"), trControl=trainControl)
# GLM
set.seed(2)
fit.glm <- train(medv~., data=dataset, method="glm", metric=metric, preProc=c("center",
                                                                              "scale", "BoxCox"), trControl=trainControl)
# GLMNET
set.seed(2)
fit.glmnet <- train(medv~., data=dataset, method="glmnet", metric=metric,
                    preProc=c("center", "scale", "BoxCox"), trControl=trainControl)
# SVM
set.seed(2)
fit.svm <- train(medv~., data=dataset, method="svmRadial", metric=metric,
                 preProc=c("center", "scale", "BoxCox"), trControl=trainControl)
# CART
set.seed(2)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(medv~., data=dataset, method="rpart", metric=metric, tuneGrid=grid,
                  preProc=c("center", "scale", "BoxCox"), trControl=trainControl)
# KNN
set.seed(2)
fit.knn <- train(medv~., data=dataset, method="knn", metric=metric, preProc=c("center",
                                                                              "scale", "BoxCox"), trControl=trainControl)

# Compare algorithms
transformResults <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm,
                                   CART=fit.cart, KNN=fit.knn))
summary(transformResults)
dotplot(transformResults)
print(fit.svm)


# tune SVM sigma and C parametres
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
set.seed(2)
grid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
fit.svm <- train(medv~., data=dataset, method="svmRadial", metric=metric, tuneGrid=grid,
                 preProc=c("BoxCox"), trControl=trainControl)
print(fit.svm)
plot(fit.svm)


# try ensembles
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# Random Forest
set.seed(2)
fit.rf <- train(medv~., data=dataset, method="rf", metric=metric, preProc=c("BoxCox"),
                trControl=trainControl)
# Stochastic Gradient Boosting
set.seed(2)
fit.gbm <- train(medv~., data=dataset, method="gbm", metric=metric, preProc=c("BoxCox"),
                 trControl=trainControl, verbose=FALSE)
# Cubist
set.seed(2)
fit.cubist <- train(medv~., data=dataset, method="cubist", metric=metric,
                    preProc=c("BoxCox"), trControl=trainControl)


# Compare algorithms
ensembleResults <- resamples(list(RF=fit.rf, GBM=fit.gbm, CUBIST=fit.cubist))
summary(ensembleResults)
dotplot(ensembleResults)
# look at parameters used for Cubist
print(fit.cubist)



# Tune the Cubist algorithm
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
set.seed(2)
grid <- expand.grid(.committees=seq(15, 25, by=1), .neighbors=c(3, 5, 7))
tune.cubist <- train(medv~., data=dataset, method="cubist", metric=metric,
                     preProc=c("BoxCox"), tuneGrid=grid, trControl=trainControl)
print(tune.cubist)
plot(tune.cubist)


# prepare the data transform using training data
set.seed(2)
x <- dataset[,1:13]
y <- dataset[,14]
preprocessParams <- preProcess(x, method=c("BoxCox"))
transX <- predict(preprocessParams, x)
# train the final model
finalModel <- cubist(x=transX, y=y, committees=25)
summary(finalModel)

# transform the validation dataset
set.seed(2)
valX <- validation[,1:13]
trans_valX <- predict(preprocessParams, valX)
valY <- validation[,14]
# use final model to make predictions on the validation dataset
predictions <- predict(finalModel, newdata=trans_valX, neighbors=3)
# calculate RMSE
rmse <- RMSE(predictions, valY)
r2 <- R2(predictions, valY)
print(rmse)
summary(predictions)
