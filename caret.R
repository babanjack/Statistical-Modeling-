library(caret)
library(kernlab)
library(randomForest)
# attach the iris dataset to the environment
data("iris")
# rename the dataset
dataset <- iris

# create a list of 80% of the rows in the original dataset we can use for training
ValidationIndex <- createDataPartition(dataset$Species,p = 0.8,list = FALSE)

# select 20% of the data for validation
validation <- dataset[-ValidationIndex,]

# use the remaining 80% of data to training and testing the models
dataset <- dataset[ValidationIndex,]

# dimensions of the dataset
dim(dataset)

#list types of each attribute
sapply(dataset,class)

# peek at the first 5 rows of the data
head(dataset,5)

# list the levels for the class
levels(dataset$Species)

# summarize the class distribution
 percentage=prop.table(table(dataset$Species))*100
cbind(freq=table(dataset$Species),percentage=percentage) 

# summarize attribute distributions
summary(dataset)

# split input and output
x <- dataset[,1:4]
y <- dataset[,5]

# boxplot for each attribute on one image
par(mfrow = c(1,4))
  for(i in 1:4) {
    boxplot(x[,i], main=names(iris)[i])
  }    

# barplot for class breakdown
plot(y)
# scatterplot matrix
featurePlot(x=x,y=y,plot = "ellipse")

# box and whisker plots for each attribute
featurePlot(x=x ,y=y, plot = "box")

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)


# Run algorithms using 10-fold cross validation
trainControl <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#LDA
set.seed(2)
fit.lda <- train(Species~.,data = dataset,method="lda",metric=metric,trControl=trainControl)

# CART
set.seed(2)
fit.cart <- train(Species~.,data = dataset,method="rpart",metric=metric,trControl=trainControl)

#KNN
set.seed(2)
fit.knn <- train(Species~.,data = dataset,method="knn",metric=metric,trControl=trainControl)

# SVM
set.seed(2)
fit.svm <- train(Species~.,data = dataset,method="svmRadial",metric=metric,trControl=trainControl)

# RandomForest
set.seed(2)
fit.rf <- train(Species~.,data = dataset,method="rf",metric=metric,trControl=trainControl)

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.lda)

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda,validation)
confusionMatrix(predictions,validation$Species)
