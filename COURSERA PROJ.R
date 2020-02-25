# Execution Summary
# First loading in the data and performing exploritory analysis
# I load the required package/data for data cleaning and model fitting,
# In the Model sectiton,
# I fit the data using first using decision trees then using
# random forest with cross validation.
# Finally, in the Prediction section, I use the model to predict the test set.

# Loading in the data
train <- read.csv(file = file.choose())
test <- read.csv(file = file.choose(), header = T)


# Exploratory analysis
summary(train)

dim(train)

str(train)

# removing variables that are almost always NA
indCol <- which(colSums(is.na(train) | train=="")>0.9*dim(train)[1])
indCol
trainc <- train[,-indCol]

# The first seven columns have information on the people who took the test,
# Removing those 7 columns
trainc <- trainc[,-c(1:7)]

dim(trainc)

# Exploratory analysis
summary(trainc)

str(trainc)

dim(trainc)

# performing the same function on test set
indCol <- which(colSums(is.na(test) | test=="") > 0.9*dim(test)[1])
testc <- test[, -indCol]
testc <- testc[, -1]

###
dim(testc)
str(testc)


library(caret)
# Spliting the data into training and test set
ntrian <- createDataPartition(trainc$classe, p = 0.75, list = F)
traindata <- trainc[ntrian, ]
testdata <- trainc[-ntrian, ]

dim(traindata)

                                # MODEL SELECTION
# Prediction with classification trees
# Creating the model
library(rpart.plot)
library(rpart)
library(rattle)

set.seed(0311)
dtmdl1 <- rpart(classe ~ ., data=traindata, method="class")

# ploting the tree as dendogra
fancyRpartPlot(dtmdl1)


# # Testing our model on development data set
predtreemdl1 <- predict(dtmdl1, testdata, type = "class")
conftree <- confusionMatrix(predtreemdl1, testdata$classe)
conftree

# This gives us an accuracy of just 74% which is not good enough 


                              # Random Forest
library(doSNOW)

# By default Rstudio provides 1 core
# we need more than one core for training the Model
cl <- makeCluster(4, type = "SOCK")

# have to register < R does not have auto register for dosnow
registerDoSNOW(cl)

# using 5 fold cross validation to selet best tuning paramets
fitControl <- trainControl(method="cv", number=5, verboseIter=F)

# Training the model
set.seed(1103)
mdl <- train(classe ~ ., data=traindata, method="rf",
             trControl=fitControl)


stopCluster(cl)

# Checking the final model
mdl$finalModel

# Testing our model on development data set
predrf1 <- predict(mdl, newdata=testdata)
conf <- confusionMatrix(predrf1, testdata$classe)
conf

# The accuracy rate using the random forest is very high
# therefore the out-of-sample-error is negligible.
# But it might be due to overfitting.

# Ploting the model for errors
plot(mdl)
plot(conf$table, col = conf$byClass,
     main = paste("Random Forest Confusion Matrix: Accuracy =",
                  round(conf$overall['Accuracy'], 4)))

# Predicting on our test data set
Final <- predict(mdl, newdata=test)

Final


write.csv(Final, file = "PREDICTIONS")


