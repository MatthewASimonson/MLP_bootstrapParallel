library(MASS)
library(neuralnet)
library(foreach)
library(doParallel)
# Use a multi-layer perceptron neural network for prediction of housing prices, use parallel bootstrap aggregating (bagging) to reduce prediction error
# Set seed

set.seed(123)

# Storing data set Boston in a data frame

DataFrame <- Boston

# structure of Boston Data:

str(DataFrame)

# Histogram of medv (median value of owner-occupied homes in \$1000s.)

hist(DataFrame$medv)

# check first 3 rows

head(DataFrame,3)

# check the dimensions of the data frame

dim(DataFrame)

# Get the min and max values of each variabgle

apply(DataFrame,2,range)

# scale variables:

maxValue <- apply(DataFrame,2,max)
minValue <- apply(DataFrame,2,min)

DataFrame <- as.data.frame(scale(DataFrame,center = minValue,scale = maxValue))

# Use bagging with bootstrap aggregation to reduce prediction variability:
iter <- 1000
booted <- matrix(NA,nrow=nrow(DataFrame),ncol=iter)
bootedActual <- matrix(NA,nrow=nrow(DataFrame),ncol=iter)

#for (i in 1:iter){
registerDoParallel(cores=30)
loop_NN <- foreach(i=1:iter) %dopar% {
    
# Create the training and test sets:  
DataFrame$bootindex <- 1:nrow(DataFrame)
ind <- sample(1:nrow(DataFrame),400)
trainDF <- DataFrame[ind,]
testDF <- DataFrame[-ind,]


# Configure the neural network:
# Layers 10-10-10-1
# number of hidden layers = 3
# input layer has 10 units
# Formula:
# medv ~ crim + zn + indus + chas + nox + rm + age + dis +rad + tax + ptratio + Lstat

# Generate formula:
allVars <- colnames(DataFrame)[1:ncol(DataFrame)-1] # exclude index column
outcome <- "medv~"
predictorVars <- allVars[!allVars%in%"medv"] # excluded outcome variable
predictorVar <- paste(predictorVars,collapse="+")
form <- as.formula(paste(outcome,predictorVar,collapse="+"))

neuralModel <- neuralnet(formula=form,hidden = c(10,10,10), linear.output = TRUE,data = trainDF)

# Plot the neural net model

 plot(neuralModel)

# Predict from test data set

predFromMod <- compute(neuralModel,testDF[,1:13])
#str(predFromMod)

unscale <- (max(testDF$medv)-min(testDF$medv))+min(testDF$medv)
predictions <- predFromMod$net.result*unscale

# store predictions from training and test set at appropriate index
booted[DataFrame$bootindex[-ind],i] <- predictions
bootedActual[DataFrame$bootindex[-ind],i] <- DataFrame$medv[-ind] # store test set for interation
print(i)
}

bootedValues <- apply(booted,1,mean,na.rm=TRUE)
actualValues <- apply(bootedActual,1,mean,na.rm=TRUE)
both <- cbind.data.frame(bootedValues,actualValues)

# Get bootstrapped R-square (% reduction in error)
Rsq <- cor(both$bootedValues,both$actualValues)^2






