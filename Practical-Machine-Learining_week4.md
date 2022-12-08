## Overview

One thing that people regularly do is quantify how much of a particular
activity they do, but they rarely quantify how well they do it.

In this project, I will use data from accelerators on the belt, forearm,
arm, and dumbbell of 6 participants. The goal of this project is to
predict the manner in which they did the exercise. This is the “classe”
variable in the training set. Report describes building of a model,
using of a cross validation, what I expect out of sample error to be.

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

## Data Processing

    library(caret)

    ## Loading required package: ggplot2

    ## Loading required package: lattice

    library(randomForest)

    ## randomForest 4.7-1.1

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    library(corrplot)

    ## corrplot 0.92 loaded

    library(rpart)

## Data Processing

    trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    trainFile <- "./data/pml-training.csv"
    testFile  <- "./data/pml-testing.csv"
    if (!file.exists("./data")) {
      dir.create("./data")
    }
    if (!file.exists(trainFile)) {
      download.file(trainUrl, destfile=trainFile, method="curl")
    }
    if (!file.exists(testFile)) {
      download.file(testUrl, destfile=testFile, method="curl")
    }

## Read the Data

    trainRaw <- read.csv("./data/pml-training.csv")
    testRaw <- read.csv("./data/pml-testing.csv")
    dim(trainRaw)

    ## [1] 19622   160

    dim(testRaw)

    ## [1]  20 160

The training data set contains 19622 observations and 160 variables,
while the testing data set contains 20 observations and 160 variables.
The “classe” variable in the training set is the outcome to predict.

## Clean the data

First, we remove columns that contain NA missing values.

    trainRaw <- trainRaw[, colSums(is.na(trainRaw)) < .9] 
    trainRaw <- trainRaw[,-c(1:7)]

Removing near zero variance variables.

    nvz <- nearZeroVar(trainRaw)
    trainRaw <- trainRaw[,-nvz]
    dim(trainRaw)

    ## [1] 19622    53

Now, the cleaned training data set contains 19622 observations and 53
variables.

Next, we can split the cleaned training set into into a pure training
data set (70%) and a validation data set (30%). We will use the
validation data set to conduct cross validation in future steps.

    inTrain <- createDataPartition(y=trainRaw$classe, p=0.7, list=F)
    train <- trainRaw[inTrain,]
    valid <- trainRaw[-inTrain,]

## Data Modeling

We fit two models, namely Random Forest and Support Vector Machine.
First, I set up control for training to use threefold cross validation,
than the valid set.

# Random Forest

    controlRf <- trainControl(method="cv", 5)
    mod_rf <- train(classe~., data=train, method="rf", trControl = controlRf, tuneLength = 5)
    mod_rf

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10989, 10989, 10989, 10990, 10991 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9906821  0.9882124
    ##   14    0.9927932  0.9908835
    ##   27    0.9900272  0.9873836
    ##   39    0.9878434  0.9846198
    ##   52    0.9815100  0.9766059
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 14.

    pred_rf <- predict(mod_rf, valid)
    confusionMatrix(pred_rf, factor(valid$classe))

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1671    4    0    0    0
    ##          B    3 1130    5    0    0
    ##          C    0    5 1019    1    0
    ##          D    0    0    2  962    9
    ##          E    0    0    0    1 1073
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9949          
    ##                  95% CI : (0.9927, 0.9966)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9936          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9982   0.9921   0.9932   0.9979   0.9917
    ## Specificity            0.9991   0.9983   0.9988   0.9978   0.9998
    ## Pos Pred Value         0.9976   0.9930   0.9941   0.9887   0.9991
    ## Neg Pred Value         0.9993   0.9981   0.9986   0.9996   0.9981
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2839   0.1920   0.1732   0.1635   0.1823
    ## Detection Prevalence   0.2846   0.1934   0.1742   0.1653   0.1825
    ## Balanced Accuracy      0.9986   0.9952   0.9960   0.9978   0.9957

# Support Vector Machine

    mod_svm <- train(classe~., data=train, method="svmLinear", trControl = controlRf, tuneLength = 5, verbose = F)
    mod_svm

    ## Support Vector Machines with Linear Kernel 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10990, 10988, 10990, 10990, 10990 
    ## Resampling results:
    ## 
    ##   Accuracy  Kappa    
    ##   0.784669  0.7262753
    ## 
    ## Tuning parameter 'C' was held constant at a value of 1

    pred_svm <- predict(mod_svm, valid)
    confusionMatrix(pred_svm, factor(valid$classe))

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1521  167   77   65   50
    ##          B   36  809   98   39  137
    ##          C   58   69  796  105   71
    ##          D   55   16   27  699   60
    ##          E    4   78   28   56  764
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.7798         
    ##                  95% CI : (0.769, 0.7903)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.7201         
    ##                                          
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9086   0.7103   0.7758   0.7251   0.7061
    ## Specificity            0.9147   0.9347   0.9376   0.9679   0.9654
    ## Pos Pred Value         0.8090   0.7230   0.7243   0.8156   0.8215
    ## Neg Pred Value         0.9618   0.9308   0.9519   0.9473   0.9358
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2585   0.1375   0.1353   0.1188   0.1298
    ## Detection Prevalence   0.3195   0.1901   0.1867   0.1456   0.1580
    ## Balanced Accuracy      0.9117   0.8225   0.8567   0.8465   0.8358

The better model is Random Forest model, with accuracy .9958 compared to
Support Vector Machine Model, with accuracy 0.7832.

## Predicting for Test Data Set

Now, we apply the model to the original testing data set downloaded from
the data source.

    result <- predict(mod_rf, testRaw)
    print(result)

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

## Appendix: Figures

Correlation Matrix Visualization

    corrPlot <- cor(trainRaw[, -length(names(trainRaw))])
    corrplot(corrPlot, method="color")

![](Practical-Machine-Learining_week4_files/figure-markdown_strict/unnamed-chunk-11-1.png)
Plotting the models - Random Forest

    plot(mod_rf)

![](Practical-Machine-Learining_week4_files/figure-markdown_strict/unnamed-chunk-12-1.png)
