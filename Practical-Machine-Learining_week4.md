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
    ## Summary of sample sizes: 10991, 10989, 10989, 10991, 10988 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9909741  0.9885821
    ##   14    0.9927937  0.9908841
    ##   27    0.9913378  0.9890424
    ##   39    0.9895176  0.9867399
    ##   52    0.9850048  0.9810286
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 14.

    pred_rf <- predict(mod_rf, valid)
    confusionMatrix(pred_rf, factor(valid$classe))

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    9    0    0    0
    ##          B    0 1129    6    0    0
    ##          C    0    1 1013    5    1
    ##          D    0    0    7  958    2
    ##          E    0    0    0    1 1079
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9946          
    ##                  95% CI : (0.9923, 0.9963)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9931          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9912   0.9873   0.9938   0.9972
    ## Specificity            0.9979   0.9987   0.9986   0.9982   0.9998
    ## Pos Pred Value         0.9947   0.9947   0.9931   0.9907   0.9991
    ## Neg Pred Value         1.0000   0.9979   0.9973   0.9988   0.9994
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1918   0.1721   0.1628   0.1833
    ## Detection Prevalence   0.2860   0.1929   0.1733   0.1643   0.1835
    ## Balanced Accuracy      0.9989   0.9950   0.9929   0.9960   0.9985

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
    ## Summary of sample sizes: 10989, 10990, 10988, 10990, 10991 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.7784836  0.7183794
    ## 
    ## Tuning parameter 'C' was held constant at a value of 1

    pred_svm <- predict(mod_svm, valid)
    confusionMatrix(pred_svm, factor(valid$classe))

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1534  143   89   67   49
    ##          B   34  833  102   42  139
    ##          C   56   53  780  109   73
    ##          D   41   26   35  700   59
    ##          E    9   84   20   46  762
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7832          
    ##                  95% CI : (0.7724, 0.7936)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.7244          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9164   0.7313   0.7602   0.7261   0.7043
    ## Specificity            0.9174   0.9332   0.9401   0.9673   0.9669
    ## Pos Pred Value         0.8151   0.7243   0.7283   0.8130   0.8274
    ## Neg Pred Value         0.9650   0.9354   0.9489   0.9475   0.9355
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2607   0.1415   0.1325   0.1189   0.1295
    ## Detection Prevalence   0.3198   0.1954   0.1820   0.1463   0.1565
    ## Balanced Accuracy      0.9169   0.8323   0.8502   0.8467   0.8356

The better model is Random Forest model, with accuracy .9939 compared to
Support Vector Machine Model, with accuracy 0.7825.

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
