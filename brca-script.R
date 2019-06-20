## Breast cancer biopsy prediction - modeling code
## Amy Gill - June 6, 2019

# libraries and options
library(tidyverse)
library(RColorBrewer)
library(gplots)
library(GGally)
library(factoextra)
library(matrixStats)
library(caret)
options(digits = 3)

# import data
url <- "http://mlr.cs.umass.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
brca_nuclei <- read.csv(url, header = FALSE)

# add variable names - info from wdbc.names file
x <- c("radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_pts", "symmetry", "fractal_dim")
y <- c("mean", "se", "worst")
temp <- expand.grid(x = x, y = y)    # build variable names as all combos of x,y
names(brca_nuclei) <- c("id", "type", paste(temp$x, temp$y, sep = "_"))

# arrange by type to group like samples together
brca_nuclei <- brca_nuclei %>% arrange(type)

# extract numeric features
features <- brca_nuclei %>% select(-id, -type) %>% as.matrix()

# extract tumor type
type <- brca_nuclei$type

# combine features and type into brca object
brca <- list(x = features, y = type)

# scale x values
x_centered <- sweep(brca$x, 2, colMeans(brca$x))
x_scaled <- sweep(x_centered, 2, colSds(brca$x), FUN = "/")

# split brca$x and brca$y into 20% test and 80% training sets
set.seed(1)
test_index <- createDataPartition(brca$y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index,]
test_y <- brca$y[test_index]
train_x <- x_scaled[-test_index,]
train_y <- brca$y[-test_index]

# table of train/test proportions of benign/malignant
data.frame(Dataset = c("Train", "Test"),
           Benign = c(mean(train_y == "B"), mean(test_y == "B")),
           Malignant = c(mean(train_y == "M"), mean(test_y == "M")))

# function to generate predictions from a k-means model
predict_kmeans <- function(x, k) {
    centers <- k$centers    # extract cluster centers
    # calculate distance to cluster centers
    distances <- sapply(1:nrow(x), function(i){
        apply(centers, 1, function(y) dist(rbind(x[i,], y)))
    })
    max.col(-t(distances))  # select cluster with min distance to center
}

# train k-means model, generate k-means predictions
set.seed(3)
k <- kmeans(train_x, centers = 2) 
kmeans_preds <- ifelse(predict_kmeans(test_x, k) == 1, "B", "M")
mean(kmeans_preds == test_y)

# train logistic regression model, generate regression predictions
train_glm <- train(train_x, train_y, method = "glm")
glm_preds <- predict(train_glm, test_x)
mean(glm_preds == test_y)

# train LDA model, generate LDA predictions
train_lda <- train(train_x, train_y, method = "lda")
lda_preds <- predict(train_lda, test_x)
mean(lda_preds == test_y)

# train QDA model, generate QDA predictions
train_qda <- train(train_x, train_y, method = "qda")
qda_preds <- predict(train_qda, test_x)
mean(qda_preds == test_y)

# train loess model, generate loess predictions
set.seed(5)
train_loess <- train(train_x, train_y, method = "gamLoess")
loess_preds <- predict(train_loess, test_x)
mean(loess_preds == test_y)

# train kNN model, find best k, generate predictions
set.seed(7)
tuning <- data.frame(k = seq(3, 21, 2))    # try odd values of k from 3 to 21
train_knn <- train(train_x, train_y,
                   method = "knn", 
                   tuneGrid = tuning)
train_knn$bestTune    # best value of k
knn_preds <- predict(train_knn, test_x)    # generate predictions
mean(knn_preds == test_y)

# train random forest model, find best mtry, generate predictions, measure accuracy
set.seed(9)
tuning <- data.frame(mtry = seq(3, 21, 2))    # try odd values of mtry from 3 to 21
train_rf <- train(train_x, train_y,
                  method = "rf",
                  tuneGrid = tuning,
                  importance = TRUE)
train_rf$bestTune    # best value of mtry
rf_preds <- predict(train_rf, test_x)    # generate predictions
mean(rf_preds == test_y)    # calculate accuracy

## create data frame of model performance
# model names
models <- c("K means", "Logistic regression", "LDA", "QDA", "Loess", "K nearest neighbors", "Random forest")
# model accuracies
accuracy <- c(mean(kmeans_preds == test_y),
              mean(glm_preds == test_y),
              mean(lda_preds == test_y),
              mean(qda_preds == test_y),
              mean(loess_preds == test_y),
              mean(knn_preds == test_y),
              mean(rf_preds == test_y))
# calculate model sensitivity (rate of correct malignant classification)
sens <- c(sensitivity(factor(kmeans_preds), test_y, positive = "M"),
          sensitivity(factor(glm_preds), test_y, positive = "M"),
          sensitivity(factor(lda_preds), test_y, positive = "M"),
          sensitivity(factor(qda_preds), test_y, positive = "M"),
          sensitivity(factor(loess_preds), test_y, positive = "M"),
          sensitivity(factor(knn_preds), test_y, positive = "M"),
          sensitivity(factor(rf_preds), test_y, positive = "M"))
# calculate model specificity (rate of correct benign classification)
specif <- c(sensitivity(factor(kmeans_preds), test_y, positive = "B"),
            sensitivity(factor(glm_preds), test_y, positive = "B"),
            sensitivity(factor(lda_preds), test_y, positive = "B"),
            sensitivity(factor(qda_preds), test_y, positive = "B"),
            sensitivity(factor(loess_preds), test_y, positive = "B"),
            sensitivity(factor(knn_preds), test_y, positive = "B"),
            sensitivity(factor(rf_preds), test_y, positive = "B"))
# data frame of model performance
data.frame(Model = models, Accuracy = accuracy, Sensitivity = sens, Specificity = specif) %>%
    mutate("False Negative Rate" = 1-Sensitivity)

# generate ensemble by building a matrix of all model predictions
ensemble <- cbind(kMeans = ifelse(kmeans_preds == "B", 0, 1),
                  logisticReg = ifelse(glm_preds == "B", 0, 1),
                  LDA = ifelse(lda_preds == "B", 0, 1),
                  QDA = ifelse(qda_preds == "B", 0, 1),
                  loess = ifelse(loess_preds == "B", 0, 1),
                  kNN = ifelse(knn_preds == "B", 0, 1),
                  randomForest = ifelse(rf_preds == "B", 0, 1))

# generate predictions from 7-model ensemble
ensemble_preds <- ifelse(rowMeans(ensemble) < 0.5, "B", "M")

# generate ensemble of LDA and kNN only and generate predictions
ensemble2 <- cbind(LDA = ifelse(lda_preds == "B", 0, 1),
                   kNN = ifelse(knn_preds == "B", 0, 1))
ensemble2_preds <- ifelse(rowSums(ensemble2) > 0, "M", "B")

# summary statistics for ensemble model
data.frame(Model = c("7-model ensemble"),
           Accuracy = c(mean(ensemble_preds == test_y)),
           Sensitivity = c(sensitivity(factor(ensemble_preds), test_y, positive = "M")),
           Specificity = c(sensitivity(factor(ensemble_preds), test_y, positive = "B"))) %>%
    mutate("False Negative Rate" = 1 - Sensitivity)