#Create working directory
setwd('/Users/bhagyashreejagdish/Downloads')
#get the working directory
getwd()
# Load the dataset
heart_disease <- read.csv("heart-disease.csv")

# Load required libraries
library(tidyverse) # For data manipulation and visualization
library(glmnet) # For ridge regression
library(caret) # For model evaluation
library(gridExtra) # For arranging multiple plots
library(ggplot2) # For ridge regression plot
library(rpart) # For decision trees
library(rpart.plot) # For plotting decision trees
library(e1071) # For SVM
library(randomForest) # For Random Forest
library(pROC) # For ROC Curve plotting
library(xgboost) # For Boosting

##1 EXPLORATORY DATA ANALYSIS (EDA)
str(heart_disease) # Check the structure of the dataset
summary(heart_disease) # Summary statistics of numeric variables
sum(is.na(heart_disease)) # Check for missing values

# Initialize an empty list to store individual plots
plot_list <- list()
# Iterate over each numeric variable (excluding ID and CHD column)
numeric_cols <- names(heart_disease)[sapply(heart_disease, is.numeric)]
numeric_cols <- setdiff(numeric_cols, c("id", "chd"))
for (col in numeric_cols) {
  # Create a distribution plot for each factor with CHD
  p <- ggplot(heart_disease, aes_string(x = col, fill = factor(heart_disease$chd))) +
    geom_density(alpha = 0.5) +
    labs(title = paste("Distribution of", col, "by CHD Status"),
         x = col,
         y = "Density",
         fill = "CHD Status") +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 10)  # Adjust title text size
    )
  # Store the plot in the list
  plot_list[[col]] <- p
}
# Arrange the plots in a grid
multi_plot <- do.call(grid.arrange, c(plot_list, ncol = 3))
# Display the grid of plots
multi_plot

##2 LINEAR REGRESSION WITH RIDGE PENALTY
# Define predictors and target variable
predictors <- c("sbp", "tobacco", "ldl", "famhist", "typea", "obesity", "alcohol", "age", "adiposity")
target <- "chd"

# Prepare data for modeling
x <- model.matrix(as.formula(paste(target, "~ .")), data = heart_disease)[, -1] # Exclude the intercept column
y <- heart_disease[[target]]

# Fit logistic regression with ridge penalty
set.seed(123)  # for reproducibility
ridge_model <- cv.glmnet(x, y, alpha = 0, family = "binomial")
print(ridge_model)

# Plot the binomial deviance against the penalty parameter (lambda)
plot(ridge_model, xvar = "lambda", label = TRUE)

# Extract coefficients for different lambda values
coef_matrix <- as.matrix(coef(ridge_model$glmnet.fit))
log_lambda <- log(ridge_model$glmnet.fit$lambda)

# Plot the coefficient paths
matplot(log_lambda, t(coef_matrix[-1, ]), type = "l", lty = 1, col = 1:nrow(coef_matrix[-1, ]),
        xlab = "Log(Lambda)", ylab = "Coefficients", main = "Coefficient Paths for Ridge Regression")

# Add legend
legend("topright", legend = predictors, col = 1:length(predictors), lty = 1, cex = 0.7)

# Model Evaluation
y_pred <- predict(ridge_model, newx = x, s = "lambda.min", type = "response")
y_pred_binary <- ifelse(y_pred > 0.5, 1, 0)
accuracy <- mean(y_pred_binary == y)
cat("Ridge Penalty Regression Accuracy:", accuracy, "\n")

##3 OTHER CLASSIFIERS
#3.1: Decision Trees
dt_model <- rpart(chd ~ ., data = heart_disease, method = "class")
actual <- factor(heart_disease$chd)
predicted <- predict(dt_model, type = "class")
predicted <- factor(predicted, levels = levels(actual))
# Calculate confusion matrix and accuracy
confusion_matrix <- confusionMatrix(predicted, actual)
accuracy <- confusion_matrix$overall['Accuracy']
cat("Decision Tree Accuracy:", accuracy, "\n")
# Plot the decision tree
rpart.plot(dt_model, main = "Decision Tree for Heart Disease Prediction")

#3.2: Random Forests
heart_disease$chd <- factor(heart_disease$chd)
# Split data into training (70%) and testing (30%)
set.seed(123)  # Set seed for reproducibility
train_index <- createDataPartition(heart_disease$chd, p = 0.7, list = FALSE)
train_data <- heart_disease[train_index, ]
test_data <- heart_disease[-train_index, ]
# Fit Random Forest model on training data
rf_model <- randomForest(chd ~ ., data = train_data, method = "class")
# Predict on testing data
rf_predicted <- predict(rf_model, newdata = test_data)
# Evaluate accuracy on testing data
rf_accuracy <- confusionMatrix(rf_predicted, test_data$chd)$overall['Accuracy']
cat("Random Forest Accuracy:", rf_accuracy, "\n")
# Plot variable importance
varImpPlot(rf_model)

#3.3: Support Vector Machines (SVM)
# Define predictors and target variable
predictors <- c("sbp", "tobacco", "ldl", "famhist", "adiposity", "typea", "obesity", "alcohol", "age")
target <- "chd"
# Prepare data for modeling
x <- heart_disease[, predictors]
y <- as.factor(heart_disease[[target]])
# Split data into training and testing sets
set.seed(123) # for reproducibility
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
x_train <- x[train_index, ]
y_train <- y[train_index]
# Combine predictors and target variable into a data frame
train_data <- data.frame(x_train, chd = y_train)
# Fitting SVM model and evaluate accuracy
tryCatch({
  # Fit SVM model
  svm_model <- svm(chd ~ ., data = train_data, kernel = "linear")
  
  # Predict on testing data
  x_test <- x[-train_index, ]
  predicted <- predict(svm_model, newdata = x_test)
  
  # Ensure predicted and actual values have the same levels
  y_test <- y[-train_index]
  predicted <- factor(predicted, levels = levels(y_test))
  
  # Evaluate model accuracy
  accuracy <- confusionMatrix(predicted, y_test)$overall['Accuracy']
  cat("SVM Accuracy:", accuracy, "\n")
}, error = function(e) {
  cat("Error message:", e$message, "\n")
})

##4 EVALUATING THE DECISION TREE MODEL
# Example new input (replace with actual values)
new_input <- data.frame(
  sbp = 130,
  tobacco = 12.0,
  ldl = 5.0,
  famhist = "Present",
  typea = 50,
  obesity = 25.0,
  alcohol = 14.0,
  age = 45,
  adiposity = 23.0
)
# Predict the class for the new input
new_prediction <- predict(dt_model, new_input, type = "class")
cat("Prediction for new input (Decision Tree):", new_prediction, "\n")
