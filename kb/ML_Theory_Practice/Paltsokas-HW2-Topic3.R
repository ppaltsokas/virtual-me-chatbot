################ Topic 3b ################
# Input TPR and FPR data
tpr <- c(0, 0.2, 0.4, 0.4, 0.6, 0.6, 1.0, 1.0)
fpr <- c(0, 0, 0, 0.2, 0.2, 0.6, 0.6, 1.0)

# Plot the ROC curve
plot(fpr, tpr, type = "l", col = "red", lty = 1, xlab = "False Positive Rate (FPR)", ylab = "True Positive Rate (TPR)", main = "ROC Curve", xlim = c(0, 1), ylim = c(0, 1))

# Add diagonal line for random choice
abline(a = 0, b = 1, col = "grey", lty = 2)


# Add labels and legend
title("ROC Curve")
legend("bottomright", legend = "Random choice", col = "grey", lty = 2)


################ Topic 3c ################
# install.packages("caret")
# Load the care library to create the confusion matrix
library(caret)

# Input our data and threshold
predicted_prob <- c(0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.5)
actual_class <- c(1, 1, 0, 1, 0, 0, 1, 1, 0, 0)
threshold <- 0.65

# Classify predictions based on the threshold so that if the predicted probability is greater than the threshold
# the prediciton is 1, otherwise 0
predicted_class <- ifelse(predicted_prob >= threshold, 1, 0)

# Create a confusion matrix and set the positive case to "1"
conf_matrix <- confusionMatrix(factor(predicted_class), factor(actual_class), positive = "1")

# Print confusion matrix
print(conf_matrix)
