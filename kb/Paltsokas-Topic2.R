# Load required packages and libraries
library(carData)
salaries_data <- Salaries

################ Topic 2a ################
#Making sure my plot layout is a 1x1 region
par(mfrow = c(1, 1))

# Exploring the data's structure using str() and head() which returns the 6 first rows of the dataframe.
cat("################ Topic 2a ################\n\n")
cat("Using str(salaries_data)\n\n")
str(salaries_data)
cat("\n")
cat("Using head(salaries_data)\n\n")
print(head(salaries_data))
cat("\n\n")

################ Topic 2b ################
cat("################ Topic 2b ################\n\n")
cat("Creating a contigency table :\n\n")

# Create a contingency table setting the rank as rows and sex as columns
contingency_table <- table(salaries_data$rank, salaries_data$sex, dnn=c("rank","sex"))
# Add row and column sums
contingency_table <- addmargins(contingency_table, FUN = list(SUM = sum))
cat("\n")
# Print the contingency table
print(contingency_table)

################ Topic 2c ################
cat("\n")
cat("################ Topic 2c ################\n\n")
cat("Plot 1\n\n")

# Create a subset of the previous contigency_table without the SUMS
contingency_table_subset <- contingency_table[-nrow(contingency_table), -ncol(contingency_table)]
# Define colors for each attribute value
colors <- c("aquamarine", "coral", "deepskyblue3")
# Create a mosaic plot
mosaicplot(t(contingency_table_subset), color = colors, main = "Mosaic Plot of Rank and Sex", cex.axis = 1.2)

################ Topic 2d ################
cat("################ Topic 2d ################\n\n")
cat("Plot 2\n\n")
# Create the histogram of at least 7 bins
hist(salaries_data$salary, breaks = 7, col = "skyblue", main = "Absolute Frequency of Salaries", xlab = "Salary", ylab = "Frequency", xlim=c(50000,250000), ylim=c(0,120))
