# Load required packages and libraries
library(carData)
states_data <- States

################ Topic 3a ################
cat("################ Topic 3a ################\n\n")
cat("Plot 1\n\n")
cat("The scatterplot indicates that there is a medium negative correlation between Average SAT Verbal Scores and \nThousands of Dollars Spend on Public Education.\n\n")

# Making sure my plot layout is a 1x1 region
par(mfrow = c(1, 1))
# Create a scatter plot using alpha for low opacity that allows to spot points that land on top of another. The more bright a red point is, the higher the count of points that land there. 
plot(states_data$dollars, states_data$SATV, 
     xlab = "Thousands of Dollars Spent on Public Education", 
     ylab = "SAT Average Verbal Scores", 
     main = "SATV vs. State Spending",
     cex.axis = 1.1, 
     col = rgb(1, 0, 0, alpha = 0.4),  # Red with lower alpha
     pch = 16) # Solid circle marker


################ Topic 3b ################
# We will visualize all four plots in the same window to be able to visually
# identify which pair of variables has the greatest correlation.

# First, create a 2x2 layout for the plots
par(mfrow = c(2, 2))

# Scatter plot for SATV and pay
plot(states_data$pay, states_data$SATV, 
     xlab = "Average Teacher's Salary", 
     ylab = "SAT Average Verbal Scores",
     main = "Teacher's Salary vs. SATV",
     cex.axis = 1.1,
     col = rgb(1, 0, 0, alpha = 0.4),  # Red with lower alpha
     pch = 16  # Solid circle marker
)

# Scatter plot for SATM and dollars
plot(states_data$dollars, states_data$SATM, 
     xlab = "Thousands of dollars Spent on Public Education", 
     ylab = "SAT Average Math Scores",
     main = "Spending on Education vs. SATM",
     cex.axis = 1.1,
     col = rgb(1, 0, 0, alpha = 0.4),  
     pch = 16  
)

# Scatter plot for SATV and pop
plot(states_data$pop, states_data$SATV, 
     xlab = "Region's Population", 
     ylab = "SAT Average Verbal Scores",
     main = "Region's Population vs. SATV",
     cex.axis = 1.1,
     col = rgb(1, 0, 0, alpha = 0.4),  
     pch = 16  
)

# Scatter plot for SATM and pop
plot(states_data$pop, states_data$SATM, 
     xlab = "Region's Population", 
     ylab = "SAT Average Math Scores",
     main = "Region's Population vs. SATM",
     cex.axis = 1.1,
     col = rgb(1, 0, 0, alpha = 0.4),  
     pch = 16  
)
cat("################ Topic 3b ################\n\n")
cat("Plot 2\n\n")
cat("We observe that all pairs have negative correlation with the Teacher's Salary and SATV average score having the greatest correlation by absolute value since the data points in this plot seem to form a straight line better than the others.",
    "\n\n")

################ Topic 3c ################
# Verifying by calculating the correlation coefficients.

# Calculate Spearman correlation coefficients
cor_pay_satv_sp <- round(cor(states_data$pay, states_data$SATV, method = "spearman"),3)
cor_dollars_satm_sp <- round(cor(states_data$dollars, states_data$SATM, method = "spearman"),3)
cor_pop_satv_sp <- round(cor(states_data$pop, states_data$SATV, method = "spearman"),3)
cor_pop_satm_sp <- round(cor(states_data$pop, states_data$SATM, method = "spearman"),3)

cat("################ Topic 3c ################\n\nCalculating Spearman correlation coefficients for requested variable pairs:\n\n")
# Display Spearman correlation coefficients
cat("Spearman Correlation coefficient between Spending on Education and SATM:", cor_dollars_satm_sp, "\n")
cat("Spearman Correlation coefficient between Teacher's Salary and SATV:", cor_pay_satv_sp, "\n")
cat("Spearman Correlation coefficient between Region's Population and SATV:", cor_pop_satv_sp, "\n")
cat("Spearman Correlation coefficient between Region's Population and SATM:", cor_pop_satm_sp, "\n")

# Find the pair with the greatest absolute Spearman correlation coefficient by selecting the maximum absolute value.
max_abs_correlation_spearman <- max(
  abs(cor_pay_satv_sp),
  abs(cor_dollars_satm_sp),
  abs(cor_pop_satv_sp),
  abs(cor_pop_satm_sp)
)

# Display the result
cat("\nThe pair with the greatest correlation by absolute value is Teacher's Salary and SATV Average Score, with an absolute Spearman correlation coefficient value of :", max_abs_correlation_spearman, 
    "\n")


# Reseting my plot layout to a 1x1 region
par(mfrow = c(1, 1))





