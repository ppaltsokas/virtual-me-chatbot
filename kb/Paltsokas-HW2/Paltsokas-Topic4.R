# getwd()
# setwd("C:\\Users\\user\\Desktop\\ΕΑΠ\\DAMA 51\\My_assignments\\HW2\\R_scripts")
data <- read.csv("productivity_data.csv", sep = ";")
cat("########################## Topic 4b ########################## \n")
cat("######### Inspect the first 5 rows of the dataframe #########\n")

print(head(data))


cat("\n######### Inspect the structure of the dataframe #########\n")

str(data)


cat("\n######### Defining the hypotheses for question (i) #########

Null Hypothesis (H0): The training program did not significantly change the employee's productivity.
Alternative Hypothesis (H1): The training program changed significantly the employee's productivity.\n\n")


cat("######### Conducting a paired t-test #########")

result <- t.test(data$After, data$Before, paired = TRUE)
print(result)


cat("######### Display the p-value #########\n")

pvalue <- round(result$p.value,3)
print(pvalue)

alpha <- 0.05
cat("\n# Conclusion #\n")
if (result$p.value < alpha) {
  cat("At a significance level of 0.05, the p-value is ", {pvalue}," so we reject the null hypothesis. There is evidence that the training had an effect on productivity.\n")
} else {
  cat("\nAt a significance level of 0.05, the p-value is ", {pvalue}," so we fail to reject the null hypothesis. There is no significant evidence that the training had an effect on productivity.\n")
}
cat("\n-----------------------------------------------------------------------------------------------")



cat("\n\n######### Defining the hypotheses for question (ii) #########

Null Hypothesis (H0): The training program did not significantly improve the employee's productivity.
Alternative Hypothesis (H1): The training program imrpoved significantly the employee's productivity.\n\n")


cat("######### Conducting a paired t-test #########")

result_improve <- t.test(data$After, data$Before, paired = TRUE, alternative = "greater")
print(result_improve)


cat("######### Display the p-value #########\n")

improve_pvalue <- round(result_improve$p.value,3)
print(improve_pvalue)

cat("\n# Conclusion #\n")
if (result_improve$p.value < alpha) {
  cat("At a significance level of 0.05, the p-value is ", {improve_pvalue}," so we reject the null hypothesis. There is evidence that the training improved productivity.\n")
} else {
  cat("At a significance level of 0.05, the p-value is ", {improve_pvalue}," so we fail to reject the null hypothesis. There is no significant evidence that the training improved productivity.\n")
}
cat("\n-----------------------------------------------------------------------------------------------")





cat("\n\n#########################################\n")
cat("############## ANOTHER WAY ##############\n")
cat("#########################################\n\n")

####################################
#### t-test for mean difference ####
####################################
cat("\n(i)\n\nNull Hypothesis (H0): μ0 == 0
Alternative Hypothesis (H1): μ0 != 0 
where μ0 is the mean difference\n\n")

library(tidyverse)

cat("######## Create a new column for the differences ########\n")
newdata <- data %>% 
  mutate(Difference = After - Before)

# Display the first few rows of the updated dataframe
print(head(newdata))

cat("\n######## Perform a one-sample t-test ########\n")
result_one_sample <- t.test(newdata$Difference, mu = 0)

# Display the results
print(result_one_sample)

cat("\n# Conclusion #\n")
if (result_one_sample$p.value < alpha) {
  cat("At a significance level of 0.05, the p-value is ", {round(result_one_sample$p.value,3)}," so we reject the null hypothesis. There is evidence that the training changed productivity.\n")
} else {
  cat("At a significance level of 0.05, the p-value is ", {round(result_one_sample$p.value,3)}," so we fail to reject the null hypothesis. There is no significant evidence that the training changed productivity.\n")
}
cat("\n-----------------------------------------------------------------------------------------------")


cat("\n\n(ii)\n\nNull Hypothesis (H0): μ0 <= 0
Alternative Hypothesis (H1): μ0 > 0 
where μ0 is the mean difference\n\n")


cat("\n######## Perform a one-sample t-test ########\n")
result_one_sample_improve <- t.test(newdata$Difference, mu = 0, alternative = "greater")

# Display the results
print(result_one_sample_improve)

cat("\n# Conclusion #\n")
if (result_one_sample_improve$p.value < alpha) {
  cat("At a significance level of 0.05, the p-value is ", {round(result_one_sample_improve$p.value,3)}," so we reject the null hypothesis. There is evidence that the training improved productivity.\n")
} else {
  cat("At a significance level of 0.05, the p-value is ", {round(result_one_sample_improve$p.value,3)}," so we fail to reject the null hypothesis. There is no significant evidence that the training imrpoved productivity.\n")
}
cat("\n-----------------------------------------------------------------------------------------------")



cat("\n\n\n########################## Topic 4e ########################## \n")

cat("\n######## Create the table ########\n\n")
trainprog <- matrix(c(35, 23, 41, 16), nrow = 2, byrow = TRUE)
rownames(trainprog) <- c("Male", "Female")
colnames(trainprog) <- c("Satisfied", "Unsatisfied")
print(trainprog)
cat("\n\n######## Perform the chi-squared test ########\n")
result <- chisq.test(trainprog, correct = FALSE)
print(result)
cat("\n######## print the p-value ########\n\np-value = ", {round(result$p.value,3)})


cat("\n\n\n########################## Extra ########################## \n")
cat("\nI created a density plot to help me visualize any change in productivity before and after the training \n")
# Load necessary libraries
library(ggplot2)

# Calculate means and variances
mean_before <- mean(data$Before)
mean_after <- mean(data$After)
var_before <- var(data$Before)
var_after <- var(data$After)

# Create kernel density plots with means and variances
print(ggplot(data, aes(x = Before, fill = "Before")) +
  geom_density(alpha = 0.5) +
  geom_density(aes(x = After, fill = "After"), alpha = 0.5) +
  geom_vline(xintercept = mean_before, linetype = "dashed", color = "blue", linewidth = 1) +
  geom_vline(xintercept = mean_after, linetype = "dashed", color = "red", linewidth = 1) +
  annotate("rect", xmin = mean_before - sqrt(var_before), xmax = mean_before + sqrt(var_before), ymin = 0, ymax = Inf, fill = "blue", alpha = 0.2) +
  annotate("rect", xmin = mean_after - sqrt(var_after), xmax = mean_after + sqrt(var_after), ymin = 0, ymax = Inf, fill = "red", alpha = 0.2) +
  labs(title = "Density Plot of Productivity Before and After Training",
       x = "Productivity",
       y = "Density") +
  scale_fill_manual(values = c("Before" = "blue", "After" = "red")) +
  theme_minimal())


