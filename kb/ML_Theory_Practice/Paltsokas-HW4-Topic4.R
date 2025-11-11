# getwd()
# setwd("C:\\Users\\user\\Desktop\\ΕΑΠ\\DAMA 51\\My_assignments\\HW4\\R_scripts")

################ Topic 4a ################
cat("\n################ Topic 4a ################\n")
# Load libraries for association rules and visualization
library("arules")
library("arulesViz")
# Load the data from a csv file as specified
visits <- read.transactions("countries.csv", format = "basket", header=FALSE, sep=",", rm.duplicates = FALSE)
# Print the summary of the visits transaction dataset
print(summary(visits))
# Plot the top10 most frequent items
itemFrequencyPlot(visits, topN = 10, col = "darkred", main = "Top 10 Most Frequent Items")
# Extract the most frequent item (country visited)
most_visited_country <- names(which.max(itemFrequency(visits)))
# Extract the names of the distinct items
unique_countries <- nitems(visits)
# Extract the item matrix density
density <- round(summary(visits)@density,3)
# Extract the maximum and minimum visits
min_visits <- min(size(visits))
max_visits <- max(size(visits))
# Print the requested data
cat("\nMost frequently visited country:", most_visited_country,
    "\nNumber of different  countries visited:", unique_countries, 
    "\nThe item matrix density is", density,
    "\nMaximum number of countries visited by a traveler:", max_visits, 
    "\nMinimum number of countries visited by a traveler:", min_visits)

################ Topic 4b ################
cat("\n\n################ Topic 4b ################\n")
# Generate association rules based on given parameters
rules <- apriori(visits, parameter = list(supp=0.2, conf=0.8, minlen=2, target= "rules"))
# Print the summary of the rules and inspect
print(summary(rules))
inspect(rules)
# Extract and store requested data in variables
rules_length <- length(rules)
minimum_items_rules <- sum(size(rules)==2)
maximum_items_rules <- sum(size(rules)==3)
# Print the requested data
cat("\nNumber of identified rules:", length(rules),
    "\nNumber of rules with maximum number of items involved:", maximum_items_rules, 
    "\nNumber of rules with minimum number of items involved:", minimum_items_rules)
# Generate association rules with Hungary as the antecedent
Hungary_lhs <- apriori(data = visits,
                         parameter = list(supp = 0.2, conf = 0.8, minlen = 2, target = "rules"), 
                         appearance = list(default="rhs",lhs="Hungary"))
# Inspect the rules
inspect(Hungary_lhs)
# Generate association rules with Belgium as the antecedent
Belgium_lhs <- apriori(data = visits,
                       parameter = list(supp = 0.2, conf = 0.8, minlen = 2, target = "rules"), 
                       appearance = list(rhs="Spain",lhs="Belgium"))
# Inspect the rules
inspect(Belgium_lhs)
# Generate association rules with France as the consequent
BelEsp <- apriori(data = visits,
                  parameter = list(supp = 0.2, conf = 0.8, minlen = 2, target = "rules"),
                  appearance = list(rhs = "France", default = "lhs"))
# Inspect the rules and look for the rule {Belgium, Spain} => {France}
inspect(BelEsp)
# Extract the support of the rule {Belgium, Spain} => {France}
BelEspSup <- quality(BelEsp)$support[4]
cat("\nThe support for the rule {Belgium, Spain} => {France} is",round(BelEspSup,3))
# Generate association rule with Cyprus as the antecedent and Greece as the consequent
CypGre <- apriori(data = visits,
                  parameter = list(supp = 0.2, conf = 0.8, minlen = 2, target = "rules"),
                  appearance = list(rhs = "Greece", lhs = "Cyprus"))
# Inspect the rule
inspect(CypGre)
# Extract the lift of the rule {Cyprus} => {Greece}
CypGreLift <- quality(CypGre)$lift
cat("\nThe lift for the rule {Cyprus} => {Greece} is",round(CypGreLift,3))

################ Topic 4c ################
cat("\n\n################ Topic 4c ################\n")

# Initialize minimum support threshold
i<-0.125
# Initialize an empty list to store the number of rules for each support value
visit_results <- list()
# Loop through support thresholds, incrementing by a step of 0.025, storing the number of rules for each value of 'i'.
while(TRUE){
  rules = apriori(data = visits,
          parameter = list(supp = i, conf = 0.8, minlen = 2, target = "rules"),control = list(verbose = FALSE)) # Set 'verbose = False' to avoid printing the details in the console and allowing for a more readable output.
  visit_results[[as.character(i)]] <- length(rules)
  print(paste("For minimum support threshold of:", i, ", the number of association rules are:", length(rules)))
  i <- i + 0.025
  if (i > 0.25){ # Break the loop when the support value exceeds 0.25
    break
    }
}
# Convert the names of the visit_results to numeric for plotting puproses
support_values <- as.numeric(names(visit_results))
# Convert the rule counts of visit_results to a numeric vector
rule_lengths <- unlist(visit_results)
# Plot the Number of Rules vs Minimum Support Threshold
plot(support_values, rule_lengths, type = "b",
     xlab = "Minimum Support Threshold", ylab = "Number of Rules",
     main = "Number of Rules vs. Minimum Support Threshold",
     col = "red", pch = 15)
# Fit a trend line to observer the general relationship
trend_line <- lm(rule_lengths ~ support_values)
abline(trend_line, col = "gray")
# Add a legend to describe the plot lines
legend("topright", legend = c("Number of Rules", "Trend Line"),
       col = c("red", "gray"), lty = 1:1, pch = 15)
# Annotate each point with its coordinates where x = 'support value' and y = 'rule lengths'
text(support_values, rule_lengths, labels = paste("(", support_values, " , ", rule_lengths, ")", sep = ""), pos = 3, offset = 0.5, col = "red", cex=0.6)
# Add a grid to the plot
grid()


################ Topic 4d ################
cat("\n################ Topic 4d ################\n")
# Generate association rules with Cyprus as the antecedent
Cyprus_lhs <- apriori(data = visits,
                       parameter = list(supp = 0.2, conf = 0.8, minlen = 2, target = "rules"), 
                       appearance = list(default="rhs",lhs="Cyprus"))
# Inspect the rules
inspect(Cyprus_lhs)
# Extract the countries that appear as the consequent in the rules where Cyprus is the antecedent
consequents <- rhs(Cyprus_lhs)
# List the aforementioned countries' names in a list
consequent_list <- as(consequents, "list")
cat("The countries that are included in the consequent in the rules where Cyprus is the antecedent are", consequent_list[[1]], "and", consequent_list[[2]])

################ Topic 4e ################
cat("\n\n################ Topic 4e ################\n")
cat("The rule suggests that there is strong indication that a significant proportion of travelers who visit Cyprus also tend to visit Greece. Geographical proximity, cultural similarities, historical ties or even shared language, traditions, lifestyle and cuisine  could be leading to a natural flow of tourists between the two countries.")
