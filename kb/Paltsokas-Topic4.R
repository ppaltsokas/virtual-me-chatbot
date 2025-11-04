# Load required packages and libraries
library(carData)
salaries_data <- Salaries

################ Topic 4a ################
cat("################ Topic 4a ################\n")
# We filter our dataframe and keep only the professors of Discipline A regardless of rank
discipline_A = salaries_data[salaries_data$discipline == "A",]
# We will now count the rows of the new dataframe we created
discipline_A_count=nrow(discipline_A)
# which will give us the required number of professors of discipline A
cat("The number of professors of Discipline A is", discipline_A_count, "\n\n")

################ Topic 4b ################
cat("################ Topic 4b ################\n")
# We filter our dataframe, keeping only the professors that have a salary over $150k
high_pay = salaries_data[salaries_data$salary > 150000,]
# Now we will filter the new dataframe according to sex
male_high_pay = high_pay[high_pay$sex == "Male",]
cat("There are", nrow(male_high_pay), "male professors and",  
    nrow(high_pay) - nrow(male_high_pay), "female professors with a salary of more than $150000\n\n")

################ Topic 4c ################
cat("################ Topic 4c ################\n")
# To calculate the average years of service of all professors we will sum up their experiences and divide by their count
cat("The average years of service of all professors is", round(sum(salaries_data$yrs.service)/nrow(salaries_data),3), "years \n")
# We filter our dataframe and keep only the Associate professors
associate_professors = salaries_data[salaries_data$rank == "AssocProf", ]
# Same as before we calculate the average years of service of Associate professors
cat("The average years of service of the Associate professors is", 
    round(sum(associate_professors$yrs.service)/nrow(associate_professors),3),"years \n\n")

################ Topic 4d ################
cat("################ Topic 4d ################\n")
# Create a new column 'career_stage' where we assign values early, mid and late to the intervals (-1,10], (10,25] and (25,60] respsectively.
salaries_data$career_stage <- cut(salaries_data$yrs.service, breaks = c(-1, 10, 25, 60), labels = c("early", "mid", "late"))

# Count the number of professors at an early career stage
num_early_career_professors <- sum(salaries_data$career_stage == "early")

cat("There are", num_early_career_professors, "early career stage professors.\n\n")

################ Topic 4e ################
cat("################ Topic 4e ################\n")
# Create a copy of Salaries dataframe, organized in a decreasing order according to the yrs.service attribute.
salaries_data_ordered <- salaries_data[order(-salaries_data$yrs.service), ]

# Print out the data of the top 3 professors with the most years of service
top3_professors <- head(salaries_data_ordered, 3)
print(top3_professors)

