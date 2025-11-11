setwd("C:\\Users\\user\\Desktop\\ΕΑΠ\\DAMA 51\\My_assignments\\HW5\\")
getwd()

################ Topic 2a ################
cat("\n################ Topic 2a ################\n")
eyes <- read.csv2("eyes.csv", sep=',')
# Inspect the dataset
dim(eyes)
names(eyes)
str(eyes)
summary(eyes)
head(eyes)
tail(eyes)

# # Store the attributes values for later use
# ageval <- unique(eyes$Age)
# visionval <- unique(eyes$Vision)
# astigmatismval <- unique(eyes$Astigmatism)
# UseOfGlassesval <- unique(eyes$UseOfGlasses)
# Classval <- unique(eyes$Class)

# # Entropy Function
# subs=subset(eyes,eyes['Vision'] == 'Farsightedness')
# s1<-nrow(subset(subs,subs$Class == 'A'))
# s2<-nrow(subset(subs,subs$Class == 'B'))
# s<- nrow(subs)
# y <- c(s1/s, s2/s)
# E <- -y[1]*(log(y[1])/log(2))-y[2]*(log(y[2])/log(2))
# E

################ Topic 2b ################
cat("\n################ Topic 2b ################\n")
# Create the function that calculates the entropy left after the split of dataframe x on attribute attr, for the generated subset with attribute value subsetParam.
eye_attr_entropy_calc <- function(x, attr, subsetParam){ # The function asks for 3 parameters, x, attr and subsetParam.
  subs <- subset(x,x[attr] == subsetParam) # Then creates a subset of dataframe x with value of attr being subsetParam.
  s <- nrow(subs) # store the number of rows of the subset
  s1 <- nrow(subset(subs,subs$Class == 'A')) # store the number of rows of the subset that are classified as A
  s2 <- nrow(subset(subs,subs$Class == 'B')) # store the number of rows of the subset that are classified as B
  y <- c(s1/s, s2/s) # Create a vector that stores the probabilities p1 and p2 of an item of the subset belonging in class A and class B respectively.
  if (s1==0 || s2==0) # if p1 V p2 == 0 returns 0
    return (0)
  else
    return (-y[1]*(log2(y[1]))-y[2]*(log2(y[2]))) # if p1 ^ p2 != 0, it calculates the Shannon Entropy
}
# Use the function to calculate the entropy left for the requested subsets after splitting on the requested attribute.
e_middle_aged <- eye_attr_entropy_calc(eyes,'Age', 'Middle-aged')
e_myopia <- eye_attr_entropy_calc(eyes,'Vision','Myopia')
e_Astigmatism_yes <- eye_attr_entropy_calc(eyes,'Astigmatism','Yes')
e_glasses_often <- eye_attr_entropy_calc(eyes,'UseOfGlasses','Rare')
# Store the attributes, the subset parameters and the entropies in vectors to create a dataframe for the presentation of the results
results <- c(e_middle_aged, e_myopia, e_Astigmatism_yes, e_glasses_often) # Store the entropies in a vector
# Create the dataframe 
df_2b <- data.frame(attr = c("Age","Vision","Astigmatism","UseOfGlasses"),
                    subsetParam = c("Middle-Ages","Myopia","Yes","Rare"),
                    Output_entropy = round(results,3))
print(df_2b)



################ Topic 2c ################
cat("\n################ Topic 2c ################\n")
# Calculate Entropy Before the split
A_count <- sum(eyes$Class == 'A') # Store the number of items in the original dataset that belong in class A
B_count <- sum(eyes$Class == 'B') # Store the number of items in the original dataset that belong in class B
pA <- A_count/nrow(eyes) # Calculate the probability of an item of the original dataset belonging in class A
pB <- B_count/nrow(eyes) # Calculate the probability of an item of the original dataset belonging in class B
entropy_before <- -pA*log2(pA)-pB*log2(pB) # Calculate the entropy before the split

# Create a function that calculates the relative frequency of each subset defined by `subsetParam` within the attribute `attr`. 
weight_calc <- function(attr, subsetParam){
  s <- nrow(eyes)
  s1 <- sum(eyes[attr] == subsetParam) 
  return(s1/s)
}

# Create a function that calculates the entropy after requested attribute split
entropy_after <- function(attr){
  unique_values <- unique(eyes[[attr]]) # Get the unique values of the attribute
  total_entropy <- 0 # Initialize the total entropy

  for (value in unique_values){ # Loop over each unique attribute
    weight <- weight_calc(attr, value) # Calculate the weight for the current value
    entropy <- eye_attr_entropy_calc(eyes, attr, value) # Calculate the entropy for the current value
    total_entropy <- total_entropy + (weight * entropy) # Sum the weighted entropies
  }

  return(total_entropy)
}

# Calculate entropy after splitting by 'Age', 'Vision', 'Astigmatism', and 'UseOfGlasses', and the information gain from each split by subtracting the entropy after the split from the entropy before the split.
entropy_after_age <- entropy_after('Age')
information_gain_age <- entropy_before - entropy_after_age


entropy_after_vision <- entropy_after('Vision')
information_gain_vision <- entropy_before - entropy_after_vision


entropy_after_astigmatism <- entropy_after('Astigmatism')
information_gain_astigmatism <- entropy_before - entropy_after_astigmatism


entropy_after_uog <- entropy_after('UseOfGlasses')
information_gain_uog <- entropy_before - entropy_after_uog

cat("Information gain after splitting by Age: ", round(information_gain_age,3))
cat("\nInformation gain after splitting by Vision: ", round(information_gain_vision,3))
cat("\nInformation gain after splitting by Astigmatism: ", round(information_gain_astigmatism,3))
cat("\nInformation gain after splitting by UseOfGlasses: ", round(information_gain_uog,3))


################ Topic 2d ################
cat("\n\n################ Topic 2d ################\n")
library(caret)
library(rpart.plot)
eye_data <-read.csv('eyes.csv', stringsAsFactors = TRUE) # Read the dataset, treating strings as categorical factors
colnames(eye_data)[5] <- "output" # Rename the attribute 'Class' to 'output' for clarity and terminology reasons
dt <- rpart(output ~ ., data = eye_data, method="class", control=rpart.control(minsplit=1)) # Create a decision tree using the rpart function, the “Information Gain” as the splitting method, and setting the minimum number of observations in a node for a split to be attempted equal to 1
rpart.plot(dt) # Plot the decision tree
cat("Plot the Decision Tree\n")

# Create a dataframe with the new records to be classified (including the ones not requested for validation reasons)
df_2d <- data.frame(Age = c("Young","Young","Young","Elderly","Elderly"),
                    Vision = c("Myopia","Myopia", "Farsightedness","Myopia","Farsightedness"),
                    Astigmatism = c("Yes","No","No","Yes","No"),
                    UseOfGlasses = c("Often", "Often","Often","Often","Rare"))

# Convert each column to a factor with levels matching those in eye_data
for (col_name in colnames(df_2d)){
  df_2d[[col_name]] <- factor(df_2d[[col_name]], levels=levels(eye_data[[col_name]]))
}

# Use the decision tree model to predict the class of the new records
predicted_classes <- predict(dt, df_2d, type = "class")
print(predicted_classes)

# Add the predicted classes to the dataframe and print
df_2d$Class <- predicted_classes
cat("\nDataframe with predicted classes\n")
print(df_2d)
