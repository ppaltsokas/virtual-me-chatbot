# getwd()
# setwd("C:\\Users\\user\\Desktop\\ΕΑΠ\\DAMA 51\\My_assignments\\HW2\\R_scripts")

# install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)

################ Topic 4a ################

# load the “abalone” dataset 
data(abalone)
# store the number of rows in a variable n
n <- nrow(abalone)
# define a matrix “X”, by concatenating by column a vector containing only ones with the 2nd to 8th column of the abalone dataset 
X <- cbind(matrix(1,n,1), abalone[, 2:8])
# remove the names of the columns
colnames(X)<-NULL
# Store the column Rings of the "abalone" dataset in a variable y
y <- abalone$Rings
# define a 8x1 vector that contains only zeroes
theta <- as.vector(matrix(0,8,1))


################ Topic 4b ################

# Define the function that calculates the MSE
mse_cost <- function(X, y, theta){
  
  n <- length(y)
  # calculate the mse cost
  s <- (1/(2*n))*sum((X %*% theta - y)^2)
  return(s)
  
}

# test the cost function
X0 <- cbind(rep(1,10), rep(1,10) )
y0 <- 5*rep(1,10)
theta0 <- rep(1,2)
mse_result <- mse_cost(X0,y0,theta0)
print(mse_result)


################ Topic 4c ################

gradientDescent <- function(X, y, theta, learning_rate, num_iter){
  
  nrOfObj <- length(y)
  #define a vector of zeros for storing the losses at each iteration
  l_loss <- numeric(num_iter)
  
  for( i in 1:num_iter){
    #calculate the gradient at X, y, using the current theta parameters
    gradient <- (1/nrOfObj) * (t(X) %*% ((X %*% theta) - y))
    #calculate the update of the parameters theta and store it in theta
    theta <- theta - learning_rate * gradient
    #calculate the loss at the I iteration and store it at the “l_loss” vector.
    l_loss[i] <- mse_cost(X, y, theta)
  }
  # for returning 2 objects in R we need to pack them in a list
  res <-list(theta = theta, l_loss = l_loss)
  return(res)
}

################ Topic 4d ################

gd_result <- gradientDescent(as.matrix(X), y, theta, 0.1, 150)

# Create a table to display calculated parameters after 150 steps
table_params <- data.frame(Parameter = paste0("θ[", 0:(nrow(gd_result$theta) - 1),"]"), Value = round(gd_result$theta,3))

print(table_params)

# Plot the MSE error at each step
plot(gd_result$l_loss, type = "l", xlab = "Iteration", ylab = "Mean Squared Error", main = "MSE Error at Each Step of Gradient Descent")