setwd("C:\\Users\\user\\Desktop\\ΕΑΠ\\DAMA 51\\My_assignments\\HW5\\")
getwd()

################ Topic 4a ################
cat("\n################ Topic 4a ################\n")

      
ds <- data.frame( x = c(rep(5,12), rep(6,8), rep(3,5), rep(4,14),
rep(2,12), rep(1,3)), y = c(rep(700,12), rep(750,8), rep(600,5),
rep(650,14), rep(400,12), rep(350,3)))

A <- matrix(c(nrow(ds), sum(ds$x), sum(ds$x), sum(ds$x^2)), nrow = 2) # Setup matrix A for the normal equations based on the design matrix
b <- matrix(c(sum(ds$y), sum(ds$x * ds$y)), nrow = 2) # Setup vector b for the normal equations based on the output vector
sol <- solve(A,b)
print(round(sol,3))      

# Validate the results using the lm() function
x <- ds[,1]
y <- ds[,2]
model <- lm(y~x)
summary(model)
plot(x, y, main = "Oldness vs Maintenance Costs", xlab = "Oldness", ylab = "Maintenance Cost", pch = 20)
abline(lm(y~x), col="red")

a <- sol[1]
b <- sol[2]
cat("\nRregression line (ε) : y=",a,"+",b,"x\n\n")


x_new <- c(7,10,12,15) # Store the new values of oldness in a vector
y_pred <- a + b * x_new # Compute predicted maintenance costs for new values of oldness

results <- data.frame(X=x_new, Predicted_Y = round(y_pred,3)) # Create a dataframe with the results and print
print(results)

################ Topic 4b ################
cat("\n################ Topic 4b ################\n")

calc_output <- function (x, w, b){ # Define the function to calculate the neuron's output using the ReLU activation
  return (max(0,sum(w*x)+b))
}
# Set requested input values, weights, and biases
x <- cbind(c(0.1,0.3,0.5,1),c(0.1,0.5,0.9,1))
w <- cbind(c(0.75, 0.75, 0.1, 0.25), c(0.75,0.25,0.6,0.25))
b <- c(0.5, 0.1, 0.8, 1.25)

# Initialize a vector to store results
results <- numeric(4)

# Loop through each set of inputs, weights, and biases to compute neuron outputs
for (j in 1:4) {
  results[j] <- calc_output(x[j, ], w[j, ], b[j])
}

# Create a dataframe to store and print the results
final_table <- data.frame(
  "Input_1" = x[, 1],
  "Input_2" = x[, 2],
  "Weights_1" = w[, 1],
  "Weights_2" = w[, 2],
  "Bias" = b,
  "Output_of_the_Model" = results
)

print(final_table)


################ Topic 4c ################
cat("\n################ Topic 4c ################\n")
update_w <- function (x, w, b, delta, a){
  w <- w + a*delta*x # update the weights
  b <- b + a*delta # update the bias
  return (c(w, b))
}

# Set requested input values, weights, and biases
x <- cbind(c(0.1,0.3,0.7,0.8), c(0.9,0.1,0.9,0.2))
w <- cbind(c(0.7,0.4,0.4,0.1), c(0.3,0.1,0.2,1.1))
b <- c(1, -1, 0.5, 0.5)
delta <- c(-1, 1, 3, 2)
a <- c(0.1, 0.2, 0.01, 0.1)

# Create an empty list to store the results
results <- list()
# Apply updates and compile results into the list
for (j in 1:4) {
  results[[j]] <- update_w(x[j, ], w[j, ], b[j], delta[j], a[j])
}
# Create a dataframe to store and print the results
final_table <- data.frame(
  "Input_1" = x[, 1],
  "Input_2" = x[, 2],
  "Weights_1" = w[, 1],
  "Weights_2" = w[, 2],
  "Bias" = b,
  "Delta" = delta,
  "Learning_Rate" = a,
  "Updated_Weight_1" = sapply(results, function(v) v[1]),
  "Updated_Weight_2" = sapply(results, function(v) v[2]),
  "Updated_Bias" = sapply(results, function(v) v[3])
)

print(final_table)


################ Topic 4d ################
cat("\n################ Topic 4d ################\n")
# Set requested input values, desired outputs,weights and bias, and learning rate
x <- cbind(c(0,1,0,1), c(0,0,1,1))
y <- c(0,0,0,1)
w <- c(0.5, 0.5, 0.5) # third value is bias
a <- 0.1

# Train the neuron for a specified number of epochs
for (i in 1:1000){ # number of epochs
  for (j in 1:4){ # iteration over each of the input vectors
    output <- calc_output(x[j,],w[1:2],w[3]) # calculates the predicted output using the current weights and bias
    delta <- y[j] - output # the error term
    w <- update_w(x[j,],w[1:2],w[3],delta,a) # updates the weights and bias
  }
}

# Create a list to store the results
results <- list()
# Calculate final outputs for all input combinations
for (j in 1:4) {
  results[j] <- calc_output(x[j, ], w[1:2], w[3])
}
# Create a dataframe to store and print the results
cat("\n### Final Table ###\n")
final_table <- data.frame(
  "Input_1" = x[,1],
  "Input_2" = x[,2],
  "Desired_Output" = y,
  "Output_of_the_Model" = sapply(results, function(v) v[1])
)
print(final_table)
# Print the final weights and bias
cat("\n### Final Weights and Bias ###\n")
final_weights_bias <- data.frame("Weight1" = w[1], "Weight2" = w[2], "Bias" = w[3])
print(final_weights_bias)

