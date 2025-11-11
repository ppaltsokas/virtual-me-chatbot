setwd("C:\\Users\\user\\Desktop\\ΕΑΠ\\DAMA 51\\My_assignments\\HW5\\")
getwd()

################ Topic 3a ################
cat("\n################ Topic 3a ################\n")
library(dplyr)
data("iris")
iris <- iris %>% mutate(Class = case_when(Species =='setosa' ~ 1,Species!= 'setosa' ~ -1))
ds <- iris[,c(3,4,6)]
w = c(1,1)
C <- 1
r <- 0.1

calc_J <- function (ds, w, C){
  #calculate the max
  tmp <- 1 - (w[1]*ds$Petal.Length+w[2]*ds$Petal.Width)*ds$Class
  mMax <- pmax(tmp, 0, na.rm = TRUE)
  return (1/2 * sum(w^2) + C * sum(mMax)) # Return the value of the objective function
}

# Define weights and constants to be used in the objective function calculations
weights <- list(c(0.1,0.1), c(0.1,0.1), c(0.5,0.5), c(0.5,0.5), c(1,1), c(1,1)) 
constants <- c(1, 0.5, 1, 0.5, 1, 0.5)

# Print table headers for objective function results
cat(sprintf("%-15s %-15s %-15s\n", "Weights", "Constant", "Objective Function J(W)")) 
for (i in 1:6){ # Loop through each set of weights and constants to calculate and print J values
  w = weights[[i]]
  C = constants[i]
  J = calc_J(ds, w, C)
  cat(sprintf("w = (%.1f,%.1f)    C = %.1f        J = %.2f\n", w[1], w[2], C, J)) # Define their format and assign the values to be printed
}

################ Topic 3b ################
cat("\n################ Topic 3b ################\n")

calc_DJ <- function (ds, w, C){
  #calculate the max
  tmp <- 1-(w[1]*ds$Petal.Length+w[2]*ds$Petal.Width)*ds$Class
  mMax <- pmax(tmp, 0, na.rm = TRUE)
  #find values which are 0
  indices <-t(which(mMax!=0, arr.ind = TRUE))
  d1 <- -C * sum(ds$Class[indices] * ds$Petal.Length[indices]) / nrow(ds) # Calculate gradient with regard to w1
  d2 <- -C * sum(ds$Class[indices] * ds$Petal.Width[indices]) / nrow(ds) # Calculate gradient with regard to w2
  return (c(d1,d2))
}

# Define weights and constants to be used in the gradient calculations
weights_b <- list(c(0.5,0.5), c(0.2,0.2), c(0.8,0.8), c(1,1))
constants_b <- c(1, 0.5, 0.1, 0.5)

# Print table headers for gradient results
cat(sprintf("%-15s %-15s %-15s\n", "Weights", "Constant", "Gradient ∇J(w)"))
for (i in 1:4){ # Loop through each set of weights and constants to calculate and print gradient values
  w = weights_b[[i]]
  C = constants_b[i]
  DJ <- calc_DJ(ds, w, C)
  cat(sprintf("w = (%.1f,%.1f)    C = %.1f      DJ = (%.3f, %.3f)\n", w[1], w[2], C, DJ[1],DJ[2])) # Define their format and assign the values to be printed
}

################ Topic 3c ################
cat("\n################ Topic 3c ################\n")
# Initialize weights, learning rate, and regularization constant
w <- c(1, 1)  # Initial weight vector
C <- 1        # Regularization constant
r <- 0.1      # Learning rate

# Print headers for iteration results
cat(sprintf("%-15s %-15s %-25s %-25s\n", "Iteration", "J", "Gradient ∇J(w)", "Updated w"))

# Perform iterations
for (i in 1:5) {
  J <- calc_J(ds, w, C) # Calculate the current objective function value
  DJ <- calc_DJ(ds, w, C) # Calculate the gradient
  w <- w - r * DJ  # # Update weights with gradient descent formula
  # Define their format and assign the values to be printed
  cat(sprintf("%d            %.3f            (%.3f, %.3f)       (%.3f, %.3f)\n", i, J, DJ[1], DJ[2], w[1], w[2]))
}

################ Topic 3d ################
cat("\n################ Topic 3d ################\n")
w = c(1,1) # Initialize vector w, so as to avoid continuing from previous experiments
set.seed(2024)
for( i in 1:5 ){
  indices <- round(runif(50, min=1, max=nrow(ds)),0)
  J <- calc_J(ds[indices,], w, C) # Calculate the objective function J for the random subset
  DJ <- calc_DJ(ds[indices,], w, C) # Calculate the gradient of J for the random subset
  w <- w - r * DJ
  cat('iteration = ', i, '\n')
  cat('J = ', round(J,3), '\n')
  cat('dJ = ', round(DJ,3), '\n')
  cat('w = ', round(w,3), '\n\n')
}







