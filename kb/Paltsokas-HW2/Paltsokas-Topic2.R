getwd()
setwd("C:\\Users\\user\\Desktop\\ΕΑΠ\\DAMA 51\\My_assignments\\HW2\\R_scripts")

################ Topic 2a ################
cat("################ Topic 2a ################\n\n")
# Reading the data using read.csv into wine_data dataframe
wine_data <- read.csv("wine.csv", sep=";")

cat("############# Reading the structure of the dataframe and printing the types of the requested attributes #############\n\n")
str(wine_data)
# Printing the type of the attributes requested
cat("\n\nMagnesium feature type is: ", {(typeof(wine_data$magnesium))})
cat("\n\nFlavonoids feature type is: ", {(typeof(wine_data$flavanoids))})

cat("\n\n################ Printing the first five rows of the dataframe ################\n\n")

print(head(wine_data,5))

################ Topic 2b ################
cat("\n################ Topic 2b ################\n\n")
# Removing the last column
wine_data <- wine_data[, -ncol(wine_data)]

# Calculating the means of all attributes 
means <- round(colMeans(wine_data),3)
sd <- round(apply(wine_data,2,sd),3)
wine_means_sd <- data.frame(Wine_Atrribute = names(wine_data), Mean = means, Standard_Deviation = sd, row.names = NULL) 
print(wine_means_sd)

################ Topic 2c ################
cat("\n################ Topic 2c ################\n\n")
# Standardize the data
wine_data_scaled <- scale(wine_data)
# Compute the covariance matrix
cov_matrix <- cov(wine_data_scaled)
# Compute the eigenvalues and the eigenvectors
eigen_info <- eigen(cov_matrix)
eigenvalues <- eigen_info$values
eigenvectors <- eigen_info$vectors
# Calculate the proportion of variance explained by each component
variance_explained <- eigenvalues / sum(eigenvalues)
variance_explained_pct <- round(variance_explained*100,3)
wine_data_pca = prcomp(wine_data_scaled)
wine_data_pca
# Export the principal components' names PC1, PC2, etc to use them in the dataframe
PCnames = colnames(wine_data_pca$rotation)
# Create a dataframe with the principal components and their respective percentages of variance explained
variance_explained_df <- data.frame(Principal_Component = PCnames, Variance_Explained_Pct = variance_explained_pct, row.names = NULL)
print(variance_explained_df)

################ Topic 2d ################
# Plot the scree plot to visualize the percentage of variance explained per principal component
plot(1:length(variance_explained_pct), variance_explained_pct, type = "b", 
     main = "Scree Plot", xlab = "Principal Component", ylab = "Percentage of Variance Explained")




