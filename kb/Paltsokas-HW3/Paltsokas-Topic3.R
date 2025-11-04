# getwd()
# setwd("C:\\Users\\user\\Desktop\\ΕΑΠ\\DAMA 51\\My_assignments\\HW2\\R_scripts")

################ Topic 3a ################
cat("\n####### Topic 3a #######\n")
#install.packages("cluster")
library(cluster)
data(USArrests)
# Load the USArrests dataset for clustering analysis
df_data <- USArrests
# Scale the data so that each feature has a mean of 0 and a standard deviation of 1
df_data_scaled <- as.data.frame(scale(df_data))
# Display the structure of the original and scaled data to understand the transformations applied
cat("Structure of Original dataframe\n")
str(df_data)
cat("\nStructure of Scaled dataframe\n")
str(df_data_scaled)
# Calculate and print the mean and standard deviation for the original and scaled data for comparison
original <- data.frame(means=round(sapply(df_data, mean),3), sd=round(sapply(df_data, sd),3))
scaled <- data.frame(means=round(sapply(df_data_scaled, mean),3),sd= round(sapply(df_data_scaled, sd),3))
cat("\nOriginal Dataframe: \n")
print(original)
cat("Scaled Dataframe: \n\n")
print(scaled)


################ Topic 3b ################
cat("\n####### Topic 3b #######\n")
# define a function that takes a method x and computes the agglomerative coefficient
# by the agnes function on the scaled data
agglom_coef <- function(x) {
  agglomerative_coefficient <- agnes(df_data_scaled, method = x)$ac
  return(agglomerative_coefficient)
}
# define an array of the methods to be tested
linkage_methods <- c("average", "single", "complete")
#apply the agglom_coef function to the methods defined earlier
coefficients <- round(sapply(linkage_methods, agglom_coef),3)
cat("Agglomerative coefficients of the clustering of the linkage methods\n")
print(coefficients)
cat("\n")

################ Topic 3c ################
cat("####### Topic 3c #######\nDendrogram Plot\n\n")
# calculate the Euclidean distance matrix
d_eucl <- dist(df_data_scaled, method = "euclidean")

# use the hclust method for calculating the agglomerative clustering
clust_complete <- hclust(d_eucl, method = "complete")

# plot the dendogram
plot(clust_complete, main = "Dendrogram - Hierarchical Clustering", xlab = "States", ylab = "Dissimilarity")


################ Topic 3d ################
cat("####### Topic 3d #######\n")
# Use cutree to assign data points to clusters
ind <- cutree(clust_complete, k=5)

# Create a data frame with original features and cluster assignment
data_5_clusters <- data.frame(State = rownames(df_data), df_data, Cluster = ind)
cat("Clustering in 5 groups\n")
print(head(data_5_clusters))
cat("\n")

# Calculate and report the mean values of the 4 attributes per cluster
means_per_cluster <- aggregate(. ~ Cluster, data = data_5_clusters[, -1], mean)
cat("Mean values of the four attributes per cluster\n")
print(round(means_per_cluster,3))



