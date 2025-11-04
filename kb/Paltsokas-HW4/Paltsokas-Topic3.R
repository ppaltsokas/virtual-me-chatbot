# getwd()
# setwd("C:\\Users\\user\\Desktop\\ΕΑΠ\\DAMA 51\\My_assignments\\HW4\\R_scripts")

################ Topic 3a ################
cat("\n################ Topic 3a ################\n")
fruits_df_orig <- read.csv2("fruits.csv",header=TRUE, sep=";")

# Inspect the dataset by checking the dimensions, column names, structure, summary, head, and tail.
cat("------- Dimensions: -------\n")
print(dim(fruits_df_orig))
cat("\n------- Columns Names: -------\n")
print(names(fruits_df_orig))
cat("\n------- Structure: -------\n")
print(str(fruits_df_orig))
cat("\n------- Summary: -------\n")
print(summary(fruits_df_orig))
cat("\n------- Head: -------\n")
print(head(fruits_df_orig,5))
cat("\n------- Tail: -------\n")
print(tail(fruits_df_orig,5))

# Make a copy of the original dataset
fruits_df <- fruits_df_orig
# Set the row names according to the fruit names
rownames(fruits_df) <- fruits_df[,1]
# Remove the first column with the fruit names
fruits_df <- fruits_df[,2:14]
# Scale the data and store in a new dataframe
fruits_df_scaled <- data.frame(scale(fruits_df))
# Calculate the median value of the Sugars_g attribute
sugars_median <- median(fruits_df_scaled$Sugars_g)
cat("\nThe median value of the 'Sugars_g' attribute is", (round(sugars_median,3)))
# Calculate the maximum value of the Energy_kcal attribute
energy_max <- max(fruits_df_scaled$Energy_kcal)
cat("\nThe maximum value of the 'Energy_kcal' attribute is",(round(energy_max,3)))
# Calculate the minimum value of the Water_g attribute
water_min <- min(fruits_df_scaled$Water_g)
cat("\nThe minimum value of the 'water_g' attribute is", (round(water_min,3)))

################ Topic 3b ################
cat("\n\n################ Topic 3b ################")
# Compute the Euclidean distance matrix from the df data and save it as a matrix
dissim_dist_matrix <- as.matrix(dist(fruits_df_scaled, method = 'euclidean'))
# Print out the required Euclidean distances
cat("\nEuclidean Distances:\nApple - Orange :",round(dissim_dist_matrix["Apple","Orange"],3)
    ,"\nBanana - Peach :",round(dissim_dist_matrix["Banana","Peach"],3)
    ,"\nLemon - Mango  :",round(dissim_dist_matrix["Lemon","Mango"],3))
# Find the minimum distance from the other fruits to Pear. Exclude the distance from Pear to itself
min_dist_Pear <- min(dissim_dist_matrix["Pear", -which(colnames(dissim_dist_matrix) == "Pear")])
cat("\n\nThe distance of the fruit closest to Pear is:",round(min_dist_Pear,3))
# Find the fruit which is closest to Pear
closest_fruit_Pear <- colnames(dissim_dist_matrix)[which.min(dissim_dist_matrix["Pear", -which(colnames(dissim_dist_matrix) == "Pear")])]
cat("\nThe fruit closest to Pear is:", closest_fruit_Pear)

################ Topic 3c ################
cat("\n\n################ Topic 3c ################")
cat("\n Dendrograms")
# Calculate the Euclidean distance matrix
dissim_dist <- dist(fruits_df_scaled, method = 'euclidean')
# Perform hierarchical clustering using complete linkage
complete_hclust <- hclust(dissim_dist, method = "complete")
# Perform hierarchical clustering using single linkage
single_hclust <- hclust(dissim_dist, method = "single")
# Plot the dendrogram for the complete linkage clustering
plot(complete_hclust, main = "Complete Linkage Dendrogram", sub="", xlab="")
# Plot the dendrogram for the single linkage clustering
plot(single_hclust, main = "Single Linkage Dendrogram", sub="", xlab="")

################ Topic 3d ################
cat("\n\n################ Topic 3d ################")
# Cut the dendrogram into 5 clusters for complete linkage
clusters_complete <- cutree(complete_hclust, k = 5)

# Plot the dendrogram with 5 clusters
plot(complete_hclust, main = "Complete Linkage Dendrogram (5 Clusters)", sub="", xlab="")

# Optional Visualization
# rect.hclust(complete_hclust, k = 5, border = 2:6)

# Extract the names of the fruits in cluster 1
cluster_1 <- names(clusters_complete[clusters_complete == 1])

# Print the names of the fruits in cluster 1
print(cluster_1)

################ Topic 3e ################
cat("\n\n################ Topic 3e ################")
# Store all the desired fruits in a vector
desired_fruits <- c("Orange", "Grapefruit", "Nectarine", "Lemon", "Mandarin")

# Initialize the index i and maximum cluster number
i <- 1
max_clstr_number <- NULL
# Starting with 1 cluster that contains all the fruits.
while(TRUE) {
  # Cut the dendrogram into i clusters for single linkage
  clusters <- cutree(single_hclust, k = i)

  # Check if all desired fruits are in the first cluster
  if (all(desired_fruits %in% names(clusters[clusters == 1]))) {
    i<- i + 1 #while the desired fruits are all in the same cluster, increment 'i'.
  }
  else{
    max_clstr_number <- i-1 # When the desired fruits are not in the same cluster anymore, it means that 
    # the last value of k was the maximum number of clusters for which all of the Citrus fruits belong in 
    # the same cluster
    cat("\nThe maximum number of clusters (single-linkage) so that Orange, Grapefruit, Nectarine, Lemon, and Mandarin are in the same cluster is:", max_clstr_number)
    break
  }

}


# Initialize the index i and minimum height
i <- 1
min_height <- NULL


while (TRUE) {
  # Cut the dendrogram at the height corresponding to the ith position to form clusters
  clusters <- cutree(single_hclust, h = single_hclust$height[i])
  # Check if all desired fruits are in the first cluster
  if (all(desired_fruits %in% names(clusters[clusters == 1]))) {
    # If so, store the height at this point as the minimum height
    min_height <- single_hclust$height[i]
    break
  }
  # If not all desired fruits are in the first cluster, increment 'i' to test the next height
  else{
    i <- i + 1
  }
}

cat("\nThe minimum height at which Orange, Grapefruit, Nectarine, Lemon, and Mandarin are in the same cluster is:", round(min_height,3))

plot(single_hclust, main = "Single Linkage Dendrogram", sub="", xlab="")
abline(h = min_height, col = "red")
text(x = 14, y = min_height + 0.2, labels = "Minimum height in order for the citrus fruits to belong in the same cluster", pos = 4, col = "red", cex = 0.8)
text(x = 0, y = min_height + 0.1, labels = "2.637229", pos = 4, col = "red", cex=0.7)

