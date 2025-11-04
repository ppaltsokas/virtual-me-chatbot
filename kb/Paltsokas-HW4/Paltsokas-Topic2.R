# setwd("C:\\Users\\user\\Desktop\\ΕΑΠ\\DAMA 51\\My_assignments\\HW4\\R_scripts")
# getwd()

################ Topic 2a ################
cat("\n################ Topic 2a ################\n")
library("readxl")
# Read the excel using the "readxl" library
SOIL_DATA_GR <- read_excel("SOIL DATA GR.xlsx")

# Set the seed for reproducibility
set.seed(123)
# [i] Remove the first column of the dataframe and store it in df_soil
df_soil <- SOIL_DATA_GR[2:17]
# [ii] Check the summary() of the df to manually spot NA values
summary(df_soil)
# Print the number and location of the NA value
na_counts <- colSums(is.na(df_soil))
cat("There is ", sum(is.na(df_soil)), "missing value in the '", names(na_counts[na_counts > 0]), "'column")
# [iii] Remove the NA value
df_soil <- na.omit(df_soil)
# Confirm there are no NA values
sum(is.na(df_soil))

# [iv]
# Scale the data and store as data frame
scaled_df <- as.data.frame(scale(df_soil))
# Print out the maximum value of the pH attribute in the scaled data
maxpH <- max(scaled_df$pH)
cat("\nThe maximum value of the pH attribute in the scaled data is: ",round(maxpH,3) )
# Print out the minimum value of the Sand % attribute in the scaled data
minSandPct <- min(scaled_df$"Sand %")
cat("\nThe minimum value of the Sand % attribute in the scaled data is: ",round(minSandPct,3) )
# Print out the median value of the Clay % attribute in the scaled data
medianClayPct <- median(scaled_df$"Clay %")
cat("\nThe median value of the Clay % attribute in the scaled data is: ",round(medianClayPct,3) )


################ Topic 2b ################
cat("\n\n################ Topic 2b ################\n")
# Load the scaled dataset
kmdata <- scaled_df
# Initialize a numeric vector to store the within cluster sum of squares for each value of k
wss <- numeric(6)
# Set the seed for reproducibility
set.seed(123)
# Perform k-means clustering for 1 through 6 clusters
for (k in 1:6) wss[k] <- sum(kmeans(kmdata, centers=k)$withinss)
# Plot the within cluster sum of squares for each number of clusters
plot(1:6, wss, type="b", xlab="Number of Clusters", ylab="Within Sum of Squares")
cat('The visual results are inconclusive since there is not a clear "elbow". That is why we will use the silhouette method.')

################ Topic 2c ################
cat("\n\n################ Topic 2c ################\n")
# install.packages("cluster")
library(cluster)
# Define a function that calculates the silhouette scores
silhouette_score <- function(k){
  set.seed(123) # Set a random seed for reproducibility purposes
  cl <- kmeans(kmdata, centers=k) # Perform k-means clustering for the specified amount of clusters
  ss <- silhouette(cl$cluster, dist(kmdata)) # Calculate the silhouette scores for each point in kmdata based on the clustering result
  mean(ss[,3]) # Extract the silhouette widths and return their mean
}

# Set the desired number of clusters
n = 6
# Initialize a numeric vector to store the silhouette scores. We choose size of n-1 because the silhouette for k=1 is not computed since the silhouette method requires at least two clusters to calculate the similarity of points within their own cluster compared to neighboring clusters
sscores <- numeric(n-1)
# Loop through 1 to 6 clusters, compute the silhouette scores in each case and store them in the created vector
for (k in 2:n) sscores[k] <- silhouette_score(k)
# Extract the index of the highest value in sscores and store it
optimal_k <- which.max(sscores)
# Capture the best score to use it later in the plot
best_score <- sscores[optimal_k]
# Plot the average silhouette score per cluster number
plot(2:n, sscores[-1], type = "b", xlab = "Number of Clusters k", ylab = "Average Silhouette Score")
# Add a vertical line at the optimal number of clusters
abline(v=optimal_k, col="red", lty=2)
# Annotate the plot with the optimal k value
text(optimal_k, best_score, labels=paste("k=", optimal_k), pos=4, col="red")
cat("According to the plot, the optimal number of clusters is: ", optimal_k)

################ Topic 2d ################
cat("\n\n################ Topic 2d ################\n")
library(factoextra)
# Set the seed for reproducibility
set.seed(123)
# Perform k-means algorithm for the optimal number of clusters
kmresults <- kmeans(kmdata, centers=3)
# Visualize the clustering results using the factoextra package
viz1 <- fviz_cluster(kmresults, data=kmdata,
     palette = "Twilight",
     geom = "point",
     ellipse.type = "convex",
     ggtheme = theme_bw(),
     alpha=0.4
)
print(viz1)

cat("The value of pH for the center of cluster 1 is", round(kmresults$centers[1,"pH"],3))
cat("\nThe calue of Mg ppm for the center of cluster 2 is", round(kmresults$centers[2,"Mg ppm"],3))
cat("\nThe data instance of row 100 has been assigned to cluster", kmresults$cluster[100])
cat("\nThe data instance of row 101 has been assigned to cluster", kmresults$cluster[101])

################ Topic 2e ################
cat("\n\n################ Topic 2e ################\n")
# Create a subset of kmdata containing only the first four columns
subset_df <- kmdata[,1:4]
# Set the seed for reproducibility
set.seed(123)
# Perform k-means algorithm for 3 clusters
subset_kmresults <- kmeans(subset_df, centers=3)
# Visualize the clustering results using the factoextra package
viz2 <- fviz_cluster(subset_kmresults, data=subset_df,
                     palette = "Twilight",
                     geom = "point",
                     ellipse.type = "convex",
                     ggtheme = theme_bw(),
                     alpha=0.4
)
print(viz2)
cat("Better clustering result is achieved in approach: e")


cat("\n\n\n################ Extra ################\n")
# Calculate silhouette width for the original clustering
sscore_original <- silhouette(kmresults$cluster, dist(kmdata))
avg_sscore_original <- mean(sscore_original[, 3])
fviz_original <- fviz_silhouette(sscore_original) + ggtitle("Original Data Silhouette Plot")
print(fviz_original)
# Calculate silhouette width for the subset clustering
sscore_subset <- silhouette(subset_kmresults$cluster, dist(subset_df))
avg_sscore_subset <- mean(sscore_subset[, 3])
fviz_subset <- fviz_silhouette(sscore_subset) + ggtitle("Subset Data Silhouette Plot")
print(fviz_subset)
# Compare the average silhouette widths
cat("\nAverage Silhouette Width - Original Data: ", round(avg_sscore_original,3), "\n")
cat("Average Silhouette Width - Subset Data: ", round(avg_sscore_subset,3), "\n")
cat("\nA higher average silhouette score for the subset data, combined with our visual analysis of the silhouette plot, suggests that we have better clustering for the subset data (case e).")
