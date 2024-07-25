# Clustering: Separating the data into sub-groups
# Considered unsupervised since no prediction, just trying to understand the data

penguins <- read_csv("penguins.csv")
penguins

bill_length_standardized <- (penguins$bill_length_mm - mean(penguins$bill_length_mm)) / 
  sd(penguins$bill_length_mm)
flipper_length_standardized <- (penguins$flipper_length_mm - mean(penguins$flipper_length_mm)) / 
  sd(penguins$flipper_length_mm)

penguins_standardized <- tibble(bill_length_standardized, flipper_length_standardized)
penguins_standardized

ggplot(penguins_standardized,
       aes(x = flipper_length_standardized,
           y = bill_length_standardized)) +
  geom_point() +
  xlab("Flipper Length (standardized)") +
  ylab("Bill Length (standardized)") +
  theme(text = element_text(size = 12))

# Measure of quality of a cluster:
# Within-cluster sum-of-squared-distances (WSSD)
# Find the centre of the cluster by taking average of coordinates.
# Then, take sum of squared distances. 

# K-means algorithm is iterative. Randomly assign equal number of points to K clusters.
# Compute the centre of each cluster. 
# Reassign each data point to the cluster with the nearest center.
# Repeat until cluster assignments do not change.

install.packages("tidyclust")
library(tidyclust)

# Ignore previous standardization and use this one
kmeans_recipe <- recipe(~ ., data=penguins) |>
  step_scale(all_predictors()) |>
  step_center(all_predictors())
kmeans_recipe

# K-means with 3 clusters
kmeans_spec <- k_means(num_clusters = 3) |>
  set_engine("stats")
kmeans_spec

kmeans_fit <- workflow() |>
  add_recipe(kmeans_recipe) |>
  add_model(kmeans_spec) |>
  fit(data = penguins)
kmeans_fit

clustered_data <- kmeans_fit |>
  augment(penguins)
clustered_data

# Visualize the clusters
cluster_plot <- ggplot(clustered_data,
                       aes(x = flipper_length_mm,
                           y = bill_length_mm,
                           color = .pred_cluster),
                       size = 2) +
  geom_point() +
  labs(x = "Flipper Length",
       y = "Bill Length",
       color = "Cluster") +
  scale_color_manual(values = c("steelblue",
                                "darkorange",
                                "goldenrod1")) +
  theme(text = element_text(size = 12))

cluster_plot

glance(kmeans_fit)
# tot.withinss represents the WSSD

# Calculate WSSD for a variety of K's
penguin_clust_ks <- 
  tibble(k = 1:9) |>
  mutate(
    kclust = map(k, ~kmeans(penguins, .x)),
    tidied = map(kclust, tidy),
    glanced = map(kclust, glance),
    augmented = map(kclust, augment, penguins)
  )
penguin_clust_ks

clusters <- 
  penguin_clust_ks %>%
  unnest(cols = c(tidied))
clusters

assignments <- 
  penguin_clust_ks %>% 
  unnest(cols = c(augmented))
assignments

clusterings <- 
  penguin_clust_ks %>%
  unnest(cols = c(glanced))
clusterings

kmeans_results <- clusterings |>
  select(k, tot.withinss) |>
  rename(num_clusters = k, total_WSSD = tot.withinss)
kmeans_results

# Plot the elbow plot
elbow_plot <- ggplot(kmeans_results, aes(x = num_clusters, y = total_WSSD)) +
  geom_point() +
  geom_line() +
  xlab("K") +
  ylab("Total within-cluster sum of squares") +
  scale_x_continuous(breaks = 1:9) +
  theme(text = element_text(size = 12))
elbow_plot

######### Exercises ##########

library(forcats)

# Clustering is ideal for identifying sub-groups or segmenting customers.
beers <- read.csv("data-science-a-first-intro-worksheets-main/worksheet_clustering/data/beers.csv")
beers

beer_plot <- ggplot(beers, aes(x = ibu, y = abv)) + 
  geom_point() + 
  labs(x = "International bittering units", y = "Alcoholic content")
beer_plot

clean_beer <- beers |>
  drop_na(ibu)

clean_beer_plot <- ggplot(clean_beer, aes(x = ibu, y = abv)) + 
  geom_point() + 
  labs(x = "International bittering units", y = "Alcoholic content")
clean_beer_plot

kmeans_recipe <- recipe(~ ibu + abv, data = clean_beer) |>
  step_scale(all_predictors()) |>
  step_center(all_predictors())
kmeans_recipe

kmeans_spec <- k_means(num_clusters = 2) |>
  set_engine("stats")
kmeans_spec

kmeans_fit <- workflow() |>
  add_recipe(kmeans_recipe) |>
  add_model(kmeans_spec) |>
  fit(data = clean_beer)
kmeans_fit

labelled_beer <- kmeans_fit |>
  augment(clean_beer)
labelled_beer

# Visualize the clusters
cluster_plot <- ggplot(labelled_beer, aes(x = ibu, y = abv, color = .pred_cluster)) + 
  geom_point() + 
  labs(x = "International bittering units", y = "Alcoholic content")
cluster_plot

# How to choose K?
# Perform clustering for a variety of possible Ks. 
# Choose the one where within-cluster sum of squares distance starts to decrease less.

clustering_stat <- glance(kmeans_fit)
clustering_stat

totalWSSD <- clustering_stat |>
  select(tot.withinss) |>
  pull()
totalWSSD

# ALL OF THIS CODE DOESN'T WORK - NAs PRODUCED IN RESULT TABLE
beer_ks <- tibble(num_clusters = 1:9)

kmeans_spec_tune <- k_means(num_clusters = tune()) |>
  set_engine("stats", nstart = 10)

kmeans_tuning_stats <- workflow() |>
  add_recipe(kmeans_recipe) |>
  add_model(kmeans_spec_tune) |>
  tune_cluster(resamples = apparent(clean_beer), grid = beer_ks) |>
  collect_metrics()
kmeans_tuning_stats


# The following is alternative method I found online but I don't think that 
# it does any of recipe and model specification calls
# https://www.tidymodels.org/learn/statistics/k-means/

clean_beer_red <- clean_beer |>
  select(ibu, abv)

beer_ks <- 
  tibble(k = 1:9) |>
  mutate(
    kclust = map(k, ~kmeans(clean_beer_red, .x)),
    tidied = map(kclust, tidy),
    glanced = map(kclust, glance),
    augmented = map(kclust, augment, clean_beer_red)
  )
beer_ks

clusters <- 
  beer_ks %>%
  unnest(cols = c(tidied))
clusters

assignments <- 
  beer_ks %>% 
  unnest(cols = c(augmented))
assignments

clusterings <- 
  beer_ks %>%
  unnest(cols = c(glanced))
clusterings

kmeans_results <- clusterings |>
  select(k, tot.withinss) |>
  rename(num_clusters = k, total_WSSD = tot.withinss)
kmeans_results

# Plot the elbow plot
elbow_plot <- ggplot(kmeans_results, aes(x = num_clusters, y = total_WSSD)) +
  geom_point() +
  geom_line() +
  xlab("K") +
  ylab("Total within-cluster sum of squares") +
  scale_x_continuous(breaks = 1:9) +
  theme(text = element_text(size = 12))
elbow_plot

