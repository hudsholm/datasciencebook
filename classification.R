# Link for GitHub data
# https://github.com/UBC-DSCI/introduction-to-datascience/tree/main

library(tidyverse)

# Reading in and cleaning the data
cancer <- read.csv("wdbc.csv")
cancer <- cancer |>
  mutate(Class = as_factor(Class)) |>
  mutate(Class = fct_recode(Class, "Malignant" = "M", "Benign" = "B"))
glimpse(cancer)

# Percent malignant and benign
num_obs <- nrow(cancer)
cancer |>
  group_by(Class) |>
  summarize(
    count = n(),
    percentage = n() / num_obs * 100
  )

# Relationship between perimeter and concavity
perim_concav <- cancer |>
  ggplot(aes(x = Perimeter, y = Concavity, color = Class)) +
  geom_point(alpha = 0.6) +
  labs(x = "Perimeter (standardized)",
       y = "Concavity (standardized)",
       color = "Diagnosis") +
  scale_color_manual(values = c("darkorange", "steelblue")) +
  theme(text = element_text(size = 12))
perim_concav

# K-nearest neighbours approach
# Find the most similar observations to the one we want to predict.
# What is the dominating categorical label of the nearest neighbours.

# Use Euclidean distance from new observation to determine the k-nearest
# neighbours.

new_obs_Perimeter <- 0
new_obs_Concavity <- 3.5
cancer |>
  select(ID, Perimeter, Concavity, Class) |>
  mutate(dist_from_new = sqrt((Perimeter - new_obs_Perimeter)^2 +
                                (Concavity - new_obs_Concavity)^2)) |>
  slice_min(dist_from_new, n = 5) # take the 5 rows of minimum distance from new

# Same process if more than 2 explanatory variables
new_obs_Symmetry <- 1
cancer |>
  select(ID, Perimeter, Concavity, Symmetry, Class) |>
  mutate(dist_from_new = sqrt((Perimeter - new_obs_Perimeter)^2 +
                                (Concavity - new_obs_Concavity)^2 +
                                (Symmetry - new_obs_Symmetry)^2)) |>
  slice_min(dist_from_new, n = 5) # take the 5 rows of minimum distance from new

# Instead of doing this manually, we can use a package.
library(tidymodels)
install.packages("kknn")

# Simplified dataset
cancer_train <- cancer |>
  select(Class, Perimeter, Concavity)
cancer_train

# K-nearest neighbours - model specification
# Note: rectangular means that each neighbouring point has the same weight
knn_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = 5) |>
  set_engine("kknn") |>
  set_mode("classification")
knn_spec

# K-nearest neighbours - model fit
knn_fit <- knn_spec |>
  fit(Class ~ Perimeter + Concavity, data = cancer_train)
knn_fit

# K-nearest neighbours - prediction of a new observation
new_obs <- tibble(Perimeter = 0, Concavity = 3.5)
predict(knn_fit, new_obs)

# Standarizing variables through centering and scaling
# This ensures that all predictors have a mean of 0 and sd of 1
unscaled_cancer <- read_csv("wdbc_unscaled.csv") |>
  mutate(Class = as_factor(Class)) |>
  mutate(Class = fct_recode(Class, "Benign" = "B", "Malignant" = "M")) |>
  select(Class, Area, Smoothness)
unscaled_cancer

# Standarize using Tidymodels
uc_recipe <- recipe(Class ~ ., data = unscaled_cancer)
uc_recipe

uc_recipe <- uc_recipe |>
  step_scale(all_predictors()) |>
  step_center(all_predictors()) |>
  prep()
uc_recipe

# Apply the recipe to standardize
scaled_cancer <- bake(uc_recipe, unscaled_cancer)
scaled_cancer

# Imbalance - what if many more data points for one particular label?
# Consider the case where malignant tumors were rare.
rare_cancer <- bind_rows(
  filter(cancer, Class == "Benign"),
  cancer |> filter(Class == "Malignant") |> slice_head(n = 3)
) |>
  select(Class, Perimeter, Concavity)

rare_plot <- rare_cancer |>
  ggplot(aes(x = Perimeter, y = Concavity, color = Class)) +
  geom_point(alpha = 0.5) +
  labs(x = "Perimeter (standardized)",
       y = "Concavity (standardized)",
       color = "Diagnosis") +
  scale_color_manual(values = c("darkorange", "steelblue")) +
  theme(text = element_text(size = 12))

rare_plot

# Since only three malignant, if K = 7, the preductor will always be benign.
# We can re-balance the data by oversampling the rare cases. 
install.packages("themis")
library(themis)

ups_recipe <- recipe(Class ~ ., data = rare_cancer) |>
  step_upsample(Class, over_ratio = 1, skip = FALSE) |>
  prep()
ups_recipe

# After baking it, we see that the class numbers are equal now.
upsampled_cancer <- bake(ups_recipe, rare_cancer)

upsampled_cancer |>
  group_by(Class) |>
  summarize(n = n())

# How to handle missing data?
missing_cancer <- read_csv("wdbc_missing.csv") |>
  select(Class, Radius, Texture, Perimeter) |>
  mutate(Class = as_factor(Class)) |>
  mutate(Class = fct_recode(Class, "Malignant" = "M", "Benign" = "B"))
missing_cancer

# Option 1: Drop missing values - easier when given larger dataset
no_missing_cancer <- missing_cancer |> drop_na()
no_missing_cancer

# Option 2: Mean imputation
impute_missing_recipe <- recipe(Class ~ ., data = missing_cancer) |>
  step_impute_mean(all_predictors()) |>
  prep()
impute_missing_recipe

imputed_cancer <- bake(impute_missing_recipe, missing_cancer)
imputed_cancer

# Example workflow:

# load the unscaled cancer data
# and make sure the response variable, Class, is a factor
unscaled_cancer <- read_csv("wdbc_unscaled.csv") |>
  mutate(Class = as_factor(Class)) |>
  mutate(Class = fct_recode(Class, "Malignant" = "M", "Benign" = "B"))

# create the K-NN model
knn_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = 7) |>
  set_engine("kknn") |>
  set_mode("classification")

# create the centering / scaling recipe
uc_recipe <- recipe(Class ~ Area + Smoothness, data = unscaled_cancer) |>
  step_scale(all_predictors()) |>
  step_center(all_predictors())

knn_fit <- workflow() |>
  add_recipe(uc_recipe) |>
  add_model(knn_spec) |>
  fit(data = unscaled_cancer)
knn_fit

# Two new unstandardized observations
new_observation <- tibble(Area = c(500, 1500), Smoothness = c(0.075, 0.1))
prediction <- predict(knn_fit, new_observation)
prediction

# Prediction map visualization
# create the grid of area/smoothness vals, and arrange in a data frame
are_grid <- seq(min(unscaled_cancer$Area),
                max(unscaled_cancer$Area),
                length.out = 100)
smo_grid <- seq(min(unscaled_cancer$Smoothness),
                max(unscaled_cancer$Smoothness),
                length.out = 100)
asgrid <- as_tibble(expand.grid(Area = are_grid,
                                Smoothness = smo_grid))

# use the fit workflow to make predictions at the grid points
knnPredGrid <- predict(knn_fit, asgrid)

# bind the predictions as a new column with the grid points
prediction_table <- bind_cols(knnPredGrid, asgrid) |>
  rename(Class = .pred_class)

# plot:
# 1. the colored scatter of the original data
# 2. the faded colored scatter for the grid points
wkflw_plot <-
  ggplot() +
  geom_point(data = unscaled_cancer,
             mapping = aes(x = Area,
                           y = Smoothness,
                           color = Class),
             alpha = 0.75) +
  geom_point(data = prediction_table,
             mapping = aes(x = Area,
                           y = Smoothness,
                           color = Class),
             alpha = 0.02,
             size = 5) +
  labs(color = "Diagnosis",
       x = "Area",
       y = "Smoothness") +
  scale_color_manual(values = c("darkorange", "steelblue")) +
  theme(text = element_text(size = 12))

wkflw_plot

# Split the data into training and test set
# Training builds the classifier
# Test evaluates the performance of the classifier

# Prediction accuracy = # of correct predictions / total predictions
# Confusion matrix gives more detail into our predictions
# This contains true positives, true negatives, false positives, false negatives

# Precision = # of correct positive predictions / total positive predictions
# Recall = # of correct positive predictions / total positive test set obs

# Set the seed when using randomness but want to ensure reproducibility
set.seed(1)

# Load data
cancer <- read_csv("wdbc_unscaled.csv") |>
  mutate(Class = as_factor(Class))  |>
  mutate(Class = fct_recode(Class, "Malignant" = "M", "Benign" = "B"))

# create scatter plot of tumor cell concavity versus smoothness,
# labeling the points be diagnosis class
perim_concav <- cancer |>
  ggplot(aes(x = Smoothness, y = Concavity, color = Class)) +
  geom_point(alpha = 0.5) +
  labs(color = "Diagnosis") +
  scale_color_manual(values = c("darkorange", "steelblue")) +
  theme(text = element_text(size = 12))

perim_concav

# Typically want larger sample for training than testing
# Here, 75% training, 25% testing
cancer_split <- initial_split(cancer, prop = 0.75, strata = Class)
cancer_train <- training(cancer_split)
cancer_test <- testing(cancer_split)

# We can see that the percentages of the response variable are still
# representative of the complete dataset
cancer_proportions <- cancer_train |>
  group_by(Class) |>
  summarize(n = n()) |>
  mutate(percent = 100*n/nrow(cancer_train))
cancer_proportions

# Standardize the training data
cancer_recipe <- recipe(Class ~ Smoothness + Concavity, data = cancer_train) |>
  step_scale(all_predictors()) |>
  step_center(all_predictors())

# Training the classifier
knn_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = 3) |>
  set_engine("kknn") |>
  set_mode("classification")

knn_fit <- workflow() |>
  add_recipe(cancer_recipe) |>
  add_model(knn_spec) |>
  fit(data = cancer_train)

knn_fit

# Predict labels in the test set
cancer_test_predictions <- predict(knn_fit, cancer_test) |>
  bind_cols(cancer_test)

cancer_test_predictions

# Accuracy of our predictions
cancer_test_predictions |>
  metrics(truth = Class, estimate = .pred_class) |>
  filter(.metric == "accuracy")

# Precision and recall of our predictions
# First, check the order of the Class variable cateogries
cancer_test_predictions |> pull(Class) |> levels()
# Malignant is first and that represent our positive label so 
# in our precision and recall fn calls, we specify event_level = "first"
cancer_test_predictions |>
  precision(truth = Class, estimate = .pred_class, event_level = "first")
cancer_test_predictions |>
  recall(truth = Class, estimate = .pred_class, event_level = "first")

# Confusion matrix
confusion <- cancer_test_predictions |>
  conf_mat(truth = Class, estimate = .pred_class)
confusion

# How do we know which value of k to choose? Cross-validation
# create the 75/25 split of the training data into training and validation
cancer_split <- initial_split(cancer_train, prop = 0.75, strata = Class)
cancer_subtrain <- training(cancer_split)
cancer_validation <- testing(cancer_split)

# recreate the standardization recipe from before
# (since it must be based on the training data)
cancer_recipe <- recipe(Class ~ Smoothness + Concavity,
                        data = cancer_subtrain) |>
  step_scale(all_predictors()) |>
  step_center(all_predictors())

# fit the knn model (we can reuse the old knn_spec model from before)
knn_fit <- workflow() |>
  add_recipe(cancer_recipe) |>
  add_model(knn_spec) |>
  fit(data = cancer_subtrain)

# get predictions on the validation data
validation_predicted <- predict(knn_fit, cancer_validation) |>
  bind_cols(cancer_validation)

# compute the accuracy
acc <- validation_predicted |>
  metrics(truth = Class, estimate = .pred_class) |>
  filter(.metric == "accuracy") |>
  select(.estimate) |>
  pull()

acc

# c-fold cross-validation
# Each observation is in the validation set one time
# Split training data into C even chunks.
# Iteratively use one chunk as validation while C-1 other chunks as training

cancer_vfold <- vfold_cv(cancer_train, v = 5, strata = Class)
cancer_vfold

# recreate the standardization recipe from before
# (since it must be based on the training data)
cancer_recipe <- recipe(Class ~ Smoothness + Concavity,
                        data = cancer_train) |>
  step_scale(all_predictors()) |>
  step_center(all_predictors())

# fit the knn model (we can reuse the old knn_spec model from before)
knn_fit <- workflow() |>
  add_recipe(cancer_recipe) |>
  add_model(knn_spec) |>
  fit_resamples(resamples = cancer_vfold)

knn_fit

# Mean and standard error of the classifier's validation accuracy across folds
knn_fit |>
  collect_metrics()

# Typically, we choose between 5 and 10 folds
cancer_vfold <- vfold_cv(cancer_train, v = 10, strata = Class)

vfold_metrics <- workflow() |>
  add_recipe(cancer_recipe) |>
  add_model(knn_spec) |>
  fit_resamples(resamples = cancer_vfold) |>
  collect_metrics()

vfold_metrics

# How to try out many values of K
# First, specify neighbours = tune in model specification
knn_spec <- nearest_neighbor(weight_func = "rectangular",
                             neighbors = tune()) |>
  set_engine("kknn") |>
  set_mode("classification")
# Then, create sequence of K values that we want to evaluate
k_vals <- tibble(neighbors = seq(from = 1, to = 100, by = 5))

# Use tune_grid instead of fit_resamples
knn_results <- workflow() |>
  add_recipe(cancer_recipe) |>
  add_model(knn_spec) |>
  tune_grid(resamples = cancer_vfold, grid = k_vals) |>
  collect_metrics()

accuracies <- knn_results |>
  filter(.metric == "accuracy")
accuracies

accuracy_vs_k <- ggplot(accuracies, aes(x = neighbors, y = mean)) +
  geom_point() +
  geom_line() +
  labs(x = "Neighbors", y = "Accuracy Estimate") +
  theme(text = element_text(size = 12))
accuracy_vs_k

# We can use the graph or pull directly the maximum from the table
best_k <- accuracies |>
  arrange(desc(mean)) |>
  head(1) |>
  pull(neighbors)
best_k

# Underfitting (K large): If the model isn't influenced enough by the training data
# Here the classification boundaries will be very smooth leading to simpler model.

# Overfitting (K small): If the model is influenced too much by the training data
# Here the classification boundaries will be very sharp leading to complex model. 

# Now, we have built the model and should evaluate on the test data
# First, retrain classifer on entire training data
cancer_recipe <- recipe(Class ~ Smoothness + Concavity, data = cancer_train) |>
  step_scale(all_predictors()) |>
  step_center(all_predictors())

knn_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = best_k) |>
  set_engine("kknn") |>
  set_mode("classification")

knn_fit <- workflow() |>
  add_recipe(cancer_recipe) |>
  add_model(knn_spec) |>
  fit(data = cancer_train)

knn_fit

# Now, predictions on test set
cancer_test_predictions <- predict(knn_fit, cancer_test) |>
  bind_cols(cancer_test)

cancer_test_predictions |>
  metrics(truth = Class, estimate = .pred_class) |>
  filter(.metric == "accuracy")
cancer_test_predictions |>
  precision(truth = Class, estimate = .pred_class, event_level="first")
cancer_test_predictions |>
  recall(truth = Class, estimate = .pred_class, event_level="first")
confusion <- cancer_test_predictions |>
  conf_mat(truth = Class, estimate = .pred_class)
confusion


# What predictors to include in KNN model?
# i) Best subset selection - Try all possible subsets of predictors and 
#    pick the set that results in the best classifier.
# ii) Forward selection - Start with no predictors. Add one predictor to the 
#     model at a time and choose model with the highest cross-validation accuracy

# Forward selection
cancer_subset <- cancer |>
  select(Class,
         Smoothness,
         Concavity,
         Perimeter)

names <- colnames(cancer_subset |> select(-Class))

cancer_subset

# Model with all the predictors
example_formula <- paste("Class", "~", paste(names, collapse="+"))
example_formula

# create an empty tibble to store the results
accuracies <- tibble(size = integer(),
                     model_string = character(),
                     accuracy = numeric())

# create a model specification
knn_spec <- nearest_neighbor(weight_func = "rectangular",
                             neighbors = tune()) |>
  set_engine("kknn") |>
  set_mode("classification")

# create a 5-fold cross-validation object
cancer_vfold <- vfold_cv(cancer_subset, v = 5, strata = Class)

# store the total number of predictors
n_total <- length(names)

# stores selected predictors
selected <- c()

# for every size from 1 to the total number of predictors
for (i in 1:n_total) {
  # for every predictor still not added yet
  accs <- list()
  models <- list()
  for (j in 1:length(names)) {
    # create a model string for this combination of predictors
    preds_new <- c(selected, names[[j]])
    model_string <- paste("Class", "~", paste(preds_new, collapse="+"))
    
    # create a recipe from the model string
    cancer_recipe <- recipe(as.formula(model_string),
                            data = cancer_subset) |>
      step_scale(all_predictors()) |>
      step_center(all_predictors())
    
    # tune the K-NN classifier with these predictors,
    # and collect the accuracy for the best K
    acc <- workflow() |>
      add_recipe(cancer_recipe) |>
      add_model(knn_spec) |>
      tune_grid(resamples = cancer_vfold, grid = 10) |>
      collect_metrics() |>
      filter(.metric == "accuracy") |>
      summarize(mx = max(mean))
    acc <- acc$mx |> unlist()
    
    # add this result to the dataframe
    accs[[j]] <- acc
    models[[j]] <- model_string
  }
  jstar <- which.max(unlist(accs))
  accuracies <- accuracies |>
    add_row(size = i,
            model_string = models[[jstar]],
            accuracy = accs[[jstar]])
  selected <- c(selected, names[[jstar]])
  names <- names[-jstar]
}
accuracies

####### Exercises - Classification 1 ##########

library(repr)
options(repr.matrix.max.rows = 6)

# 0.1 = C - Training set consists of observations where we do know the true class.
# 0.2 = B - Classification problem is when we have new observations and wish to 
#           determine whether it will be a success or failure. 
# Question 1
cancer <- read.csv("data-science-a-first-intro-worksheets-main/worksheet_classification1/data/clean-wdbc-data.csv")
cancer
# Predicting area is not a classification problem since it is not a categorical variable. 

cancer <- cancer |>
  mutate(Class = as_factor(Class))

cancer_plot <- ggplot(cancer, aes(x = Symmetry, y = Radius, colour = Class)) + 
  geom_point(alpha = 0.8) + 
  labs(x = "Symmetry", y ="Radius", colour = "Diagnosis")

# According to the plot, a new observation with symmetry of 1 and radius of 1 would
# most likely be malignant.

# Calculating distance
ax <- slice(cancer, 1) |> 
  pull(Symmetry)
ay <- slice(cancer, 1) |>
  pull(Radius)
bx <- slice(cancer, 2) |>
  pull(Symmetry)
by <- slice(cancer, 2) |>
  pull(Radius)

distance <- sqrt((ax-bx)^2 + (ay-by)^2)

az <- slice(cancer, 1) |>
  pull(Concavity)
bz <- slice(cancer, 2) |>
  pull(Concavity)

distance2 <- sqrt((ax-bx)^2 + (ay-by)^2 + (az-bz)^2)

# Creating coordinate variables
point_a <- slice(cancer, 1) |>
  select(Symmetry, Radius, Concavity) |>
  as.numeric()
point_b <- slice(cancer, 2) |>
  select(Symmetry, Radius, Concavity) |>
  as.numeric()

dif_square <- (point_a - point_b)^2
dif_sum <- sum(dif_square)
root_dif_sum <- sqrt(dif_sum)

# Simple function for automatically calculating distance
dist_cancer_two_rows <- cancer  |>
  slice(1,2)  |>
  select(Symmetry, Radius, Concavity)  |>
  dist()

# Question 2
set.seed(20)
small_sample <- sample_n(cancer, 5) |>
  select(Symmetry, Radius, Class)
small_sample

small_sample_plot <- ggplot(small_sample, aes(x = Symmetry, y = Radius, colour = Class)) + 
  geom_point(alpha = 0.8) + 
  labs(x = "Symmetry", y ="Radius", colour = "Diagnosis")
small_sample_plot

newData <- small_sample |>
  add_row(Symmetry = 0.5, Radius = 0, Class = "unknown")
newData

dist_matrix <- newData |>
  select(Symmetry, Radius) |>
  dist() |>
  as.matrix()
dist_matrix

# Looks like observation 5 is closest to the new point (observation 6)
# Observation 5 is malignant so if K = 1, we would classify the new point as malignant.

# Closest 3 observations: 1,4,5
# If K = 3, the new point is now benign. 


# Question 3
knn_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = 7) |>
  set_engine("kknn") |>
  set_mode("classification")

knn_fit <- knn_spec |>
  fit(Class ~ Symmetry + Radius, data = cancer)

new_obs <- tibble(Symmetry = 1, Radius = 0)
class_prediction <- predict(knn_fit, new_obs)
class_prediction

# Now, with concavity
knn_fit <- knn_spec |>
  fit(Class ~ Symmetry + Radius + Concavity, data = cancer)

new_obs <- tibble(Symmetry = 1, Radius = 0, Concavity = 1)
class_prediction <- predict(knn_fit, new_obs)
class_prediction

# Now, with all predictors
knn_recipe <- recipe(Class ~ ., data = cancer) |>
  step_rm(ID)

preprocessed_data <- knn_recipe |> 
  prep() |> 
  bake(cancer)
preprocessed_data

knn_workflow <- workflow() |>
  add_recipe(knn_recipe) |>
  add_model(knn_spec)

new_obs_all <- tibble(ID = NA, Radius = 0, 
                      Texture = 0, 
                      Perimeter = 0, 
                      Area = 0, 
                      Smoothness = 0.5, 
                      Compactness = 0,
                      Concavity = 1,
                      Concave_points = 0,
                      Symmetry = 1, 
                      Fractal_dimension = 0)

knn_fit_all <- knn_workflow |>
  fit(data = cancer)
class_prediction_all <- predict(knn_fit_all, new_obs_all)
class_prediction_all


####### Exercises - Classification 2 ##########

# Purposing of scaling: To ensure data will be on comparable scale when computing distances.

# Question 1
fruit_data <- read.csv("data-science-a-first-intro-worksheets-main/worksheet_classification2/data/fruit_data.csv")
fruit_data
# Predicting area is not a classification problem since it is not a categorical variable. 

fruit_data <- fruit_data |>
  mutate(fruit_name = as_factor(fruit_name))

slice(fruit_data, 1:6)

point1 <- c(192, 8.4)
point2 <- c(180, 8)
point44 <- c(194, 7.2)

fruit_data |>  
  ggplot(aes(x=mass, 
             y= width, 
             colour = fruit_name)) +
  labs(x = "Mass (grams)",
       y = "Width (cm)",
       colour = 'Name of the Fruit') +
  geom_point() +
  annotate("path", 
           x=point1[1] + 5*cos(seq(0,2*pi,length.out=100)),
           y=point1[2] + 0.1*sin(seq(0,2*pi,length.out=100))) +
  annotate("text", x = 183, y =  8.5, label = "1") +
  theme(text = element_text(size = 20))

# Closest fruit to point1 is apple. 

fruit_dist_2 <- fruit_data |>
     slice(1, 2) |> # We use slice to get the first two rows of the fruit dataset
     select(mass, width) |>
     dist()
fruit_dist_2

filter(fruit_data, row_number() == 44)

fruit_data |>
  ggplot(aes(x = mass, 
             y = width, 
             colour = fruit_name)) +
  labs(x = "Mass (grams)",
       y = "Width (cm)",
       colour = 'Name of the Fruit') +
  geom_point() +
  annotate("path", 
           x=point1[1] + 5*cos(seq(0,2*pi,length.out=100)),
           y=point1[2] + 0.1*sin(seq(0,2*pi,length.out=100))) +
  annotate("text", x = 183, y =  8.5, label = "1") +
  annotate("path",
           x=point2[1] + 5*cos(seq(0,2*pi,length.out=100)),
           y=point2[2] + 0.1*sin(seq(0,2*pi,length.out=100))) +
  annotate("text", x = 169, y =  8.1, label = "2") +
  annotate("path",
           x=point44[1] + 5*cos(seq(0,2*pi,length.out=100)),
           y=point44[2]+0.1*sin(seq(0,2*pi,length.out=100))) +
  annotate("text", x = 204, y =  7.1, label = "44") +
  theme(text = element_text(size = 20))

fruit_dist_44 <- fruit_data |>
  slice(1, 44) |> # We use slice to get the first two rows of the fruit dataset
  select(mass, width) |>
  dist()
fruit_dist_44

# The distance between 1 and 2 is closer than 1 and 44 on the scatterplot but the
# calculated distance says opposite. 
# This is why we need to standardize.

fruit_data_scaled <- fruit_data |>
mutate(scaled_mass = scale(mass, center = TRUE),
       scaled_width = scale(width, center = TRUE),
       scaled_height = scale(height, center = TRUE),
       scaled_color_score = scale(color_score, center = TRUE))
fruit_data_scaled

distance_2 <- fruit_data_scaled |>
  slice(1, 2) |> # We use slice to get the first two rows of the fruit dataset
  select(scaled_mass, scaled_width) |>
  dist()
distance_2

distance_44 <- fruit_data_scaled |>
  slice(1, 44) |> # We use slice to get the first two rows of the fruit dataset
  select(scaled_mass, scaled_width) |>
  dist()
distance_44

# Question 2
# Splitting into train and test set
set.seed(3456) 
fruit_split <- initial_split(fruit_data, prop = 0.75, strata = fruit_name)
fruit_train <- training(fruit_split)
fruit_test <- testing(fruit_split)
fruit_train
fruit_test

# Standardization happens after the split since it needs only be on training data
fruit_recipe <- recipe(fruit_name ~ mass + color_score, data = fruit_train) |>
  step_scale(all_predictors()) |>
  step_center(all_predictors())

knn_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = 3) |>
  set_engine("kknn") |>
  set_mode("classification")

fruit_fit <- workflow() |>
  add_recipe(fruit_recipe) |>
  add_model(knn_spec) |>
  fit(data = fruit_train)
fruit_fit

fruit_test_predictions <- predict(fruit_fit, fruit_test) |>
  bind_cols(fruit_test)
fruit_test_predictions

fruit_prediction_accuracy <- fruit_test_predictions |>
  metrics(truth = fruit_name, estimate = .pred_class) |>
  filter(.metric == "accuracy")
fruit_prediction_accuracy

fruit_mat <- fruit_test_predictions |>
  conf_mat(truth = fruit_name, estimate = .pred_class)
fruit_mat

# From the confusion matrix, we see that 13 observations were labelled correctly.
# (From the diagonal entries)

# Question 3
fruit_vfold <- vfold_cv(fruit_train, v = 5, strata = fruit_name)
fruit_vfold

set.seed(2020)
fruit_resample_fit <- workflow() |>
      add_recipe(fruit_recipe) |>
      add_model(knn_spec) |>
      fit_resamples(resamples = fruit_vfold)


fruit_resample_fit |>
  collect_metrics()

# Question 4
knn_tune <- nearest_neighbor(weight_func = "rectangular", neighbors = tune()) |>
  set_engine("kknn") |>
  set_mode("classification")

knn_results <- workflow() |>
      add_recipe(fruit_recipe) |>
      add_model(knn_tune) |>
      tune_grid(resamples = fruit_vfold, grid = 10) |>
      collect_metrics()
knn_results

accuracies <- knn_results |> 
  filter(.metric == "accuracy")
accuracies

accuracy_vs_k <- ggplot(accuracies, aes(x = neighbors, y = mean)) +
  geom_point() +
  geom_line() +
  labs(x = "Neighbors", y = "Accuracy Estimate") +
  theme(text = element_text(size = 12))
accuracy_vs_k

# We can use the graph or pull directly the maximum from the table
best_k <- accuracies |>
  arrange(desc(mean)) |>
  head(1) |>
  pull(neighbors)
best_k

# K = 2 looks to be the most optimal.