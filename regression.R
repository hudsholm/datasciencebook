# Regression is like classification, only difference being that we want
# to predict numerical instead of categorical variables. 

# Two methods:
# i) K-nearest neighbours
# ii) Linear regression

# Research question: 
# Can we use the size of a house in the Sacramento, CA area to predict its sale price?

library(tidyverse)
library(tidymodels)

install.packages("gridExtra")
library(gridExtra)

sacramento <- read_csv("sacramento.csv")
sacramento

# First step is to visualize
eda <- ggplot(sacramento, aes(x = sqft, y = price)) +
  geom_point(alpha = 0.4) +
  xlab("House size (square feet)") +
  ylab("Price (USD)") +
  scale_y_continuous(labels = dollar_format()) +
  theme(text = element_text(size = 12))
eda

small_sacramento <- slice_sample(sacramento, n = 30)
small_plot <- ggplot(small_sacramento, aes(x = sqft, y = price)) +
  geom_point() +
  labs(x = "House size (square feet)",
       y = "Price (USD)") +
  scale_y_continuous(labels = dollar_format()) +
  geom_vline(xintercept = 2000, linetype = "dashed") +
  theme(text = element_text(size = 12))
small_plot

# 5 nearest neighbours to a house that is 2,000 square feet
nearest_neighbors <- small_sacramento |>
  mutate(diff = abs(2000 - sqft)) |>
  slice_min(diff, n = 5)
nearest_neighbors

# The mean price of these nearest neighbours
prediction <- nearest_neighbors |>
  summarise(predicted = mean(price))
prediction

# The predicted price is 365,000. Therefore, if a house of this size is 
# listed at 400,000, you may be inclined to offer less. 

# One advantage about KNN is it can work with non-linear relationships too.
# There are less restrictive assumptions required.

sacramento_split <- initial_split(sacramento, prop = 0.75, strata = price)
sacramento_train <- training(sacramento_split)
sacramento_test <- testing(sacramento_split)

# After splitting, use cross-validation to choose K
# New measure of accuracy: Root mean square prediction error (RMSPE)

# Standardization is not necessary because only one variable but good practice
sacr_recipe <- recipe(price ~ sqft, data = sacramento_train) |>
  step_scale(all_predictors()) |>
  step_center(all_predictors())
sacr_spec <- nearest_neighbor(weight_func = "rectangular",
                              neighbors = tune()) |>
  set_engine("kknn") |>
  set_mode("regression")
sacr_vfold <- vfold_cv(sacramento_train, v = 5, strata = price)

sacr_wkflw <- workflow() |>
  add_recipe(sacr_recipe) |>
  add_model(sacr_spec)
sacr_wkflw

gridvals <- tibble(neighbors = seq(from = 1, to = 200, by = 3))

sacr_results <- sacr_wkflw |>
  tune_grid(resamples = sacr_vfold, grid = gridvals) |>
  collect_metrics() |>
  filter(.metric == "rmse")
sacr_results

ggplot(sacr_results, aes(x = neighbors, y = mean)) + 
  geom_point() +
  labs(x = "Neighbours",
       y = "RMSPE") +
  geom_line()

sacr_min <- sacr_results |>
  filter(mean == min(mean))
sacr_min
# Appears that the K = 22 results in the lowest amount of error

# If K is too little, overfitting (too specific).
# If K is too large, underfitting (too general).

# Now, evaluating on test set
kmin <- sacr_min |> pull(neighbors)

sacr_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = kmin) |>
  set_engine("kknn") |>
  set_mode("regression")

sacr_fit <- workflow() |>
  add_recipe(sacr_recipe) |>
  add_model(sacr_spec) |>
  fit(data = sacramento_train)

sacr_summary <- sacr_fit |>
  predict(sacramento_test) |>
  bind_cols(sacramento_test) |>
  metrics(truth = price, estimate = .pred) |>
  filter(.metric == 'rmse')
sacr_summary
# Final model’s test error is RMSPE = $90,529
# Good news is this is similar to the cross-validation RMSPE estimate
# Bad news is the $90,529 can mean a lot in a home buyer's budget.

# Predicting a range of value and superimposing the fit line
sqft_prediction_grid <- tibble(
  sqft = seq(
    from = sacramento |> select(sqft) |> min(),
    to = sacramento |> select(sqft) |> max(),
    by = 10
  )
)

sacr_preds <- sacr_fit |>
  predict(sqft_prediction_grid) |>
  bind_cols(sqft_prediction_grid)

plot_final <- ggplot(sacramento, aes(x = sqft, y = price)) +
  geom_point(alpha = 0.4) +
  geom_line(data = sacr_preds,
            mapping = aes(x = sqft, y = .pred),
            color = "steelblue",
            linewidth = 1) +
  xlab("House size (square feet)") +
  ylab("Price (USD)") +
  scale_y_continuous(labels = dollar_format()) +
  ggtitle(paste0("K = ", kmin)) +
  theme(text = element_text(size = 12))
plot_final

# Adding number of bedrooms as a predictor variable
plot_beds <- sacramento |>
  ggplot(aes(x = beds, y = price)) +
  geom_point(alpha = 0.4) +
  labs(x = 'Number of Bedrooms', y = 'Price (USD)') +
  theme(text = element_text(size = 12))
plot_beds
# We would expect increase in bedrooms means an increase in price.

sacr_recipe <- recipe(price ~ sqft + beds, data = sacramento_train) |>
  step_scale(all_predictors()) |>
  step_center(all_predictors())

sacr_spec <- nearest_neighbor(weight_func = "rectangular",
                              neighbors = tune()) |>
  set_engine("kknn") |>
  set_mode("regression")

gridvals <- tibble(neighbors = seq(1, 200))

sacr_multi <- workflow() |>
  add_recipe(sacr_recipe) |>
  add_model(sacr_spec) |>
  tune_grid(sacr_vfold, grid = gridvals) |>
  collect_metrics() |>
  filter(.metric == "rmse") |>
  filter(mean == min(mean))

sacr_k <- sacr_multi |>
  pull(neighbors)
sacr_multi
# Looks like K = 25 results in the smallest RMSPE
# The estimated cross-validation RMSPE is $84,592.
# When we just have one predictor, $87,190. So, we improved slightly. 

sacr_spec <- nearest_neighbor(weight_func = "rectangular",
                              neighbors = sacr_k) |>
  set_engine("kknn") |>
  set_mode("regression")

knn_mult_fit <- workflow() |>
  add_recipe(sacr_recipe) |>
  add_model(sacr_spec) |>
  fit(data = sacramento_train)

knn_mult_preds <- knn_mult_fit |>
  predict(sacramento_test) |>
  bind_cols(sacramento_test)

knn_mult_mets <- metrics(knn_mult_preds, truth = price, estimate = .pred) |>
  filter(.metric == 'rmse')

knn_mult_mets


# Simple linear regression
# How to choose line of best fit? By choosing the line that minimizes the 
# average squared vertical distance between itself and each of the observations
# in the training data (equivalent to minimizing the RMSE)

sacramento_split <- initial_split(sacramento, prop = 0.75, strata = price)
sacramento_train <- training(sacramento_split)
sacramento_test <- testing(sacramento_split)

lm_spec <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

# No need to standardize
lm_recipe <- recipe(price ~ sqft, data = sacramento_train)

lm_fit <- workflow() |>
  add_recipe(lm_recipe) |>
  add_model(lm_spec) |>
  fit(data = sacramento_train)
lm_fit

# Model predicts intercept = $16,572.4 (cost for 0 sqft)
# Model predicts slope = $135 (cost for every additional sqft)

# Apply on test data
lm_test_results <- lm_fit |>
  predict(sacramento_test) |>
  bind_cols(sacramento_test) |>
  metrics(truth = price, estimate = .pred)
lm_test_results

# Visualize result with plots
sqft_prediction_grid <- tibble(
  sqft = c(
    sacramento |> select(sqft) |> min(),
    sacramento |> select(sqft) |> max()
  )
)

sacr_preds <- lm_fit |>
  predict(sqft_prediction_grid) |>
  bind_cols(sqft_prediction_grid)

lm_plot_final <- ggplot(sacramento, aes(x = sqft, y = price)) +
  geom_point(alpha = 0.4) +
  geom_line(data = sacr_preds,
            mapping = aes(x = sqft, y = .pred),
            color = "steelblue",
            linewidth = 1) +
  xlab("House size (square feet)") +
  ylab("Price (USD)") +
  scale_y_continuous(labels = dollar_format()) +
  theme(text = element_text(size = 12))
lm_plot_final

# To get detailed information about the coefficients
coeffs <- lm_fit |>
  extract_fit_parsnip() |>
  tidy()
coeffs

# Comparison:
# KNN -> extrapolation at the ends usually go flat which may or may not make sense
# Linear regression -> extrapolation at the ends is straight line so the intercept
# could go negative which doesn't make sense in terms of house price.

# Add number of beds as a predictor
mlm_recipe <- recipe(price ~ sqft + beds, data = sacramento_train)
# Everything else is the same as before
mlm_fit <- workflow() |>
  add_recipe(mlm_recipe) |>
  add_model(lm_spec) |>
  fit(data = sacramento_train)
mlm_fit

lm_mult_test_results <- mlm_fit |>
  predict(sacramento_test) |>
  bind_cols(sacramento_test) |>
  metrics(truth = price, estimate = .pred)
lm_mult_test_results

mcoeffs <- mlm_fit |>
  extract_fit_parsnip() |>
  tidy()
mcoeffs

# Two main issues with linear regression: Outliers and colinear predictors
# Avoid outliers when reasonable (such as parent selling below market to children)
# Avoid linearly related predictors since coefficients will vary greately after CV


####### Exercises - Regression 1 ##########

# KNN-regression: Take the mean of the k-nearest neighbours as the predicted value
# RMSPE takes the root of the average of the squared differences in actual value
# and predicted value

# Research question:
# What predicts which athletes will perform better than others?
# Is distance ran per week during training an important factor?

marathon <- read.csv("data-science-a-first-intro-worksheets-main/worksheet_regression1/data/marathon.csv")
marathon

set.seed(2000)
marathon_50 <- marathon |>
   sample_n(50)
marathon_50

ggplot(marathon_50, aes(x = max, y = time_hrs)) + 
  geom_point() +
  labs(x = "Max distance ran per week in training (miles)",
       y = "Race time (hours)")

# 4 closest neighbours to an observation that trained 100 miles.
marathon_50 |>
  ggplot(aes(x = max, y = time_hrs)) + 
  geom_point(color = 'dodgerblue', alpha = 0.4) +
  geom_vline(xintercept = 100, linetype = "dotted") +
  xlab("Maximum Distance Ran per \n Week During Training (mi)") +
  ylab("Race Time (hours)") + 
  geom_segment(aes(x = 100, y = 2.56, xend = 107, yend = 2.56), col = "orange") +
  geom_segment(aes(x = 100, y = 2.65, xend = 90, yend = 2.65), col = "orange") +
  geom_segment(aes(x = 100, y = 2.99, xend = 86, yend = 2.99), col = "orange") +
  geom_segment(aes(x = 100, y = 3.05, xend = 82, yend = 3.05), col = "orange") +
  theme(text = element_text(size = 20))

nearest_runners <- marathon_50 |>
 mutate(diff = abs(100 - max)) |>
 slice_min(diff, n = 4) |>
 summarise(predicted = mean(time_hrs)) |>
 pull()
nearest_runners
# So, expected race time would be 2.81 hours based on 4 nearest neighbours.

nearest_two <- marathon_50 |>
  mutate(diff = abs(100 - max)) |>
  slice_min(diff, n = 2) |>
  summarise(predicted = mean(time_hrs)) |>
  pull()
nearest_two
# Expected race time would be 2.60 hours based on 2 nearest neighbours.

# How we choose k? The k with the lowest cross-validation error. 

marathon_split <- initial_split(marathon, prop = 0.75, strata = time_hrs)
marathon_training <- training(marathon_split)
marathon_testing <- testing(marathon_split)

marathon_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = tune()) |>
      set_engine("kknn") |>
      set_mode("regression")

marathon_recipe <- recipe(time_hrs ~ max, data = marathon_training) |>
      step_scale(all_predictors()) |>
      step_center(all_predictors())

marathon_vfold <- vfold_cv(marathon_training, v = 5, strata = time_hrs)

marathon_wkflw <- workflow() |>
  add_recipe(marathon_recipe) |>
  add_model(marathon_spec)
marathon_wkflw




gridvals <- tibble(neighbors = seq(from = 1, to = 81, by = 10))

marathon_results <- marathon_wkflw |>
  tune_grid(resamples = marathon_vfold, grid = gridvals) |>
  collect_metrics() |>
  filter(.metric == "rmse")
marathon_results

ggplot(marathon_results, aes(x = neighbors, y = mean)) + 
  geom_point() +
  labs(x = "Neighbours",
       y = "RMSPE") +
  geom_line()

marathon_min <- marathon_results |>
  filter(mean == min(mean))
marathon_min

kmin <- marathon_min |> pull(neighbors)

marathon_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = kmin) |>
  set_engine("kknn") |>
  set_mode("regression")

marathon_fit <- workflow() |>
  add_recipe(marathon_recipe) |>
  add_model(marathon_spec) |>
  fit(data = marathon_training)

marathon_summary <- marathon_fit |>
  predict(marathon_testing) |>
  bind_cols(marathon_testing) |>
  metrics(truth = time_hrs, estimate = .pred) |>
  filter(.metric == 'rmse')
marathon_summary

# What does the RMSPE mean? It is 0.582 which means the model can predict a runner's
# race time up to about +/- 0.582 which is around 35 minutes. 

# How does the RMSPE compare with the cross-validation estimate? 
# Off by 0.002 so very similar. 

racetime_prediction_grid <- tibble(
  max = seq(
    from = marathon |> select(max) |> min(),
    to = marathon |> select(max) |> max(),
    by = 10
  )
)

marathon_preds <- marathon_fit |>
  predict(racetime_prediction_grid) |>
  bind_cols(racetime_prediction_grid)

marathon_plot <- ggplot(marathon, aes(x = max, y = time_hrs)) +
  geom_point(alpha = 0.4) +
  geom_line(data = marathon_preds,
            mapping = aes(x = max, y = .pred),
            color = "steelblue",
            linewidth = 1) +
  xlab("Maximum distance run per week during training (miles)") +
  ylab("Race time (hours)") +
  ggtitle(paste0("K = ", kmin)) +
  theme(text = element_text(size = 12))
marathon_plot

####### Exercises - Regression 2 ##########

install.packages("cowplot")
library(cowplot)

# In KNN regression with two predictors, the predictions take the form of 
# a wiggly/flexible plane.

# In simple linear regression with one predictors, the predictions take the form of 
# a straight line.

# In simple linear regression with two predictors, the predictions take the form of 
# a flat plane.

# Consider the following hypothetical dataset
simple_data  <- tibble(X = c(1, 2, 3, 6, 7, 7),
                       Y = c(1, 1, 3, 5, 7, 6))
options(repr.plot.width = 5, repr.plot.height = 5)
base <- ggplot(simple_data, aes(x = X, y = Y)) +
  geom_point(size = 2) +
  scale_x_continuous(limits = c(0, 7.5), breaks = seq(0, 8), minor_breaks = seq(0, 8, 0.25)) +
  scale_y_continuous(limits = c(0, 7.5), breaks = seq(0, 8), minor_breaks = seq(0, 8, 0.25)) +
  theme(text = element_text(size = 20))
base 

options(repr.plot.height = 3.5, repr.plot.width = 10)
line_a <- base +
  ggtitle("Line A") +
  geom_abline(intercept = -0.897, slope = 0.9834, color = "blue") +
  theme(text = element_text(size = 20))
line_b <- base +
  ggtitle("Line B") +
  geom_abline(intercept = 0.1022, slope = 0.9804, color = "purple") +
  theme(text = element_text(size = 20))
line_c <- base +
  ggtitle("Line C") +
  geom_abline(intercept = -0.2347, slope = 0.9164, color = "green") +
  theme(text = element_text(size = 20))
plot_grid(line_a, line_b, line_c, ncol = 3)

# Average squared vertical distance between points and line A
line_a
line_a_points <- -0.897 + 0.9834*c(1, 2, 3, 6, 7, 7)
actual_points <- c(1, 1, 3, 5, 7, 6)
sum((actual_points-line_a_points)^2)/6

# Average squared vertical distance between points and line B
line_b
line_b_points <- 0.1022 + 0.9804*c(1, 2, 3, 6, 7, 7)
actual_points <- c(1, 1, 3, 5, 7, 6)
sum((actual_points-line_b_points)^2)/6

# Average squared vertical distance between points and line C
line_c
line_c_points <- -0.2347 + 0.9164*c(1, 2, 3, 6, 7, 7)
actual_points <- c(1, 1, 3, 5, 7, 6)
sum((actual_points-line_c_points)^2)/6

# Line C is the best since the distance between the line of best fit and 
# the observations is minimized.

marathon_split <- initial_split(marathon, prop = 0.75, strata = price)
marathon_training <- training(marathon_split)
marathon_testing <- testing(marathon_split)

lm_spec <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

lm_recipe <- recipe(time_hrs ~ max, data = marathon_training)

lm_fit <- workflow() |>
  add_recipe(lm_recipe) |>
  add_model(lm_spec) |>
  fit(data = marathon_training)
lm_fit

# Model predicts intercept = 4.950 (race time for 0 training)
# Model predicts slope = -0.023 (decrease in race time for every additional training mile)

marathon_preds <- lm_fit |>
  predict(marathon_training) |>
  bind_cols(marathon_training)

lm_predictions <- marathon_training |>
    ggplot(aes(x = max, y = time_hrs)) +
        geom_point(alpha = 0.4) +
        geom_line(marathon_preds,
            mapping = aes(x = max, y = .pred),
            color = "blue") +
        xlab("Max distance ran per week in training (miles)") +
        ylab("Race time (hours)") +
        theme(text = element_text(size = 20))
lm_predictions

# Apply on test data
lm_test_results <- lm_fit |>
  predict(marathon_testing) |>
  bind_cols(marathon_testing) |>
  metrics(truth = time_hrs, estimate = .pred)
lm_test_results

lm_rmspe <- lm_test_results |>
  filter(.metric == "rmse") |>
  select(.estimate) |>
  pull()
lm_rmspe

lm_predictions_test <- marathon_testing |>
  ggplot(aes(x = max, y = time_hrs)) +
  geom_point(alpha = 0.4) +
  geom_line(marathon_preds,
            mapping = aes(x = max, y = .pred),
            color = "blue") +
  xlab("Max distance ran per week in training (miles)") +
  ylab("Race time (hours)") +
  theme(text = element_text(size = 20))
lm_predictions_test

# Similar result in terms of RMSPE with KNN.
