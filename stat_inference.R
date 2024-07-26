# Population parameter: Numerical characteristic about an entire population
# Ex. Proportion of university students that have an iPhone in NA.

# But we can't ask every university student in NA. So we take a sample.
# From there, we get a sample estimate of the population parameter.

# Statistical inference: process of using a sample to make a conclusion about 
# the broader population from which it is taken.

airbnb <- read_csv("listings.csv")
airbnb

# Proportion of Airbnb places that ar entire homes or apartments
airbnb |>
  summarize(
    n =  sum(room_type == "Entire home/apt"),
    proportion = sum(room_type == "Entire home/apt") / nrow(airbnb)
  )

# Let's suppose that 0.747 is the true proportion. 
# Now, we'll take a subset of data. 
library(infer)

sample_1 <- rep_sample_n(tbl = airbnb, size = 40)
airbnb_sample_1 <- summarize(sample_1,
                             n = sum(room_type == "Entire home/apt"),
                             prop = sum(room_type == "Entire home/apt") / 40)
airbnb_sample_1

# Point estimate is 0.85. 
# If we took another sample, we would get a different estimate.
sample_2 <- rep_sample_n(airbnb, size = 40)
airbnb_sample_2 <- summarize(sample_2,
                             n = sum(room_type == "Entire home/apt"),
                             prop = sum(room_type == "Entire home/apt") / 40)
airbnb_sample_2
# Another point estimate is 0.65. 

# We take many sample, we get a sampling distribution. 
samples <- rep_sample_n(airbnb, size = 40, reps = 20000)
samples

sample_estimates <- samples |>
  group_by(replicate) |>
  summarize(sample_proportion = sum(room_type == "Entire home/apt") / 40)
sample_estimates

sampling_distribution <- ggplot(sample_estimates, aes(x = sample_proportion)) +
  geom_histogram(color = "lightgrey", bins = 12) +
  labs(x = "Sample proportions", y = "Count") +
  theme(text = element_text(size = 12))
sampling_distribution

# Mean of the sample proportions
sample_estimates |>
  summarize(mean_proportion = mean(sample_proportion))

# Now, try quantitative variable such as mean price per night
population_distribution <- ggplot(airbnb, aes(x = price)) +
  geom_histogram(color = "lightgrey") +
  labs(x = "Price per night (dollars)", y = "Count") +
  theme(text = element_text(size = 12))
population_distribution

population_parameters <- airbnb |>
  summarize(mean_price = mean(price))
population_parameters
# Mean price is $155. 

one_sample <- airbnb |>
  rep_sample_n(40)

sample_distribution <- ggplot(one_sample, aes(price)) +
  geom_histogram(color = "lightgrey") +
  labs(x = "Price per night (dollars)", y = "Count") +
  theme(text = element_text(size = 12))
sample_distribution

estimates <- one_sample |>
  summarize(mean_price = mean(price))
estimates
# From sample, mean price is $141.

samples <- rep_sample_n(airbnb, size = 40, reps = 20000)
samples

sample_estimates <- samples |>
  group_by(replicate) |>
  summarize(mean_price = mean(price))
sample_estimates

sampling_distribution_40 <- ggplot(sample_estimates, aes(x = mean_price)) +
  geom_histogram(color = "lightgrey") +
  labs(x = "Sample mean price per night (dollars)", y = "Count") +
  theme(text = element_text(size = 12))
sampling_distribution_40


# Confidence interval: Range of plausible values for the population parameter.

# Bootstrap: If our sample is large enough, we can pretend it's the population.
# Then, take more sample (with replacement) of the same size from it
# to get a distribution. 

one_sample

one_sample_dist <- ggplot(one_sample, aes(price)) +
  geom_histogram(color = "lightgrey") +
  labs(x = "Price per night (dollars)", y = "Count") +
  theme(text = element_text(size = 12))
one_sample_dist

estimates <- one_sample |>
  summarize(mean_price = mean(price))
estimates
# From sample, mean price is $141.

# Step 1 of Bootstrap
boot1 <- one_sample |>
  rep_sample_n(size = 40, replace = TRUE, reps = 1)
boot1_dist <- ggplot(boot1, aes(price)) +
  geom_histogram(color = "lightgrey") +
  labs(x = "Price per night (dollars)", y =  "Count") +
  theme(text = element_text(size = 12))
boot1_dist

mean(boot1$price)
# First bootstrap sample, mean price is $147.

# Take 20,000 bootstrap samples
boot20000 <- one_sample |>
  rep_sample_n(size = 40, replace = TRUE, reps = 20000)
boot20000

# Look at the first six bootstrap samples
boot20000 |>
  group_by(replicate...2) |>
  summarize(mean_price = mean(price))

boot20000_means <- boot20000 |>
  group_by(replicate...1) |>
  summarize(mean_price = mean(price))
boot20000_means

boot_est_dist <- ggplot(boot20000_means, aes(x = mean_price)) +
  geom_histogram(color = "lightgrey") +
  labs(x = "Sample mean price per night (dollars)", y = "Count") +
  theme(text = element_text(size = 12))
boot_est_dist

# The shape and spread of the true sampling distribution and the bootstrap 
# distribution are similar.
# The bootstrap distribution lets us get a sense of the point estimate’s variability.

# What does 95% confidence mean?
# If we took 100 random samples and calculated 100 95% confidence intervals, 
# then about 95% of the ranges would capture the population parameter’s value.

bounds <- boot20000_means |>
  select(mean_price) |>
  pull() |>
  quantile(c(0.025, 0.975))
bounds

######### Exercises - 1 ############

library(cowplot)

# Definitions:
# Point estimate: Single number calculated from a random sample that estimates 
# an unknown population parameter of interest
# Population: Entire set of entities/objects of interest
# Random sampling: Selecting a subset of observations from a population where 
# each observation is equally likely to be selected
# Representative sampling: Selecting a subset of observations from a population 
# where the sample’s characteristics are a good representation of the 
# population’s characteristics
# Population parameter: Numerical summary value about the population
# Observation: Quantity or a quality (or set of these) we collect from a given entity/object
# Sample: Collection of observations from a population
# Sampling distribution: Distribution of point estimates, where each point 
# estimate was calculated from a different random sample from the same population

set.seed(4321)
can_seniors <- tibble(age = (rexp(2000000, rate = 0.1)^2) + 65) |> 
  filter(age <= 117, age >= 65)
can_seniors

pop_dist <- ggplot(can_seniors, aes(x = age)) + 
  geom_histogram(color = "lightgrey")
pop_dist

pop_parameters <- can_seniors |>
  summarize(pop_mean = mean(age),
            pop_median = median(age),
            pop_sd = sd(age))
pop_parameters

sample1 <- rep_sample_n(can_seniors, 40)
sample1

sample_1_dist <- ggplot(sample1, aes(x = age)) +
  geom_histogram(binwidth = 1, color = "lightgrey") +
  labs(x = "Age", title = "Sample 1 Distribution")

sample_1_estimates <- sample1 |>
  summarize(pop_mean = mean(age),
            pop_median = median(age),
            pop_sd = sd(age))
sample_1_estimates

plot_grid(pop_dist, sample_1_dist, ncol = 1)

pop_parameters
sample_1_estimates |> select(-replicate)

# Similar in point estimate but sampling distribution is not similar to pop distribution

# 1500 samples
samples <- rep_sample_n(can_seniors, 40, reps = 1500)
samples

samples_estimates <- samples |>
  group_by(replicate) |>
  summarize(mean_age = mean(age))
samples_estimates

sampling_distribution <- ggplot(samples_estimates, aes(x = mean_age)) +
  geom_histogram(binwidth = 1, color = "lightgrey") +
  labs(x = "Mean Age", title = "Sampling Distribution")
sampling_distribution

samples_mean <- samples_estimates |>
  summarize(mean_age_whole = mean(mean_age))
samples_mean

###### Exercises - 2 ########

pop_dist <- ggplot(can_seniors, aes(age)) + 
  geom_histogram(binwidth = 1) +
  xlab("Age (years)") +
  ggtitle("Population distribution") +
  theme(text = element_text(size = 20), plot.margin = margin(10, 100)) # last x value was getting cut off
pop_dist

# Sample for bootstraping
one_sample <- can_seniors |> 
  rep_sample_n(40) |> 
  ungroup() |> # ungroup the data frame 
  select(age) # drop the replicate column 
one_sample

one_sample_dist <- ggplot(one_sample, aes(age)) + 
  geom_histogram(binwidth = 1, color = 'grey') +
  xlab("Age (years)") +
  ggtitle("Distribution of one sample") +
  theme(text = element_text(size = 20))
one_sample_dist

one_sample_estimates <- one_sample |>
  summarize(mean = mean(age))
one_sample_estimates

boot1 <- one_sample |>
  rep_sample_n(size = 40, replace = TRUE, reps = 1500)
boot1

boot1_dist <- ggplot(boot1, aes(age)) + 
  geom_histogram(binwidth = 1, color = 'grey') +
  xlab("Age (years)") +
  ggtitle("Distribution of bootstraps") +
  theme(text = element_text(size = 20))
boot1_dist

# Comparison
plot_grid(one_sample_dist, boot1_dist, ncol = 2)

one_sample_estimates
boot1_mean_rep <- boot1  |> 
  summarise(mean = mean(age))
boot1_mean <- boot1_mean_rep |>
  summarize(mean_age_whole = mean(mean))
boot1_mean

sampling_dist <-  ggplot(boot1_mean_rep, aes(x = mean)) +
  geom_histogram(binwidth = 1, color = "grey") +
  xlab("Mean Age (years)") +
  ggtitle("Sampling distribution of the sample means") +
  annotate("text", x = 85, y = 200, label = paste("mean = ", round(mean(boot1_mean_rep$mean), 1)), cex = 5) +
  theme(text = element_text(size = 20))
sampling_dist

boot1_mean_rep |> 
  select(mean) |> 
  pull() |> 
  quantile(c(0.025, 0.975))
# This is the 95% confidence interval for mean age of Canadian seniors