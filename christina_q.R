# 1. FIX THE NAMES FIRST
# We will use 'janitor' to make the names safe and unique
library(janitor)
library(dplyr)
library(survey)
library(broom)
library(ggplot2)
library(ranger)
library(ggeffects)
library(cvms)
library(scales)
library(stringr)

meps_sub <- meps_subset_2023

meps_ready <- meps_sub %>%
  clean_names() %>% # This turns AGE23X into age23x
  mutate(age23x = as.numeric(as.character(age23x))) %>%
  # Only keep columns where data is valid (>= 0)
  filter(unins23 >= 0, dlayca42 >= 0, empst31 >= 0, faminc23 >= 0, age23x >= 0) %>%
  # 2. Handle missing ages immediately inside the pipe
  filter(!is.na(age23x)) %>%
  mutate(
    # Create binary flags
    emp_unstable   = ifelse(empst31 != empst53, 1, 0),
    # Set "Insured all year" as baseline
    uninsured_flag = relevel(as.factor(unins23), ref = "2"),
    # Log transform (now safe because we filtered < 0)
    log_fam_income     = log(faminc23 + 1),
    dlayca42 = factor(dlayca42, levels = c(2, 1), labels = c("No","Yes")),
    married_status = factor(marry23x,
                            levels = c(1, 2, 3, 4, 5, 6),
                            labels = c("married", "widowed", "divorced", "separated", "never_married", "under16"))
  )


# 2. CHECK THE NEW NAMES
# Look for 'age23x_4' or similar in the output
names(meps_ready)

# 3. CREATE THE DESIGN WITH THE NEW DATA
# Ensure 'data = meps_ready' matches the data you just cleaned
meps_design <- svydesign(id = ~1, weights = ~perwt23f, data = meps_ready)


# 4. RUN THE MODEL (Update formula to match the new 'clean_names')
logit_model <- svyglm(uninsured_flag ~ log_fam_income + emp_unstable + 
                        married_status + age23x + dlayca42, 
                      design = meps_design, 
                      family = binomial())

summary(logit_model)


# Random Forest version (doesn't require svydesign)
rf_model <- ranger(
  formula = uninsured_flag ~ log_fam_income + emp_unstable + married_status + age23x + dlayca42,
  data = meps_ready,
  case.weights = meps_ready$perwt23f,
  importance = "permutation"
)

# Bar Chart: Unisured By Income Bracket

# 1. Prepare data using 'faminc23' instead of 'povcat23'
income_summary <- meps_ready %>%
  mutate(income_bracket = cut(faminc23, 
                              breaks = quantile(faminc23, probs = seq(0, 1, 0.2), na.rm = TRUE),
                              labels = c("Lowest 20%", "Lower-Mid", "Middle", "Upper-Mid", "Highest 20%"),
                              include.lowest = TRUE)) %>%
  group_by(income_bracket) %>%
  summarise(
    # Weighted proportion of people where unins23 == 1 (Uninsured)
    rate_uninsured = weighted.mean(unins23 == 1, w = perwt23f, na.rm = TRUE)
  )

# 2. Plot
ggplot(income_summary, aes(x = income_bracket, y = rate_uninsured, fill = income_bracket)) +
  geom_bar(stat = "identity", color = "black", alpha = 0.8) +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_brewer(palette = "Blues") +
  labs(title = "Uninsurance Rates by Family Income Quintile",
       subtitle = "Weighted estimates using PERWT23F",
       x = "Income Group (Low to High)", y = "Percent Uninsured") +
  theme_minimal() +
  theme(legend.position = "none")

# For Logistic Visual
# 1. Extract Odds Ratios and Confidence Intervals and Clean the Results
logit_results <- tidy(logit_model, conf.int = TRUE, exponentiate = TRUE) %>%
  filter(term != "(Intercept)") %>% # Remove intercept for better scaling
  # 4. Extract and Clean the Results
  mutate(
    # This removes the "married_status" prefix from the labels
    term = str_replace(term, "married_status", ""),
    # Optional: Capitalize and clean other variable names if needed
    term = str_replace(term, "log_fam_income", "Family Income (log)"),
    term = str_replace(term, "age23x", "Age"),
    term = str_replace(term, "dlayca42", "delayed care: "),
    # term = str_replace(term, "Delayed Care: 2", "No Delayed Care"),
  )

# 2. Plot
ggplot(logit_results, aes(x = estimate, y = term)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "red") +
  geom_point(size = 3, color = "darkblue") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
  labs(title = "Odds Ratios for Being Uninsured (2023)",
       subtitle = "Estimates > 1.0 indicate increased risk of uninsurance",
       x = "Odds Ratio (Log Scale)", y = "Socioeconomic Factor") +
  scale_x_log10() + # Better for visualizing Odds Ratios
  theme_minimal()

# For Random Forest Visual, Variable Importance Bar chart
# 1. Create a data frame from the ranger importance object
importance_data <- data.frame(
  Feature = names(rf_model$variable.importance),
  Importance = rf_model$variable.importance
)

# 2. Plot
ggplot(importance_data, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() + # Makes long variable names readable
  scale_y_continuous(
    breaks = seq(0, 0.020, by = 0.0025), # Sets ticks every 0.0025
    labels = scales::comma             # Formats with commas if needed
  ) +
  labs(title = "Key Drivers of Uninsurance",
       x = "Variable", y = "Permutation Importance Score") +
  theme_minimal()

# Comparing Both: Predicted Probability
# Using 'ggeffects' to calculate marginal effects
# 1. Calculate the predicted probabilities for income
predict_income <- ggpredict(logit_model, terms = "log_fam_income [all]")

# 2. Plot
plot(predict_income) +
  scale_x_continuous(
    breaks = c(0, 2, 4, 6, 8, 10, 12, 14), # Manual tick placement
    expand = c(0, 0)                      # Removes extra padding at edges
  ) +
  labs(title = "Probability of Being Uninsured by Income Level",
       x = "Log of Family Income", y = "Predicted Probability") +
  theme_minimal()

# The Audit Matrix, Confusion Heatmap
# library(cvms) For clean confusion matrix plots
# 1. Get predictions from the Random Forest
meps_ready$pred <- predict(rf_model, data = meps_ready)$predictions

# 2. Create the matrix
cfm <- as.data.frame(table(meps_ready$uninsured_flag, meps_ready$pred))

ggplot(cfm, aes(x = Var1, y = Var2, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Model Accuracy: Predicted vs. Actual Uninsurance",
       x = "Actual Status (0=Insured, 1=Uninsured)", 
       y = "Model Prediction") +
  theme_minimal()

# Explore the predictor age
# 1. Create logical age bins based on policy milestones
age_summary <- meps_ready %>%
  mutate(age_group = cut(age23x, 
                         breaks = c(0, 18, 26, 35, 45, 55, 64, 100),
                         labels = c("0-18 (Child)", "19-26 (Young Adult)", "27-35", 
                                    "36-45", "46-55", "56-64", "65+ (Medicare)"),
                         include.lowest = TRUE)) %>%
  group_by(age_group) %>%
  summarise(
    rate_uninsured = weighted.mean(unins23 == 1, w = perwt23f, na.rm = TRUE),
    n_obs = n()
  )

# 2. Plot the binned data
ggplot(age_summary, aes(x = age_group, y = rate_uninsured, fill = age_group)) +
  geom_bar(stat = "identity", color = "black", alpha = 0.8) +
  geom_text(aes(label = percent(rate_uninsured, accuracy = 0.1)), vjust = -0.5, fontface = "bold") +
  scale_y_continuous(labels = percent_format(), limits = c(0, max(age_summary$rate_uninsured) * 1.2)) +
  scale_fill_viridis_d(option = "mako", begin = 0.2, end = 0.8) +
  labs(title = "Weighted Uninsurance by Age Group",
       x = "Age Group", y = "Percent Uninsured") +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1))
