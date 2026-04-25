# negative value meanings: -1 (inapplicable), -2 (determined in previous round), -7 (refused),
# -8 (don't know), -10 (top coded), -15 (can't be computed)

# importing libraries
library(glmnet)
library(randomForest)
library(xgboost)
library(survey)
library(ggplot2)
library(dplyr)
library(pdp)

# read in the data
meps23 <- read.csv("meps_subset_2023.csv")

# data cleaning
# removing -1 (inapplicable), -7 (refused), -8 (don't know), -15 (can't be computed)
meps23[] <- lapply(meps23, function(x) replace(x, x %in% c(-1, -7, -8, -15), NA))

# recode 2 -> 0 for all chronic condition variables
chronic_vars <- c("HIBPDX", "CHDDX", "ANGIDX", "MIDX", "STRKDX", "DIABDX_M18", "ARTHDX", "ASTHDX", "EMPHDX")
meps23[chronic_vars][meps23[chronic_vars] == 2] <- 0

# creating chronic_sum — only count rows where we have data for ALL 9 conditions
# if any condition is NA, the sum will be NA (safer than na.rm = TRUE)
meps23$chronic_sum <- rowSums(meps23[chronic_vars], na.rm = FALSE)

# making INSCOV23 a factor (reference level = 1, any private insurance)
meps23$INSCOV23 <- as.factor(meps23$INSCOV23)

# log-transform TOTSLF23 to reduce right skew (most people spend little, few spend a lot)
# adding 1 before logging to handle $0 values since log(0) is undefined
meps23$log_TOTSLF23 <- log(meps23$TOTSLF23 + 1)

# drop rows with any NA in the variables used
model_vars <- c("log_TOTSLF23", "TOTSLF23", "chronic_sum", "POVLEV23", "INSCOV23", "DLAYCA42", "PERWT23F")
meps23_clean <- meps23[complete.cases(meps23[, model_vars]), ]
meps23_clean <- meps23_clean[meps23_clean$PERWT23F > 0, ]

# recode 2 -> 0 for no delay care
meps23_clean$DLAYCA42 <- ifelse(meps23_clean$DLAYCA42 == 1, 1, 0)

# create binary indicator
meps23_clean$any_spend <- as.integer(meps23_clean$TOTSLF23 > 0)

# survey design
meps_design <- svydesign(
  id      = ~VARPSU, 
  strata  = ~VARSTR, 
  weights = ~PERWT23F, 
  data    = meps23_clean, 
  nest    = TRUE
)

# Part 1: who spends anything at all?

part1 <- svyglm(any_spend ~ chronic_sum + POVLEV23 + INSCOV23 + DLAYCA42,
                design = meps_design,
                family = quasibinomial())

# Part 2: among spenders, how much?
meps_spenders <- meps23_clean[meps23_clean$TOTSLF23 > 0, ]
meps_design_spenders <- svydesign(
  id      = ~VARPSU, 
  strata  = ~VARSTR, 
  weights = ~PERWT23F, 
  data    = meps_spenders, 
  nest    = TRUE
)
part2 <- svyglm(log_TOTSLF23 ~ chronic_sum + POVLEV23 + INSCOV23 + DLAYCA42,
                design = meps_design_spenders)

summary(part1)
summary(part2)

# extract coefficients from both models
part1_coefs <- as.data.frame(summary(part1)$coefficients)
part2_coefs <- as.data.frame(summary(part2)$coefficients)

part1_coefs$var <- rownames(part1_coefs)
part2_coefs$var <- rownames(part2_coefs)
part1_coefs$model <- "Any Spending"
part2_coefs$model <- "Amount Among Spenders"

coef_df <- bind_rows(part1_coefs, part2_coefs) |>
  filter(var != "(Intercept)") |>
  rename(estimate = Estimate, se = `Std. Error`) |>
  mutate(
    lower = estimate - 1.96 * se,
    upper = estimate + 1.96 * se,
    var = recode(var,
                 "chronic_sum" = "Chronic Conditions",
                 "POVLEV23"    = "Poverty Level",
                 "INSCOV232"   = "Public Only",
                 "INSCOV233"   = "Uninsured",
                 "DLAYCA42"    = "Delayed Care"
    )
  )

ggplot(coef_df, aes(x = estimate, y = var, color = model)) +
  geom_point(position = position_dodge(0.5), size = 3) +
  geom_errorbarh(aes(xmin = lower, xmax = upper),
                 position = position_dodge(0.5), height = 0.2) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_color_manual(values = c("firebrick", "steelblue")) +
  labs(
    title = "Two-Part Model Coefficients",
    x = "Coefficient Estimate",
    y = NULL,
    color = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

# Histogram of out of pocket spending
hist(meps23_clean$log_TOTSLF23,
     main = "Distribution of Log Out-of-Pocket Spending",
     xlab  = "Log Total Out-of-Pocket ($)",
     col   = "firebrick")

# split data
set.seed(1000)
sample_size <- floor(0.75 * nrow(meps23_clean))
train_index <- sample(seq_len(nrow(meps23_clean)), size = sample_size)

meps_train <- meps23_clean[train_index, ]
meps_test  <- meps23_clean[-train_index, ]

x_train <- model.matrix(log_TOTSLF23 ~ chronic_sum + POVLEV23 + INSCOV23 + DLAYCA42, 
                        data = meps_train)[, -1]
x_test  <- model.matrix(log_TOTSLF23 ~ chronic_sum + POVLEV23 + INSCOV23 + DLAYCA42, 
                        data = meps_test)[, -1]

y_train <- meps_train$log_TOTSLF23
y_test  <- meps_test$log_TOTSLF23

weights_train <- meps_train$PERWT23F

# run lasso
lasso_cv <- cv.glmnet(x_train, y_train, alpha = 1, weights = weights_train)
# predict using best lambda
lasso_preds <- predict(lasso_cv, newx = x_test, s = lasso_cv$lambda.min)

# RMSE
lasso_rmse <- sqrt(mean((y_test - lasso_preds)^2))
lasso_rmse

# see which predictors LASSO kept (zeroed-out ones were dropped)
coef(lasso_cv, s = lasso_cv$lambda.min)

# run random forest
rf_model <- randomForest(x = x_train, y = y_train, ntree = 500, importance = TRUE, weights = weights_train)
rf_preds <- predict(rf_model, newdata = x_test)

rf_rmse <- sqrt(mean((y_test - rf_preds)^2))
rf_rmse

# plot variable importance
varImpPlot(rf_model)

importance(rf_model)

# set up XGBoosting
dtrain <- xgb.DMatrix(data = x_train, label = y_train, weight = weights_train)
dtest  <- xgb.DMatrix(data = x_test,  label = y_test)

xgb_model <- xgb.train(
  data    = dtrain,
  nrounds = 200,
  params  = list(
    objective = "reg:squarederror",
    eta       = 0.1,
    max_depth = 4
  ),
  evals = list(train = dtrain, test = dtest),
  verbose   = 0
)

xgb_preds <- predict(xgb_model, dtest)

xgb_rmse <- sqrt(mean((y_test - xgb_preds)^2))
xgb_rmse

xgb.importance(model = xgb_model)

# create poverty level bins
meps23_clean$pov_bin <- cut(meps23_clean$POVLEV23,
                            breaks = c(0, 100, 200, 300, 400, 500, Inf),
                            labels = c("<100%", "100-200%", "200-300%", 
                                       "300-400%", "400-500%", "500%+"))

# weighted mean log spending by bin
pov_means <- sapply(levels(meps23_clean$pov_bin), function(b) {
  sub <- meps23_clean[meps23_clean$pov_bin == b & !is.na(meps23_clean$pov_bin), ]
  weighted.mean(sub$log_TOTSLF23, sub$PERWT23F, na.rm = TRUE)
})

pov_n <- table(meps23_clean$pov_bin)
par(mfrow = c(1, 1))
plot(1:length(pov_means), pov_means,
     type = "b",
     pch  = 16,
     col  = "steelblue",
     xaxt = "n",
     xlab = "Poverty Level (% of Federal Poverty Line)",
     ylab = "Weighted Mean Log Out-of-Pocket ($)",
     main = "Out-of-Pocket Spending by Poverty Level")

axis(1, at = 1:length(pov_means), labels = names(pov_means))

# PDP for chronic_sum
pdp_chronic <- partial(rf_model, pred.var = "chronic_sum", 
                       train = x_train)

par(mfrow = c(1, 2))
plot(pdp_chronic$chronic_sum, pdp_chronic$yhat,
     type = "b", pch = 16, col = "firebrick",
     xlab = "Number of Chronic Conditions",
     ylab = "Predicted Log Out-of-Pocket ($)",
     main = "Partial Dependence: Chronic Conditions")

# PDP for POVLEV23
pdp_pov <- partial(rf_model, pred.var = "POVLEV23", 
                   train = x_train)

plot(pdp_pov$POVLEV23, pdp_pov$yhat,
     type = "l", col = "steelblue", lwd = 2,
     xlab = "Poverty Level (% of Federal Poverty Line)",
     ylab = "Predicted Log Out-of-Pocket ($)",
     main = "Partial Dependence: Poverty Level")

par(mfrow = c(1, 1))
svyboxplot(log_TOTSLF23 ~ as.factor(DLAYCA42), design = meps_design,
           main  = "Out-of-Pocket Spending by Delayed Care",
           xlab  = "Delayed Care",
           ylab  = "Log Total Out-of-Pocket ($)",
           col   = "steelblue",
           names = c("No Delay", "Delayed"))
svyboxplot(log_TOTSLF23 ~ as.factor(chronic_sum), design = meps_design,
           main  = "Out-of-Pocket Spending by Number of Chronic Conditions",
           xlab  = "Number of Chronic Conditions",
           ylab  = "Log Total Out-of-Pocket ($)",
           col   = "firebrick")

svyboxplot(log_TOTSLF23 ~ as.factor(INSCOV23), design = meps_design,
           main  = "Out-of-Pocket Spending by Insurance Type",
           xlab  = "Insurance Type",
           ylab  = "Log Total Out-of-Pocket ($)",
           col   = "steelblue",
           names = c("Any Private", "Public Only", "Uninsured"))
