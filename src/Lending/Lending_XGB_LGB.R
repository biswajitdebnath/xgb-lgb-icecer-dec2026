# L I B R A R I E S
# Install if needed
# install.packages(c("data.table", "caret", "xgboost", "ranger", "lightgbm", "pROC", "PRROC", "Matrix", "dplyr", "ggplot2", "readxl", "tidyr"))

library(data.table)
library(Matrix)
library(xgboost)
library(pROC)
library(ranger)
library(lightgbm)
library(dplyr)
library(readxl)
library(caret)
library(PRROC)
library(tidyr)
library(ggplot2)
library(tidyr)

## Load Data
data <- fread(
  "C:/Biswajit/personal/masters/whitepaper/dataset/lending/loan.csv", 
  select = c("loan_status","loan_amnt","int_rate","emp_length",
             "delinq_2yrs", "annual_inc", "tot_hi_cred_lim", "acc_open_past_24mths","installment" ,"avg_cur_bal",
             "zip_code", "total_bal_ex_mort", "total_il_high_credit_limit", "title", "revol_bal", "tot_cur_bal", 
             #"total_bal_il", "last_pymnt_amnt", "collection_recovery_fee", "total_pymnt", "total_pymnt_inv", "total_rec_int", "recoveries",
             "dti","emp_title","term")
)
colnames(data)
data <- data %>%
  filter(loan_status %in% c("Fully Paid","Charged Off","Default")) %>%
  mutate(TARGET = ifelse(loan_status == "Fully Paid", 0, 1)) %>%
  select(-loan_status)
set.seed(123)
data <- data[sample(nrow(data), 200000), ]
dim(data)

## Factorize Target
data$TARGET <- as.factor(data$TARGET)
#------
#View(data)
#prop.table(table(data$TARGET))
#dim(data)
#sapply(data, function(x) length(unique(x)))
sort(sapply(data, function(x) length(unique(x))), decreasing = TRUE)
#mean(is.na(data)) * 100
#------

## Missing Value Imputation
# Numeric columns -> median
num_cols <- names(which(sapply(data, is.numeric)))
for (col in num_cols) {
  data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
}
# Categorical columns -> mode
cat_cols <- names(which(sapply(data, is.character)))
for (col in cat_cols) {
  mode_val <- names(sort(table(data[[col]]), decreasing = TRUE))[1]
  data[[col]][is.na(data[[col]])] <- mode_val
}

# Convert character variables to factors
data[, (cat_cols) := lapply(.SD, as.factor), .SDcols = cat_cols]

## Define Utility Methods
# Model Performance Metrics
populate_performanceParam <- function(pred_prob, model_name, threshold = 0.5) {
  pred_label <- ifelse(pred_prob > threshold, 1, 0)
  roc_obj <- roc(test_y, pred_prob)
  auc_value <- auc(roc_obj)
  pr <- pr.curve(
    scores.class0 = pred_prob[test_y == 1],
    scores.class1 = pred_prob[test_y == 0],
    curve = TRUE
  )
  # Confusion matrix components
  TP <- sum(pred_label == 1 & test_y == 1)
  TN <- sum(pred_label == 0 & test_y == 0)
  FP <- sum(pred_label == 1 & test_y == 0)
  FN <- sum(pred_label == 0 & test_y == 1)
  
  # Metrics
  precision <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
  recall    <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
  f1_score  <- ifelse((precision + recall) == 0, 0,
                      2 * (precision * recall) / (precision + recall))
  accuracy  <- mean(pred_label == test_y)
  
  c(
    Model = model_name,
    Accuracy = mean(ifelse(pred_prob > 0.5, 1, 0) == test_y),
    ROC_AUC = auc_value,
    PR_AUC = pr$auc.integral,
    F1_SCORE = f1_score,
    ACCURACY = accuracy
  )
}  

# Clean column names for LightGBM
clean_names <- function(x) {
  x <- gsub("[^A-Za-z0-9_]", "_", x)
  x <- make.names(x, unique = TRUE)
  return(x)
}

#plotting comparison of metrics
plot_comparison <- function(metrics_dataset, seed_values, title, min, max, yAxis_label = "Metric Value", isCaption = FALSE, caption_text) {
  
  # Get unique iterations present in the dataset (e.g., "Iteration_1", "Iteration_2", etc.)
  unique_metrics <- unique(metrics_dataset$Metric)
  
  # Map each iteration to its corresponding seed value from the separate parameter
  seed_mapping <- data.frame(
    Metric = unique_metrics,
    Seed = seed_values[1:length(unique_metrics)]
  )
  
  # Merge the seed mapping back into the dataset
  metrics_dataset <- merge(metrics_dataset, seed_mapping, by = "Metric")
  
  # Create the combined x-axis label with iteration and seed
  metrics_dataset$X_Label <- paste0(
    gsub("Iteration_", "", metrics_dataset$Metric), 
    "\n(Seed: ", metrics_dataset$Seed, ")"
  )
  
  ggplot(metrics_dataset, aes(x = X_Label, y = Value, fill = Model)) +
    geom_col(
      position = position_dodge(width = 0.7),
      width = 0.6
    ) +
    geom_text(
      aes(label = round(Value, 3), group = Model),
      position = position_dodge(width = 0.7),
      vjust = 0.5,        # Centers text horizontally relative to the bar width
      hjust = -0.3,       # Negative value pushes vertical text UP and OUTSIDE the top of the bar
      angle = 90,         # Keeps numbers vertical
      fontface = "bold",
      size = 3
    ) +
    # Adds dynamic headroom so vertical text on the tallest bars never gets clipped
    expand_limits(y = max * 1.15) + 
    labs(
      title = title,
      x = "Iterations & Seeds",
      y = yAxis_label,
      fill = "Model",
      caption = if (isCaption) caption_text else NULL  
    ) +
    theme_minimal() +
    theme(
      legend.position = "top",
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(face = "bold", size = 9),
      plot.caption = element_text(hjust = 0, face = "italic", size = 9)
    )
}
iteration_val <- 3 # i <- 1
execution_matrix <- data.frame()

seeds <- c(90, 100, 123)

for (i in 1:iteration_val) {
  ## Model Training and Evaluation
  #Train-test split
  set.seed(seeds[i])
  train_index <- createDataPartition(data$TARGET, p = 0.7, list = FALSE)
  train_data <- data[train_index]
  test_data  <- data[-train_index]
  
  #Drop non-variant categorical columns
  drop_cols <- names(which(sapply(train_data, function(x) is.factor(x) && nlevels(x) < 2)))
  train_data <- train_data[, !drop_cols, with = FALSE]
  test_data  <- test_data[, !drop_cols, with = FALSE]
  
  
  #Model Matrix
  train_x <- sparse.model.matrix(TARGET ~ . - 1, data = train_data)
  test_x  <- sparse.model.matrix(TARGET ~ . - 1, data = test_data)
  
  colnames(train_x) <- clean_names(colnames(train_x))
  colnames(test_x)  <- clean_names(colnames(test_x))
  
  train_y <- as.numeric(train_data$TARGET) - 1
  test_y  <- as.numeric(test_data$TARGET) - 1
  
  ## Handle Class imbalance
  scale_pos_weight <- sum(train_y == 0) / sum(train_y == 1)
  
  # LightGBM 
  cat("Iteration", i, "Training LightGBM Model\n")
  start_time <- as.numeric(Sys.time())
  lgb_model <- lgb.train(
    params = list(
      objective = "binary",
      metric = "auc",
      learning_rate = 0.05,
      num_leaves = 64,
      min_data_in_leaf = 20,
      lambda_l1 = 0,
      lambda_l2 = 1,
      feature_fraction = 0.8,
      bagging_fraction = 0.8,
      bagging_freq = 5,
      scale_pos_weight = scale_pos_weight,
      verbosity = -1
    ),
    data = lgb.Dataset(train_x, label = train_y),
    nrounds = 1000,
    valids = list(valid = lgb.Dataset(test_x, label = test_y)),
    early_stopping_rounds = 50,
    verbose = -1
  )
  cat("Iteration", i, "Predicting LightGBM Model\n")
  lgb_pred_prob <- predict(lgb_model, test_x)
  lgb_time_taken <- as.numeric(Sys.time() - start_time)
  
  # XGBoost
  cat("Iteration", i, "Training XGBoost Model\n")
  start_time <- as.numeric(Sys.time())
  xgb_model <- xgb.train(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = 0.05,
      max_depth = 6,
      min_child_weight = 1,
      alpha = 0,
      lambda = 1,
      subsample = 0.8,
      colsample_bytree = 0.8,
      scale_pos_weight = scale_pos_weight,
      tree_method = "hist"
    ),
    data = xgb.DMatrix(train_x, label = train_y),
    nrounds = 1000,
    watchlist = list(train = xgb.DMatrix(train_x, label = train_y), 
                     valid = xgb.DMatrix(test_x, label = test_y)),
    early_stopping_rounds = 50,
    verbose = 0
  )
  cat("Iteration", i, "Predicting XGBoost Model\n")
  xgb_pred_prob <- predict(xgb_model, test_x)
  xgb_time_taken <- as.numeric(Sys.time() - start_time)
  
  execution_matrix[i, 1] <- xgb_time_taken
  execution_matrix[i, 2] <- lgb_time_taken
  
  perf <- populate_performanceParam(xgb_pred_prob, "XGBoost")
  execution_matrix[i, 3] <- perf['ROC_AUC']
  execution_matrix[i, 4] <- perf['PR_AUC']
  execution_matrix[i, 5] <- perf['F1_SCORE']
  execution_matrix[i, 6] <- perf['ACCURACY']
  
  
  perf <- populate_performanceParam(lgb_pred_prob, "LightGBM")
  execution_matrix[i, 7] <- perf['ROC_AUC']
  execution_matrix[i, 8] <- perf['PR_AUC']
  execution_matrix[i, 9] <- perf['F1_SCORE']
  execution_matrix[i, 10] <- perf['ACCURACY']
  gc()
  Sys.sleep(40)
}


colnames(execution_matrix) <- c("XGBoost", "LightGBM", "ROC XGBoost", "PR XGBoost", "F1 XGBoost", "ACCURACY XGBoost", "ROC LightGBM", "PR LightGBM", "F1 LightGBM", "ACCURACY LightGBM")
exec_time_plot_df <- data.frame(Model = c("XGBoost", "LightGBM"))

colnames_v <- vector()
for(i in 1:iteration_val) {
  col_name <- paste0("Iteration_", i)
  colnames_v[i] <- col_name
  exec_time_plot_df[[col_name]] <- as.numeric(execution_matrix[i, exec_time_plot_df$Model])
}

exec_time_plot_df_long <- exec_time_plot_df %>%
  pivot_longer(
    cols = colnames_v,
    names_to = "Metric",
    values_to = "Value"
  )
plot_comparison(exec_time_plot_df_long, seeds, "Execution Time Comparison for LendingClub Loan Dataset", 0, max(exec_time_plot_df_long$Value, na.rm = TRUE), "Execution Time (seconds)")

perf_plot_df <- data.frame(
  Model = c("ROC XGBoost", "PR XGBoost", "F1 XGBoost", "ACCURACY XGBoost", "ROC LightGBM", "PR LightGBM", "F1 LightGBM", "ACCURACY LightGBM")
) 

colnames_v <- vector()
for(i in 1:iteration_val) {
  col_name <- paste0("Iteration_", i)
  colnames_v[i] <- col_name
  perf_plot_df[[col_name]] <- as.numeric(execution_matrix[i, perf_plot_df$Model])
}

perf_plot_df_long <- perf_plot_df %>%
  pivot_longer(
    cols = colnames_v,
    names_to = "Metric",
    values_to = "Value"
  )
plot_comparison(perf_plot_df_long, seeds, "Performance Matrix Comparison for LendingClub Dataset", 0, max(perf_plot_df_long$Value, na.rm = TRUE), "Units")


# S H A P  
get_shap <- function(test_x_p, train_x_p, model_name) {
  set.seed(123)
  sample_idx <- sample(1:nrow(train_x_p), 2000)
  train_x_sample <- train_x[sample_idx, ]
  test_x_dense <- as.matrix(train_x_sample)
  colnames(test_x_dense) <- colnames(test_x_p) 
  
  #sample for SHAP if too large
  set.seed(123)
  shap_idx <- sample(seq_len(nrow(test_x_dense)), min(5000, nrow(test_x_dense)))
  if(model_name == "XGBoost") {
    shap_values <- predict(xgb_model, test_x_dense[shap_idx, ], predcontrib = TRUE)
  } else {
    shap_values <- predict(lgb_model, test_x_dense[shap_idx, ], type='contrib')
  }
  
  # Convert to data.table and remove bias 
  shap_dt <- as.data.table(shap_values)
  shap_dt <- shap_dt[, 1:(ncol(shap_dt)-1), with = FALSE] # remove bias term
  colnames(shap_dt)<- colnames(train_x_p) 
  
  # Compute mean |SHAP|
  shap_mat <- as.matrix(shap_dt)
  mean_abs_shap <- if(nrow(shap_mat) == 1) abs(shap_mat) else colMeans(abs(shap_mat))
  
  shap_importance <- data.table(
    feature = colnames(shap_mat),
    mean_abs_shap = mean_abs_shap
  )[order(-mean_abs_shap)]
  
  # P L O T – Global SHAP
  top_n <- 10
  shap_top <- shap_importance[1:top_n]
  shap_top$model <- model_name
  #shap_top$mean_abs_shap <- mean_abs_shap
  shap_top
}
xgb_importance <- get_shap(test_x, train_x, "XGBoost")
lgb_importance <- get_shap(test_x, train_x, "LightGBM")
shap_compare <- rbind(xgb_importance, lgb_importance)
shap_compare <- shap_compare %>% group_by(model) %>% mutate(feature = reorder(feature, mean_abs_shap)) %>% ungroup()
ggplot(shap_compare,
       aes(x = feature,
           y = mean_abs_shap,
           fill = model)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~ model, scales = "free_y", ncol = 2) +
  labs(
    title = "SHAP Feature Importance Comparison\n(XGBoost vs LightGBM)",
    x = "Feature",
    y = "Mean |SHAP value|"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    strip.text = element_text(face = "bold")
  )


# Platt calibration and Brier score
xgb_platt_model <- glm(test_y ~ xgb_pred_prob, family = binomial(link = "logit"))
xgb_pred_prob_calibrated <- predict(xgb_platt_model, type = "response")
lgb_platt_model <- glm(test_y ~ lgb_pred_prob, family = binomial(link = "logit"))
lgb_pred_prob_calibrated <- predict(lgb_platt_model, type = "response")


brier_xgb <- mean((xgb_pred_prob - test_y)^2)
brier_xgb_calibrated <- mean((xgb_pred_prob_calibrated - test_y)^2)
brier_lgb <- mean((lgb_pred_prob - test_y)^2)
brier_lgb_calibrated <- mean((lgb_pred_prob_calibrated - test_y)^2)

brier_df <- data.frame(
  Model = c("XGB Raw","XGB Calibrated","LGB Raw","LGB Calibrated"),
  Brier = c(brier_xgb, brier_xgb_calibrated, brier_lgb, brier_lgb_calibrated)
)
# Brier Score Comparison Plot
ggplot(brier_df, aes(x=Model, y=Brier, fill=Model)) +
  geom_col(width=0.6) +
  geom_text(aes(label=round(Brier,5)), vjust=-0.3) +
  theme_minimal() +
  labs(title="Brier Score Comparison for LendingClub dataset\n(Lower is Better)",
       y="Brier Score") +
  theme(legend.position="none",
        plot.title=element_text(hjust=0.5))

# Calibration curve
calibration_df <- data.frame(
  y = test_y,
  xgb_raw = xgb_pred_prob,
  xgb_cal = xgb_pred_prob_calibrated,
  lgb_raw = lgb_pred_prob,
  lgb_cal = lgb_pred_prob_calibrated
)

calibration_df <- calibration_df %>%
  mutate(bin = ntile(xgb_raw, 10)) %>%
  group_by(bin) %>%
  summarise(
    observed = mean(y),
    xgb_cal = mean(xgb_cal),
    lgb_cal = mean(lgb_cal)
  )

ggplot(calibration_df, aes(x=observed)) +
  geom_point(aes(y=xgb_cal), color="blue", size=3) +
  geom_line(aes(y=xgb_cal), color="blue") +
  geom_point(aes(y=lgb_cal), color="red", size=3) +
  geom_line(aes(y=lgb_cal), color="red") +
  geom_abline(slope=1, intercept=0, linetype="dashed") +
  theme_minimal() +
  labs(title="Calibration Curve (Platt Calibrated) for LendingClub dataset",
       x="Observed Default Rate",
       y="Predicted Probability") +
  annotate("text", x=0.25, y=0.1,
           label="Blue is XGB\nRed is LGB",
           hjust=1) +
  theme(
    plot.title = element_text(size = 12)
  )


