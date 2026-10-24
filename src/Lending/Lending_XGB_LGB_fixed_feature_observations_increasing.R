#https://www.kaggle.com/competitions/microsoft-malware-prediction/data

#install.packages("Information")

# L I B R A R I E S
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
library(Information)

# L O A D   D A T A
data_original <- fread(
  "C:/Biswajit/Personal/Masters/WhitePaper/dataset/lending/loan.csv",
  select = c("loan_status","loan_amnt","int_rate","emp_length",
             "delinq_2yrs",# "annual_inc", "tot_hi_cred_lim", "acc_open_past_24mths","installment" ,"avg_cur_bal",
             "dti","emp_title","term")
)

data_original <- data_original %>%
  filter(loan_status %in% c("Fully Paid","Charged Off","Default")) %>%
  mutate(TARGET = ifelse(loan_status == "Fully Paid", 0, 1)) %>%
  select(-loan_status)

# U T I L I T Y   M E T H O D S 
get_uniq_categorical_data_count <- function(dataframe) {
  # Select only categorical columns (character or factor)
  cat_df <- data %>%
    select(where(~ is.character(.) || is.factor(.)))
  
  # If no categorical columns exist
  if (ncol(cat_df) == 0) {
    total_unique = 0
  } else {
    # Count unique values in each categorical column
    uniq_table <- cat_df %>% summarise(across(everything(), n_distinct)) %>%
      pivot_longer(
        everything(),
        names_to = "Column",
        values_to = "Unique_Count"
      ) 
    # Total unique categories across all categorical columns
    total_unique <- sum(uniq_table$Unique_Count)
  }
}


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
plot_comparison <- function(metrics_dataset, title, min, max, yAxis_label = "Metric Value", data_count) {
  
  #Reorder Metric factor based on numeric data_count
  ordered_levels <- metrics_dataset$Metric[
    order(as.numeric(data_count[metrics_dataset$Metric]))
  ]
  metrics_dataset$Metric <- factor(metrics_dataset$Metric, levels = unique(ordered_levels))
  ggplot(metrics_dataset, aes(x = Metric, y = Value, fill = Model)) +
    
    geom_col(
      position = position_dodge(width = 0.7),
      width = 0.6
    ) +
    
    geom_text(
      aes(label = round(Value, 3), group = Model),
      position = position_dodge(width = 0.7),
      vjust = 0.5,
      hjust = -0.2,
      angle = 90,
      fontface = "bold",
      size = 3
    ) +
    
    scale_x_discrete(
      labels = function(x) {
        paste0(
          format(data_count[x], scientific = FALSE),
          " (",
          format(as.numeric(unique_categorical_data_count[1, x]), scientific = FALSE), # count of unique values
          ")"
        )
      }
    ) +
    
    scale_y_continuous(
      limits = c(min, max),
      expand = expansion(mult = c(0, 0.2)) 
    ) +
    
    labs(
      title = title,
      x = "Count of observations (Unique categorical data values)",
      y = yAxis_label,
      fill = "Model"
    ) +
    
    theme_minimal() +
    
    theme(
      legend.position = "top",
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(face = "bold")
    )
}

# M O D E L   T R A I N I N G   &   E V A L U A T I O N
data_count <- c(75000, 100000, 150000, 200000)

iteration_val <- length(data_count)
execution_matrix <- data.frame()
dim(data_original)

unique_categorical_data_count <- data.frame(matrix(NA, nrow = 1, ncol = iteration_val))

names(data_count) <- paste0("Iteration_", seq_along(data_count))
names(unique_categorical_data_count) <- paste0("Iteration_", seq_along(data_count))
for (o in 1:1) { # o <- 1 i <- 1
  for (i in 1:iteration_val) {
    set.seed(123)
    data <- data_original[sample(nrow(data_original), data_count[i]), ]
    
    print('unique count')
    unique_categorical_data_count[i] <- get_uniq_categorical_data_count(data)
    print(unique_categorical_data_count )
    
    # F A C T O R I Z E   T A R G E T
    data$TARGET <- as.factor(data$TARGET)
    
    # M I S S I N G   V A L U E   I M P U T A T I O N
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
    
    
    # T R A I N - T E S T   S P L I T
    set.seed(123)
    train_index <- createDataPartition(data$TARGET, p = 0.7, list = FALSE)
    train_data <- data[train_index]
    test_data  <- data[-train_index]
    
    # Drop non-variant categorical columns
    drop_cols <- names(which(sapply(train_data, function(x) is.factor(x) && nlevels(x) < 2)))
    train_data <- train_data[, !drop_cols, with = FALSE]
    test_data  <- test_data[, !drop_cols, with = FALSE]
    
    # M O D E L   M A T R I X
    train_x <- sparse.model.matrix(TARGET ~ . - 1, data = train_data)
    test_x  <- sparse.model.matrix(TARGET ~ . - 1, data = test_data)
    
    colnames(train_x) <- clean_names(colnames(train_x))
    colnames(test_x)  <- clean_names(colnames(test_x))
    
    train_y <- as.numeric(train_data$TARGET) - 1
    test_y  <- as.numeric(test_data$TARGET) - 1
    
    # H A N D L E   C L A S S   I M B A L A N C E
    scale_pos_weight <- sum(train_y == 0) / sum(train_y == 1)
    class_counts <- table(train_data$TARGET)
    class_weights <- c(
      "0" = class_counts["1"] / sum(class_counts),
      "1" = class_counts["0"] / sum(class_counts)
    )
    
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
        min_child_weight = 20,
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
    
    execution_matrix[i, 1] <- ifelse((is.null(execution_matrix[i, 1]) || is.na(execution_matrix[i, 1])), xgb_time_taken, (execution_matrix[i, 1] + xgb_time_taken) / 2)
    execution_matrix[i, 2] <- ifelse((is.null(execution_matrix[i, 2]) || is.na(execution_matrix[i, 2])), lgb_time_taken, (execution_matrix[i, 2] + lgb_time_taken) / 2)
    
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
    
    data <- NULL
    mode_val <- train_data <- test_data <- drop_cols <- train_x <- test_x <- train_y <- test_y <- lgb_model <- lgb_pred_prob <- xgb_model <- xgb_pred_prob <- NULL    #xgb_model <- xgb_pred_prob <- xgb_time_taken <- lgb_pred_prob <- lgb_time_taken <- scale_pos_weight <- test_y <- train_y <- data <- num_cols <- cat_cols <- train_data <- test_data <- drop_cols <- train_x <- test_x <- NULL
    gc()
    Sys.sleep(20)
  }
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
plot_comparison(exec_time_plot_df_long, "Execution Time Comparison for LendingClub Dataset", 0, max(exec_time_plot_df_long$Value, na.rm = TRUE), "Execution Time (seconds)", data_count)

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
plot_comparison(perf_plot_df_long, "Performance Matrix Comparison for LendingClub Dataset", 0, max(perf_plot_df_long$Value, na.rm = TRUE), "Units", data_count)


