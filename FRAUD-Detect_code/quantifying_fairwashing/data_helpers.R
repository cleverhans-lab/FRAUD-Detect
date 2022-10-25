library(tidyverse)
library(dict)
library(argparse)

# dataset map
suffix_map <- dict()
suffix_map[["sg"]]  <- c("attack", "sg")
suffix_map[["test"]]  <- c("test", "test")


convert <- function(ll) {
  n = length(ll)
  res = rep(0, n)

  for (i in 1:n){

    if (ll[i] <= 0.5){
      res[i] <- 0  
    }
    else{
      res[i] <- 1 
    }
  }
  return(res)
}

compute_fidelity <- function(ll_ref, ll) {
  n = length(ll_ref)
  res = rep(0, n)

  for (i in 1:n){

    if (ll_ref[i] == ll[i]){
      res[i] <- 1  
    }
    else{
      res[i] <- 0
    }
  }
  return(100*mean(res))
}

#process Adult Income
process_adult_income <- function(df, df_labels) {
  df_processed = df %>%
  select(-c(income)) %>%
  mutate(
    income = df_labels$prediction
  )

  #Construct protected class variable: 0 = MAle, 1 = Female
  df_processed = df_processed %>%
  mutate(
    A = gender_Female
  ) %>%
  select(-c(gender_Female,gender_Male))

  # Construct outcome variable
  df_processed = df_processed %>%
  mutate(
    Y = case_when(
      income == 1 ~ 1, 
      income == 0 ~ 0
    )
  ) %>%
  select(-c(income))
  
  return(df_processed)
}

# process COMPAS
process_compas <- function(df, df_labels) {
  df_processed = df %>%
  select(-c(two_year_recid)) %>%
  mutate(
    two_year_recid = df_labels$prediction
  )

  #Construct protected class variable: 0 = Caucasian, 1 = African-American
  df_processed = df_processed %>%
  mutate(
    A = race_African_American
  ) %>%
  select(-c(race_African_American,race_Caucasian))

  # Construct outcome variable
  df_processed = df_processed %>%
  mutate(
    Y = case_when(
      two_year_recid == 1 ~ 1, 
      two_year_recid == 0 ~ 0
    )
  ) %>%
  select(-c(two_year_recid))
  
  return(df_processed)
}

# process Default Credit
process_default_credit <- function(df, df_labels) {
  df_processed = df %>%
  select(-c(DEFAULT_PAYEMENT)) %>%
  mutate(
    DEFAULT_PAYEMENT = df_labels$prediction
  )

  #Construct protected class variable: 0 = Male, 1 = Female
  df_processed = df_processed %>%
  mutate(
    A = SEX_Female
  ) %>%
  select(-c(SEX_Female,SEX_Male))

  # Construct outcome variable
  df_processed = df_processed %>%
  mutate(
    Y = case_when(
      DEFAULT_PAYEMENT == 1 ~ 1, 
      DEFAULT_PAYEMENT == 0 ~ 0
    )
  ) %>%
  select(-c(DEFAULT_PAYEMENT))
  
  return(df_processed)
}

# process Marketing
process_marketing <- function(df, df_labels) {
  df_processed = df %>%
  select(-c(subscribed)) %>%
  mutate(
    subscribed = df_labels$prediction
  )

  #Construct protected class variable: 0 = Male, 1 = Female
  df_processed = df_processed %>%
  mutate(
    A = age_age_not_30_60
  ) %>%
  select(-c(age_age_30_60,age_age_not_30_60))

  # Construct outcome variable
  df_processed = df_processed %>%
  mutate(
    Y = case_when(
      subscribed == 1 ~ 1, 
      subscribed == 0 ~ 0
    )
  ) %>%
  select(-c(subscribed))
  
  return(df_processed)
}

# Load data
load_data <- function(dataset, rseed, model_class, group, epsilon){
  group_file        <- sprintf("./data/preprocessed/%s/%s_%sOneHot_%s.csv", dataset, dataset, suffix_map[["sg"]][1], rseed)
  prediction_file   <- sprintf("./data/labels/%s/%s_%s_%s.csv", dataset, model_class, suffix_map[["sg"]][2], rseed)
  explainer_prediction_file   <- sprintf("./data/explainers_labels/%s/%s_eps_%s_%s.csv", dataset, model_class, epsilon, rseed)

  df        = read.csv(file = group_file)
  df_labels = read.csv(file = prediction_file)

  df_explainer_labels = read.csv(file = explainer_prediction_file)

  predictions = df_labels$prediction
  explainer_predictions = df_explainer_labels$prediction

  explainer_loss = mean(.loss_logisticRegression_helper(explainer_predictions, predictions))
  

  df_processed = NULL

  if (dataset == "adult_income"){
    df_processed = process_adult_income(df, df_labels)
  }

  if (dataset == "compas"){
    df_processed = process_compas(df, df_labels)
  }

  if (dataset == "default_credit"){
    df_processed = process_default_credit(df, df_labels)
  }

  if (dataset == "marketing"){
    df_processed = process_marketing(df, df_labels)
  }

  

  return(list("data" = df_processed, "predictions" = predictions, "loss" = explainer_loss))
}
