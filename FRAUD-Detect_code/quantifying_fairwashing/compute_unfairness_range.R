library(tidyverse)
library(dict)
library(argparse)

library(lattice)
library(caret)

source("core/Disparity_And_Loss_Functions.R")
source("core/Lagrangian_bestResponse_Functions.R")
source("core/ExponentiatedGradient.R")
source("core/Discretization.R")

source("data_helpers.R")


parser <- ArgumentParser()
parser$add_argument("--dataset", type="character", default="adult_income", help="Dataset name. Available options: adult_income, compas, default_credit, marketing.")
parser$add_argument("--rseed", type="integer", default=0, help="Random seed. Available options: 0 - 9.")
parser$add_argument("--bbox", type="character", default="DNN", help="Model class of the black-box. Available options: AdaBoost, DNN, RF, XgBoost.")
parser$add_argument("--pos", type="integer", default=1, help="Position of the epsilon value.")
parser$add_argument("--kl", type="double", default=0.05, help="Value of the kl constraint.")
parser$add_argument("--niters", type="integer", default=500, help="Number of iterations.")





args <- parser$parse_args()
epsilon_values_all = c(0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99)

val = epsilon_values_all[args$pos]


input_sg = load_data(args$dataset, args$rseed, args$bbox, "sg", val)
sg_data  = input_sg$data
sg_predictions  = input_sg$predictions
explainer_loss  = input_sg$loss
print("--------- explainer loss ---------------------")
print(explainer_loss)

N = 40


epsilon_values = c(explainer_loss)


# compute min_disp and max_disp
SP_extremes_analysis = tibble()


for (eps in 1:length(epsilon_values)) {
  print(sprintf("-------------------------------------------------------->>>>>> Starting epsilon = %.2f ------ \n", epsilon_values[eps]))
  
  print(sprintf("-------------------------------- Staring min ---------------------- \n"))
  SP_min_results = run_expgrad_extremes(data = sg_data, disparity_measure = "SP", 
                                        protected_class = 1, loss_function = "logistic", 
                                        learner = "logistic", eps = epsilon_values[eps], kl=args$kl, 
                                        B = 0.5*sqrt(nrow(sg_data)), N = N, n.iters = args$niters, debug = FALSE)
  
  #SP_min_reduction = reduceSolutions_extremes(costs = SP_min_results$costs, disps = SP_min_results$disps, epshat = SP_min_results$param$epshat)

  print(sprintf("-------------------------------- Staring max ---------------------- \n"))
  SP_max_results = run_expgrad_extremes(data = sg_data, disparity_measure = "SP",
                                        protected_class = 0, loss_function = "logistic",
                                        learner = "logistic", eps = epsilon_values[eps], kl=args$kl, 
                                        N = N, n.iters = args$niters, B = sqrt(nrow(sg_data)), 
                                        debug = FALSE)
  
  #SP_max_reduction = reduceSolutions_extremes(costs = SP_max_results$costs, disps = SP_max_results$disps, epshat = SP_max_results$param$epshat)

  
  # save result
  SP_extremes_analysis = bind_rows(
    SP_extremes_analysis,  
    tibble(
      id           = val,
      kl_constr    = SP_min_results$kl_constr,
      loss         = explainer_loss,
      kl_min       = mean(SP_min_results$kls),
      kl_max       = mean(SP_max_results$kls),
      min_disp     = mean(SP_min_results$disps),
      max_disp     = -mean(SP_max_results$disps),
      min_feasible = SP_min_results$feasible,
      max_feasible = SP_max_results$feasible
    )
  )
  
  #rm(SP_min_results, SP_max_results, SP_min_reduction, SP_max_reduction)
  rm(SP_min_results, SP_max_results)
}

save_path <- sprintf("results_all/%s", args$dataset)
dir.create(save_path, showWarnings = FALSE, recursive = TRUE)

filename <- sprintf("results_all/%s/dataset=%s_rseed=%s_pos=%s_bbox=%s_kl=%s_niters=%s.csv", args$dataset, args$dataset, args$rseed, args$pos, args$bbox, args$kl, args$niters)

write.csv(SP_extremes_analysis, file = filename, row.names = FALSE)



  
