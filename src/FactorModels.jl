module FactorModels

using DataArrays
using DataFrames

import GLM.predict

export FactorModel, DynamicFactorModel, predict, calculate_factors, scree_plot,
    lag_vector, lagged_matrix, factor_model_DGP, normalize, pseudo_out_of_sample_forecasts, MSE, RMSE, benchmark_forecasts, benchmark_ar, diebold_mariano, matrix_to_table, print_matrix,
    targeted_predictors,
    wild_bootstrap, residual_bootstrap,
    LR_test, LM_test, LM_test_gls, Wald_test, quandt_andrews, quandt_andrews_critical_value, chow_test

include("FactorModel.jl")
include("DynamicFactorModel.jl")
include("utils.jl")
include("targeted_predictors.jl")
include("bootstrap.jl")
include("chowtest.jl")

end # module
