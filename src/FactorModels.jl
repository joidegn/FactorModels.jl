module FactorModels

using DataArrays
using DataFrames

import GLM.predict

export FactorModel, DynamicFactorModel, predict, calculate_factors,
    lag_vector, factor_model_DGP, normalize, pseudo_out_of_sample_forecasts, MSE, benchmark_forecasts,
    targeted_predictors,
    wild_bootstrap, residual_bootstrap,
    LR_test, LM_test, Wald_test

include("utils.jl")
include("FactorModel.jl")
include("DynamicFactorModel.jl")
include("targeted_predictors.jl")
include("bootstrap.jl")
include("chowtest.jl")

end # module
