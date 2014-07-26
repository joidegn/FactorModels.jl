using Distributions


function block_bootstrap_DGP(fm::FactorModel, factor_innovations::Array{Float64, 2}, block_size=10)
end
block_bootstrap_DGP(fm::FactorModel) = block_bootstrap_DGP(fm, apply(randn, size(fm.x)))

function parametric_bootstrap_DGP(fm::FactorModel, factor_innovations::Array{Float64, 2})
    # draw x from a parametric bootstrap with the parameters (estimated factors and loadings) given in fm
    # Note: this is a DGP for the factor equation (x = factors*loadings' + e where e are the innovations) only
    x = fm.factors*fm.loadings' + factor_innovations
end
parametric_bootstrap_DGP(fm::FactorModel) = parametric_bootstrap_DGP(fm, apply(randn, size(fm.x)))  # if no arguments are applied all innovations are assumed standard normal



# static factor models:


function residual_bootstrap(fm::FactorModel, B::Int, stat::Function)
    stats = Array(Float64, B)
    for b in 1:B
        resampled_x = fm.factors[:, 1:fm.number_of_factors]*fm.loadings[:, 1:fm.number_of_factors]' + fm.factor_residuals[rand(Distributions.DiscreteUniform(1, size(fm.x, 1)), size(fm.x, 1)), :]  # resample the residuals (by T index)
        stats[b] = stat(FactorModel(resampled_x, fm.number_of_factors, fm.number_of_factors_criterion, fm.factor_type, fm.targeted_predictors))
    end
    stats
end


function wild_bootstrap(fm::FactorModel, B::Int, stat::Function)  # resample residuals and multiply by random Variable with mean 0 and variance 1, do that B times and calculate statistic stat
    stats = Array(Float64, B)
    for b in 1:B
        resampled_x = fm.factors[:, 1:fm.number_of_factors]*fm.loadings[:, 1:fm.number_of_factors]' + apply(vcat, [fm.factor_residuals[t, :] .* randn() for t in 1:size(resampled_factor_residuals, 1)])
        #factor_residuals = fm.factor_residuals[rand(Distributions.DiscreteUniform(1, size(fm.x, 1)), size(fm.x, 1)), :]
        #resampled_x = fm.factors[:, 1:fm.number_of_factors]*fm.loadings[:, 1:fm.number_of_factors]' + apply(vcat, [resampled_factor_residuals[t, :] .* randn() for t in 1:size(resampled_factor_residuals, 1)])
        stats[b] = stat(FactorModel(resampled_x, fm.number_of_factors, fm.number_of_factors_criterion, fm.factor_type, fm.targeted_predictors))
    end
    stats
end

#y, x, f, lambdas, epsilon_x = factor_model_DGP(100, 60, 1; model="Breitung_Eickmeier_2011", b=0)
#fm = FactorModel(x)
#dfm = DynamicFactorModel((x,), 3)
