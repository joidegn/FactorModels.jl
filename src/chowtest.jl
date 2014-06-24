using Distributions


function residuals_subperiods(fm::FactorModel, break_period, variable_index)  # gets residuals from OLS regression of x on factors for both subsamples for variable_index (i.e. column of the x matrix)
    #fm1 = FactorModel(fm.x[1:break_period, :], "ICp1")  # select number of factors according to ICp1
    #fm2 = FactorModel(fm.x[break_period+1:end, :], "ICp1")  # select number of factors according to ICp1
    x1, y1 = fm.factors[1:break_period, 1:fm.number_of_factors], fm.x[1:break_period, variable_index]
    x2, y2 = fm.factors[break_period+1:end, 1:fm.number_of_factors], fm.x[break_period+1:end, variable_index]
    #factors1, loadings1, number_of_factors1 = calculate_factors(fm.x[1:break_period, :])
    #factors2, loadings2, number_of_factors2 = calculate_factors(fm.x[break_period+1:end, :])
    #return( (fm.x[1:break_period, variable_index] - (factors1[:, 1:fm.number_of_factors]*loadings1[:, fm.number_of_factors]')[:, variable_index]), (fm.x[break_period+1:end, variable_index] - (factors2[:, 1:fm.number_of_factors]*loadings2[:, fm.number_of_factors]')[:, variable_index]) )  # TODO: number of factors is likely to be higher if a structural break is present. --> number of factors should be reestimated for subperiods
    return( ((eye(break_period) - x1*inv(x1'x1)x1')y1), ((eye(length(y2)) - x2*inv(x2'x2)x2')y2) )   # use OLS for subperiods as in Breitung Eickmeier 2011 or Factor Model as in the 2009 version of the same paper?
end


# static factor Chow-tests Breitung, Eickmeier (2011)

# LR test
function LR_test(fm::FactorModel, break_period::Int64, variable_index::Int64)  # TODO: in 2009 version of their paper Breitung and Eickmeier note that we can also use residuals from OLS regressions because they are asymptotically the same as PCA
    T = size(fm.x, 1)
    likelihood_ratio = T * (log(sum(fm.factor_residuals[:, variable_index].^2)) - log(apply(+, map(res->sum(res.^2), residuals_subperiods(fm, break_period, variable_index)))))
    return likelihood_ratio
end

function Wald_test(fm::FactorModel, break_period::Int64, variable_index::Int64)
    design_matrix = hcat(fm.factors[:, 1:fm.number_of_factors], fm.factors[:, 1:fm.number_of_factors].*[zeros(break_period), ones(size(fm.x, 1)-break_period)])
    ols_estimates = inv(design_matrix'design_matrix)design_matrix'fm.x[:, variable_index]
    residuals = fm.x[:, variable_index] - design_matrix * ols_estimates
    # calculate variance-covariance matrix according to White(1980) TODO: maybe this should be maximum likelihood estimation of var-cov matrix (i.e. negative inverse of information matrix)
    coefficient_covariance = inv(design_matrix'design_matrix)*(design_matrix'*diagm(residuals.^2)*design_matrix)*inv(design_matrix'design_matrix)
    wald_stat = (ols_estimates[fm.number_of_factors+1:end]'inv(coefficient_covariance[fm.number_of_factors+1:end, fm.number_of_factors+1:end])ols_estimates[fm.number_of_factors+1:end])[1]
    return wald_stat
end

function LM_test(fm::FactorModel, break_period::Int64, variable_index::Int64)
    design_matrix = hcat(fm.factors[:, 1:fm.number_of_factors], fm.factors[:, 1:fm.number_of_factors].*[zeros(break_period), ones(size(fm.x, 1)-break_period)])
    ols_estimates = inv(design_matrix'design_matrix)design_matrix'fm.factor_residuals[:, variable_index]
    residuals = fm.factor_residuals[:, variable_index] - design_matrix * ols_estimates
    R_squared = 1 - sum(residuals.^2) / sum(fm.factor_residuals[:, variable_index].^2)
    lagrange_multiplier = size(fm.x, 1) * R_squared
    return lagrange_multiplier
end
