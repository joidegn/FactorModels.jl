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
end


# GLS version of LM-statistic: can be used for approximate factor models where the error term results from an AR(p) model (see Breitung, Eickmeier 2011)
function LM_test_gls(fm::FactorModel, break_period::Int64, variable_index::Int64, max_lags::Int64=10)
    # get gls transformed series for period before and after the break
    lag_polynomial = estimate_lag_polynomial(fm, variable_index, max_lags)[1]
    white_variance1, white_variance2 = estimate_white_variance_subperiods(fm, variable_index, length(lag_polynomial), break_period)
    gls_x = vcat(  # transform both subperiods separately
        gls_transform(fm.x[1:break_period, variable_index], lag_polynomial, white_variance1),
        gls_transform(fm.x[break_period+1:end, variable_index], lag_polynomial, white_variance2)
    )
    gls_factors = vcat(  # same goes for factors
        apply(hcat, [gls_transform(fm.factors[1:break_period, r], lag_polynomial, white_variance1) for r in 1:fm.number_of_factors]),
        apply(hcat, [gls_transform(fm.factors[break_period+1:end, r], lag_polynomial, white_variance2) for r in 1:fm.number_of_factors])
    )
    gls_factors_star = vcat(zeros(break_period, fm.number_of_factors), gls_factors[break_period+1:end, :])
    design_matrix = hcat(gls_factors, gls_factors_star)
    ols_estimates = inv(design_matrix'design_matrix)design_matrix'gls_x
    residuals = gls_x - design_matrix * ols_estimates
    R_squared = 1 - sum(residuals.^2) / sum(gls_x.^2)
    lagrange_multiplier = length(gls_x) * R_squared
    # TODO: this gives way too high a rejection rate
end

function gls_transform(vec::Array{Float64, 1}, polynomial::Array{Float64, 1}, variance_estimate::Float64=1)  # gls_transform a vector given a lag polynomial and an estiamtion of the variance (e.g. \hat \rho y_{it} on page 75 of Breitung, Eickmeier (2011))
    lagged_vecs = lagged_matrix(vec, [0:length(polynomial)])
    (lagged_vecs[:, 1] - sum([lagged_vecs[:, i+1]*polynomial[i] for i in 1:length(polynomial)])) / variance_estimate
end

function estimate_lag_polynomial(fm::FactorModel, variable_index::Int64, max_lags::Int64=10)  # used for GLS transformation of the model
    # Breitung, Eickmeier 2011 page 75
    # we could also use a HAC estimator (see page 75 of Breitung, Eickmeier 2011 but this can be less efficient in small samples)
    lag_num = indmin([bic_criterion(regress_on_lags(fm.factor_residuals[:, variable_index], p)[2], p) for p in 1:max_lags])
    polynomial, residuals = regress_on_lags(fm.factor_residuals[:, variable_index], lag_num)

end

function estimate_white_variance_subperiods(fm::FactorModel, variable_index::Int64, lag_number::Int64, break_period::Int64)
    coefs, residuals = regress_on_lags(fm.factor_residuals[:, variable_index], lag_number)
    x1, x2 = reshape(fm.factor_residuals[1+lag_number:break_period, variable_index], (break_period-lag_number, 1)), reshape(fm.factor_residuals[break_period+1:end, variable_index], (size(fm.x, 1)-break_period, 1))
    resid1, resid2 = residuals[1:break_period-lag_number], residuals[break_period-lag_number+1:end]
    (inv(x1'x1)*(x1'*diagm(resid1.^2)*x1)*inv(x1'x1))[1], (inv(x2'x2)*(x2'*diagm(resid2.^2)*x2)*inv(x2'x2))[1]
end

function regress_on_lags(vec::Array{Float64, 1}, p::Int64)  # returns coefficients and residuals from a regression of vec on p of its lags
    lags = lagged_matrix(vec, [0:p])
    y, x = lags[:, 1], lags[:, 2:end]
    coefs = inv(x'x)*x'y
    resid = y - x * coefs
    coefs, resid
end

function bic_criterion(residuals, k)
    T = length(residuals)
    log(var(residuals)) + k*log(T)/T # from Bai Ng (2008)
end

function quandt_andrews(fm::FactorModel, test_statistic::Function, variable_index::Int64; obs_margin_ratio=0.15)
    first_obs, last_obs = int(ceil(obs_margin_ratio*size(fm.x, 1)))+1, int(ceil((1-obs_margin_ratio)*size(fm.x, 1)))-1  # first and last obs to be tested such that there are some obs at the margins
    test_stats = [test_statistic(fm, t, variable_index) for t in first_obs:last_obs]  # note that LM_test_gls takes default value for max_lags
    stats_order = sortperm(test_stats, rev=true)
    return (first_obs+stats_order-1, test_stats[stats_order])
end

function quandt_andrews_critical_value(r::Int64, p_value::Float64=0.05)  # returns critical value of sup LM statistci for a given number of r = factors / degrees of freedom  taken from Andrews (2003)
    quandt_andrews_critical_values = [[0.01 => 12.16, 0.05 => 8.68, 0.1 => 7.12],
        [0.01 => 15.56, 0.05 => 11.72, 0.1 => 10.0],
        [0.01 => 18.07, 0.05 => 14.14, 0.1 => 12.28],
        [0.01 => 20.47, 0.05 => 16.36, 0.1 => 14.34],
        [0.01 => 22.66, 0.05 => 18.32, 0.1 => 16.30],
        [0.01 => 24.74, 0.05 => 20.24, 0.1 => 18.11],
        [0.01 => 26.72, 0.05 => 22.06, 0.1 => 19.87]
    ]  # TODO: add rest from table or write a simulation for critical values
    if r > length(quandt_andrews_critical_values)
        warn(string("r>", length(quandt_andrews_critical_values), " not implemented yet. Pull requests are welcome."))
    end
    return quandt_andrews_critical_values[r][p_value]
end

function chow_test(fm::FactorModel, test_statistic::Function, break_period::Int64, variable_index::Int64, significance_level::Float64=0.05)  # test one of the chow tests
    critical_value = quantile(Distributions.Chisq(fm.number_of_factors), 1-significance_level)
    test_statistic(fm, break_period, variable_index) > critical_value
end

function bootstrap_test(fm::FactorModel, test_statistic::Function, break_period::Int64, variable_index::Int64, bootstrap::Function=residual_bootstrap, significance_level::Float64=0.05; B=1000)  # same as chow_test but bootstraps rejections
    stat = test_statistic(fm, break_period, variable_index)
    bootstraps = bootstrap(fm, B, boot_fm->test_statistic(boot_fm, break_period, variable_index))
    p_value = mean(bootstraps .> stat)
    p_value < significance_level
end
