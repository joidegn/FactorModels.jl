using LARS

function get_t_stat(y, w, x_i)
    design_matrix = hcat(ones(size(w, 1)), w[:, 2:end], x_i[size(w, 2):end])
    coefficients = inv(design_matrix'*design_matrix)*(design_matrix'*y)  # OLS estimate
    residuals = y - design_matrix * coefficients
    # calcualte variance-covariance matrix according to White(1980)
    coefficient_covariance = inv(design_matrix'design_matrix)*(design_matrix'*diagm(residuals.^2)*design_matrix)*inv(design_matrix'design_matrix)
    t_stat = coefficients./sqrt(diag(coefficient_covariance))
    last(t_stat)
end

function targeted_predictors(y_index::Int64, x::Matrix{Float64}, number_of_lags::Int64, thresholding::String="hard"; number_of_steps_in_lars=30, significance_level=0.05)
    # Bai and Ng (2008) -> regress y on w and each x_i and keep only the predictors x which are significant
    #predictors = DataFrame(hcat(y, w, x))  # unfortunately using formulae and the GLM package is a bit clumsy here
    w = lagged_matrix(x[:, y_index], [0:number_of_lags])
    y = w[:, y_index]
    if thresholding == "hard"  # TODO: hard thresholding does not work properly, only admits one variable plus the white covariance has negative diagonal elements
        critical_t_stat = quantile(TDist(size(w, 1) - (size(w, 2) + 1)), 1-significance_level/2)  # number of parameters is constant + lag number + x_i
        t_stats = Float64[get_t_stat(y, w, x[:, i]) for i in [1:size(x, 2)]]  # t_stat for y regressed on itself will be huge!
        return abs(t_stats) .> critical_t_stat
    end
    if thresholding == "soft"  # lasso coefficients which are non 0 are used
        design_matrix = hcat(x[size(w, 2):end, [1:size(x,2)] .!= y_index], w[:, 2:end])
        #res=lars(design_matrix, y, intercept=true, standardize=true, use_gram=true, maxiter=typemax(Int), lambda_min=0.0, verbose=true)
        res = lars(design_matrix, y)  # adds an intercept by default
        in_the_model = falses(size(x, 2))
        betas_added = Int[step.added for step in res.steps]
        betas_added = betas_added[betas_added.<size(x, 2)]  # only set the values for beta coefficients to true (not for coefficients of w)
        betas_added = betas_added[betas_added.>0][1:number_of_steps_in_lars]  # in some steps no variable is added
        betas_added[betas_added.>y_index] = betas_added[betas_added.>y_index] + 1  # all indices after y_index need to be shifted by 1 to account for y
        in_the_model[betas_added] = true
        in_the_model[y_index] = true
        return in_the_model
    end
end
