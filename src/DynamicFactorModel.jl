type DynamicFactorModel
    # A dynamic factor model is the same as a factor model but it includes time dependence structure which we model by using lags of the static factor in the forecasting equation
    factor_model::FactorModel
    number_of_factor_lags::Int64  # lag level to use for forecasting equation

    function DynamicFactorModel(factor_model_args::Tuple)
        factor_model = FactorModel(factor_model_args...)
        number_of_factor_lags = estimate_number_of_lags(factor_model)  # if number of factor lags are not given we estimate them
        warn("Automatic recognition of the number of factor lags is not implemented yet")
        # TODO: this is undone! number_of_factor_lags has to be estimated from frequency domain transformation
        new(factor_model)
    end
    function DynamicFactorModel(factor_model_args::Tuple, number_of_factor_lags::Int64)
        factor_model = FactorModel(factor_model_args...)
        new(factor_model, number_of_factor_lags)
    end
end


# transforms x to the space spanned by the factors and optionally only selects active factors
#   type="active" returns only the active factors (which explain enough of the variance)
function get_factors(dfm::DynamicFactorModel, x::Matrix, factors="active")
    # TODO: this function makes no sense like this, there are no break indices in the model anymore
    break_indices = [0, dfm.break_indices, size(dfm.x, 1)]
    rotations = [loadings * inv(loadings'loadings) for loadings in dfm.loadings] # simplifies to loadings if T>N and loadings = inv(loadings')  which makes sense given x = F * loadings'
    [(normalize(x, (mean(dfm.factor_model.x), std(dfm.factor_model.x)))*rotations[i-1])[break_indices[i-1]+1:break_indices[i], factors=="active" ? (1:dfm.number_of_factors) : (1:end)] for i in 2:length(break_indices)]
end

function make_factor_model_design_matrix(y, number_of_lags, factors, number_of_factors, number_of_factor_lags, break_indices)
    w = lagged_matrix(y, Int64[1:number_of_lags])
    return(w[:, 1], hcat(w[:, 2:end], apply(vcat, factors)[end-size(w,1)+1:end, 1:number_of_factors]))
    # TODO: unfinished business: factor lags to get dynamic factor models (how many? see page 
end

#p_values(fm::FactorModel) = map(t_stat -> 2*(1-cdf(Distributions.TDist(size(fm.x, 1)-length(fm.coefficients)), abs(t_stat))), fm.t_stats)
#make_stars(p_value::Float64) = p_value<=0.001 ? "***" : (p_value<=0.01 ? "**" : (p_value<=0.05 ? "*" : (p_value<= 0.1 ? "." : " ")))
#add_stars(p_values::Array{Float64, 1}) = [string(p_values[i], map(make_stars, p_values)[i]) for i in 1:length(p_values)]

function Base.show(io::IO, dfm::DynamicFactorModel)
    @printf io "Dynamic Factor Model\n"
    @printf io "Dimensions of X: %s\n" size(dfm.factor_model.x)
    @printf io "Number of static factors used: %s\n" dfm.factor_model.number_of_factors
    @printf io "Number of lags used (dynamic factors): %s\n" dfm.number_of_factor_lags
    @printf io "Factors calculated by: %s\n" dfm.factor_model.factor_type
    @printf io "Factor model residual variance: %s\n" sum(dfm.factor_model.factor_residuals.^2)/apply(*, size(dfm.factor_model.x))
    # TODO: show coefficient values rather than p_values
    # TODO: visual separation between coefficients for factors and lags
end

function predict(dfm::DynamicFactorModel, y::Array{Float64, 1}, h::Int64=1, number_of_lags::Int64=5)
    # makes a h step ahead forecast of y using a linear regression on lags of y and static factors
    start_index_w = number_of_lags > dfm.number_of_factor_lags ? 1 : (dfm.number_of_factor_lags-number_of_lags+1)  # we might have to cut off the first few obs due to lagging
    start_index_factor_lags = number_of_lags > dfm.number_of_factor_lags ? (number_of_lags-dfm.number_of_factor_lags+1) : 1  # we might have to cut off the first few obs due to lagging
    w = lagged_matrix(y, [0, [h:number_of_lags+h-1]])[start_index_w:end, :]
    factor_lags = apply(hcat, [lagged_matrix(dfm.factor_model.factors[:, i], Int64[0:dfm.number_of_factor_lags]) for i in 1:dfm.factor_model.number_of_factors])[start_index_factor_lags:end, :]
    # estimate y_{t+h} = alpha'w_t + Gamma'x_t

    y, x = w[2:end, 1], hcat(w[:, 2:end], factor_lags)
    new_x, design_matrix = x[1,:], x[2:end, :]  # we reserve the last observation for prediction and dont use it for learning
    if size(design_matrix, 1) < size(design_matrix, 2)
        warn("more columns than rows in regression. Maybe try to reduce the number of common factors in the factor model.")
    end
    coefficients = inv(design_matrix'design_matrix)*design_matrix'y
    prediction = new_x*coefficients
    #residuals = y - prediction
    #hat_matrix = design_matrix*inv(design_matrix'design_matrix)*design_matrix'
    #residual_variance = (residuals.^2)./(1.-diag(hat_matrix))  # HC 2
    #coefficient_covariance = inv(design_matrix'design_matrix)*(design_matrix'diagm(residual_variance)design_matrix)*inv(design_matrix'design_matrix)
     
    return(prediction[1])
end

include("criteria.jl")  # defines the criteria in Bai and Ng 2002
