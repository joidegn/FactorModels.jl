using DimensionalityReduction


# TODO: approximate (generalized) principal components rather than principal components could be more efficient for N > T
type FactorModel
    # models X_t = lambda*factors_t + e_t  # Here lambda is not equal to lambda(L) --> static factors
    x::Array{Float64, 2}  # variables to calculate factors from
    number_of_factors::Int64  # columns of factors we use (which e.g. capture a certain percentage of the variation or from Bai Ng info criteria)
    number_of_factors_criterion::String  # Bai Ng info criterion to use
    factor_type::String  # "principal components" later maybe also "maximum likelihood"
    targeted_predictors::BitArray  # only use set of x for factor equation which influence some y
    factors::Array{Float64, 2}  # matrix of static factors
    loadings::Array{Float64, 2}
    factor_residuals::Array{Float64, 2}  # residuals from the factor estimation  x = factors * lambda

    function FactorModel(x::Matrix{Float64}, number_of_factors::Int64=int(ceil(minimum(size(x))/2)), number_of_factors_criterion::String="", factor_type::String="principal components", targeted_predictors::BitArray=trues(size(x, 2)))
        # (static) factor equation:
        factors, loadings, number_of_factors = calculate_factors(x, factor_type, targeted_predictors, number_of_factors)
        # TODO: check if correct factors are used (p 1136 on Bai Ng 2006)
        factor_residuals = x .- factors[:, 1:number_of_factors] * loadings[:, 1:number_of_factors]'
        #factor_residuals = x - x*loadings[:, 1:number_of_factors]*inv(loadings[:, 1:number_of_factors]'loadings[:, 1:number_of_factors])*loadings[:, 1:number_of_factors]' # rescaled version from french dude
        # TODO: for lags we have to regress residuals on its lags and use info criterio to choose lag length p. this should be offloaded into another method
        # TODO: solve chicken and egg problem between number_of_factors and number_of_factor_lags. Both require residuals. --> maybe select one after the other repeatedly until convergence?
        # TODO: do the above e.g. factors, loadings, number_of_factors = augment_factors_by_lags(factors, number_of_factor_lags)  # TODO: do we have to 

        return(new(x, number_of_factors, number_of_factors_criterion, factor_type, targeted_predictors, factors, loadings, factor_residuals))
    end

    function FactorModel(x::Matrix{Float64}, number_of_factors_criterion::String, factor_type::String="principal components", targeted_predictors::BitArray=trues(size(x, 2)); max_factors = int(ceil(minimum(size(x))/2)))
        println("finding optimal FM (with maximum number of factors=", max_factors, ")")
        # number of factors to include is not given but a criterion --> we use
        # the criterion to determine the number of factors using the Bai, Ng
        # information criteria.
        models = [apply(FactorModel, (x, number_of_factors, number_of_factors_criterion, factor_type, targeted_predictors)) for number_of_factors in 1:max_factors]
        # we keep all the models in memory which can be a problem depending on the dimensions of x. TODO: will refactor later when debugging and testing is done
        criteria = [calculate_criterion(model) for model in models]
        println("criteria_values: ", criteria)
        return models[indmin(criteria[1:max_factors])]  # keep the model with the best information criterion
    end
end


# calculate the factors and the rotation matrix to transform data into the space spanned by the factors
function calculate_factors(x::Matrix, factor_type::String="principal components", targeted_predictors=trues(size(x, 2)), number_of_factors=ceil(minimum(size(x))/2))
    T, N = size(x)  # T: time dimension, N cross-sectional dimension

    function principal_components(x)  # calculate factors and loadings for a given x matrix
        # see Stock, Watson (1998)
        if T >= N
            eigen_values, eigen_vectors = eig(x'x)
            eigen_values, eigen_vectors = reverse(eigen_values), eigen_vectors[:, size(eigen_vectors, 2):-1:1]  # reverse the order from highest to lowest eigenvalue
            loadings = sqrt(N) * eigen_vectors  # we may rescale the eigenvectors say Bai, Ng 2002
            factors = x*loadings/N  # this is from Bai, Ng 2002 except that for me the inverse of the loadings matrix primed is not the same as the loadings matrix
            #factors = factors*(factors'factors/T)^(1/2)  # TODO: not sure it is correct to rescale like this (see Bai, Ng 2002 p.198)
            #factors = 1/N*x*loadings'  # this is from Stock, Watson 2010
            #loadings = (x'x)*loadings/(N*T)  # and this is from C. Hurlin from University of Orléans... rescales factor loadings to have the same variance as x
        end
        if N > T
            eigen_values, eigen_vectors = eig(x*x')
            eigen_values, eigen_vectors = reverse(eigen_values), eigen_vectors[:, size(eigen_vectors, 2):-1:1]  # reverse the order from highest to lowest eigenvalue
            factors = sqrt(T) * eigen_vectors  # sqrt(T) comes from normalization F'F/T = eye(r) where r is number of factors (see Bai 2003), factors are Txr
            loadings = x'factors/T  # see e.g. Bai 2003 p.6 or Breitung, Eickmeier 2011 p. 80. Dimension of loadings is Nxr where r here is also N
            # betahat = loadings * chol(loadings'loadings/N)  # C. Hurlin from University of Orléans rescales like this
        end
            # resultingly factors*loadings' estimates x
        return factors, loadings
    end

    if factor_type == "principal components"
        factors, loadings = principal_components(x)
        max_factor_number = int(ceil(minimum(size(x))/2))
    elseif factor_type == "squared principal components"  # include squares of X
        factors, loadings = principal_components([x[:, targeted_predictors] x[:, targeted_predictors].^2])
        max_factor_number = minimum([size(x, 1), size(x,2)*2])
    elseif factor_type == "quadratic principal components"  # include squares of X and interaction terms - better only use in combination with targeted_predictors
        pca_cols = x[:, targeted_predictors]  # columns to use for principal components
        for i in 1:size(x[:, targeted_predictors], 2)
            for j in 1:size(x[:, targeted_predictors], 2)
                pca_cols = hcat(pca_cols, x[:, i].*x[:, j])  # not very efficient!
            end
        end
        max_factor_number = int(ceil(minimum(size(pca_cols))/2))
        factors, loadings = principal_components(pca_cols)
    end
    if number_of_factors > max_factor_number
        number_of_factors = max_factor_number
        warn("can not estimate more than `minimum(size(x))` factors with $factor_type. Number of factors set to $number_of_factors")
    end
    return factors, loadings, number_of_factors
end

function Base.show(io::IO, fm::FactorModel)
    @printf io "Static Factor Model\n"
    @printf io "Dimensions of X: %s\n" size(fm.x)
    @printf io "Number of factors used: %s\n" fm.number_of_factors
    @printf io "Factors calculated by: %s\n" fm.factor_type
    @printf io "Factor model residual variance: %s\n" sum(fm.factor_residuals.^2)/apply(*, size(fm.x))
end

function calculate_criterion(fm::FactorModel)
    if fm.number_of_factors_criterion != ""
        number_of_factors_criterion = fm.number_of_factors_criterion
        number_of_factors_criterion_value::Float64 = eval(symbol("criterion_$number_of_factors_criterion"))(fm)
    end
    return(number_of_factors_criterion_value)
end

function predict(fm::FactorModel, y::Array{Float64, 1}, h::Int64=1, number_of_lags::Int64=5)
    # makes a h step ahead forecast of y (which can also be in the factor model)
    w = lagged_matrix(y, [0, [h:number_of_lags+h-1]])
    # estimate y_{t+h} = alpha'w_t + Gamma'x_t

    #WAIT A SEC... do I have to take only factors with t=... or for all of the ts?

    y, x = w[2:end, 1], hcat(w[:, 2:end], fm.factors[end-size(w,1)+1:end, 1:fm.number_of_factors])
    new_x, design_matrix = x[1,:], x[2:end, :]  # we reserve the last observation for prediction and dont use it for learning
    coefficients = inv(design_matrix'design_matrix)*design_matrix'y
    prediction = new_x*coefficients
    #residuals = y - prediction
    #hat_matrix = design_matrix*inv(design_matrix'design_matrix)*design_matrix'
    #residual_variance = (residuals.^2)./(1.-diag(hat_matrix))  # HC 2
    #coefficient_covariance = inv(design_matrix'design_matrix)*(design_matrix'diagm(residual_variance)design_matrix)*inv(design_matrix'design_matrix)
     
    return(prediction[1])
end
