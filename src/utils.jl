using Distributions
using DataFrames
using DataArrays
using StatsBase
# allow formulae to be updated by "adding" a string to them  TODO: pull request to DataFrames.jl?
#+(formula::Formula, str::ASCIIString) = Formula(formula.lhs, convert(Symbol, *(string(formula.rhs), " + ", str)))

function lag_vector{T<:Number}(vec::Array{T,1})
    DataArray([0, vec[1:end-1]], [true, falses(length(vec)-1)])
end
function lag_vector{T<:Number}(vec::DataArray{T,1})
    DataArray([0, vec.data[1:end-1]], [true, vec.na[1:end-1]])
end
function lag_matrix{T<:Number}(matr::Array{T, 2})
    DataFrame([lag_vector(matr[:, col]) for col in 1:size(matr, 2)])
end
function lag_matrix(matr::DataFrame)
    DataFrame([lag_vector(matr[:, col]) for col in 1:size(matr, 2)])
end
function lagged_matrix(y::Array{Float64, 1}, lags::Array{Int64, 1})  # returns matrix of y and all lags asked for (cutting of the first few rows)
    lags_y = Array(DataArrays.DataArray, length(lags))
    for i in 1:length(lags)
        lags_y[i] = chain(y, lag_vector, lags[i])
    end
    apply(hcat, [lags_y[i][maximum(lags)+1:end].data for i in 1:length(lags)])
end

function chain(arg, func::Function, num_chains::Int64)  # repeatedly calls func with initial argument arg (e.g. to chain calls to lag_vector)
    for i in 1:num_chains
        arg = func(arg)  # TODO: Im not sure why apply(func, arg) does not work
    end
    arg
end

function flatten(arg, flat)
    iter_state = start(arg)
    if iter_state == false
        push!(flat, arg)  # arg cannot be flattened further
    else
        while !done(arg, iter_state)
            (item, iter_state) = next(arg, iter_state)
            flatten(item, flat)
        end
    end
    flat
end
flatten(args...) = apply(flatten, (args, Array(Any, 0)))
flatten(depth, args...) = apply(flatten, (args, Array(Any, 0)))


normalize(A::Matrix) = (A.-mean(A,1))./std(A,1) # normalize (i.e. center and rescale) Matrix A
normalize(A::Matrix, by) = (A.-by[1])./by[2] # normalize (i.e. center and rescale) Matrix A by given (mean, stddev)-tuple

function detrend(y::Array{Float64, 1})  # detrend a time series (i.e. regress on time and take residuals)
    # TODO: unfinished
end
function make_stationary(y::Array{Float64, 1})  # difference series until stationary

end


norm_vector{T<:Number}(vec::Array{T, 1}) = vec./norm(vec) # makes vector unit norm
norm_matrix{T<:Number}(mat::Array{T, 2}) = mapslices(norm_vector, mat, 2)  # call norm_vector for each column

possemidef(x) = try 
    chol(x)
    return true
catch
    return false
end

# moving average
benchmark_forecasts(x::Array{Float64, 2}, y_index::Int64; num_predictions::Int=100, window=10) = ([mean(x[end-num_predictions+i-1-window:end-num_predictions+i-1, y_index]) for i in 1:num_predictions], x[end-num_predictions+1:end, y_index])
function benchmark_ar(y::Array{Float64, 1}, p=4::Int64; num_predictions::Int64=100)  # pseudo out-of-sample forecasts of an AR(p) model with constant term
    predictions, true_values = Array(Float64,  num_predictions), Array(Float64,  num_predictions)
    for t in length(y)-num_predictions+1:length(y)  # time we still have information on (pseudo out-of-sample)
        idx = t - (length(y)-num_predictions)
        w = lagged_matrix(y[1:t], [0:p])
        yy, xx = w[1:end-1, 1], hcat(ones(size(w, 1)), w[:, 2:end])
        new_y = yy[end]
        design_matrix, new_x = xx[1:end-1, :], xx[end, :]  # we keep the last observation for forecasting
        coefficients = inv(design_matrix'design_matrix)*design_matrix'yy
        predictions[idx], true_values[idx] = (new_x*coefficients)[1], new_y
    end
    predictions, true_values
end

MSE(predictions, y) = sum((y-predictions).^2)/apply(*, size(y))
RMSE(predictions, y) = sqrt(MSE(predictions, y))

function factor_model_DGP(T::Int, N::Int, r::Int; model::String="Bai_Ng_2002", b::Number=0, delta::Number=0, default_correlation=0.1)  # T: length of series, N: number of variables, r dimension of factors, b break size
    if model=="Breitung_Kretschmer_2004"  # factors follow AR(1) process
        # TODO
    end
    if model=="Breitung_Eickmeier_2011"
        println("Generating Breitung and Eickmeier data with break b=", b)
        break_point = mod(T, 2) == 0 ? int(T/2) : int(ceil(T/2))  # note that the break occurs after the period break_point
        sigma = rand(Distributions.Uniform(0.5, 1.5), N)  # each variable has a different variance in the idiosyncratic error terms
        # note that r is equal to 1 in the paper
        f = randn(T, r)  # not specified in the paper
        Lambda = randn(N, r) .+ 1  # N(1,1)
        lambda(t, i) = t < break_point ? Lambda[i, :] : Lambda[i, :] .+ b  # in other words: there is a break in all the variables
        epsilon = apply(hcat, [randn(T)*sigma[i] for i in 1:N])
        x = Float64[(f[t, :]' * lambda(t, i))[1] for t = 1:T, i in 1:N] + epsilon # note that lambda only depends on t because of structural breaks
        return(rand(T), x, f, Lambda, epsilon)  # for this DGP y doesnt matter (Breitung and Eickmeier dont look at prediction of y)
    end

    if model=="Bai_Ng_2002"
        f = randn(T, r)
        lambda = randn(N, r)
        theta = r  # base case in Bai, Ng 2002
        epsilon_x = sqrt(theta)*randn(T, N)  # TODO: we could replace errors with AR(p) errors?
        x = f * lambda' + epsilon_x
        beta = rand(Distributions.Uniform(), r)
        epsilon_y = randn(T)  # TODO: what should epsilon be?
        y = f*beta + epsilon_y # TODO: what should beta be?
        return(y, x, f, lambda, epsilon_x, epsilon_y)
    end

    if model=="single_break"  # delta meassures correlation between y and last column (i.e. amount of "information")
        println("Generating data with break in last variable, break has size:", b, " correlation between y(first column) and last column is:", delta)
        break_point = mod(T, 2) == 0 ? int(T/2) : int(ceil(T/2))  # note that the break occurs after the period break_point
        #sigma = rand(Distributions.Uniform(0.5, 1.5), N)  # each variable has a different variance in the idiosyncratic error terms
        correlation = Float64[default_correlation for y in 1:N, x in 1:N] + diagm(Float64[1-default_correlation for i in 1:N])  # correlation matrix with some correlation for all variables, unit variance
        correlation[1,N] = correlation[N, 1] = delta
        std = chol(correlation)'
        f = randn(T, r)  # not specified in the paper
        Lambda = randn(N, r) .+ 1  # N(1,1)
        lambda(t, i) = i == r ? (t < break_point ? Lambda[i, :] : Lambda[i, :] .+ b) : Lambda[i, :]  # in other words: there is a break only in the last variable
        epsilon = randn(T, N) * std
        #x = Float64[(f[t, :]' * lambda(t, i))[1] for t = 1:T, i in 1:N] + epsilon  # note that lambda only depends on t because of structural breaks
        return(epsilon, f, Lambda, epsilon)  # TODO: this is fake
    end

end

function generate_ar(params=[0.4, 0.3, 0.2, 0.1], innovations=[])
    ar = innovations
    for i in (length(params)+1):length_series
        ar_term = (params'*ar[i-length(params):i-1])[1]
        ar[i] = ar[i] + ar_term
    end
    ar
end
generate_ar(params=[0.4, 0.3, 0.2, 0.1], size_series=(1004, )) = generate_ar(params, apply(randn, size_series))

function matrix_to_table(matrix::Matrix)  # makes a latex table from a matrix
    row_strs = Array(String, size(matrix, 1))
    for row in 1:size(matrix, 1)
        row_str = vcat([string(matrix[row, i], " & ") for i in 1:size(matrix, 2)], ["\\\\ \n"])
        row_strs[row] = apply(string, row_str)
    end
    apply(string, row_strs)
end

function print_matrix(mat)  # prints a whole matrix without the dots
    for row in 1:size(mat, 1)
        println(mat[row, :])
    end
end

function diebold_mariano(forecasts1, forecasts2, true_values, significance_level=0.05; test_type="one sided")  # could use another error function than squared error, correct for correlation, allow different forecasting window, etc...
    # taken from function dm.test from R package forecast
    residuals1, residuals2 = true_values .- forecasts1, true_values .- forecasts2
    differential = residuals1.^2 .- residuals2.^2
    d_cov = autocov(differential, [0])
    d_var = d_cov[1]  # R function corrects for covariance here but there is no covariance for h=1
    # see dm.test R-function in forecast package
    dm_stat = mean(differential)/(sqrt(d_var))
    n = length(differential)
    k = sqrt(n+1-2+(1/n))  # Harvey, Leybourne, Newbold (1997) note that h has been set to 1 here
    corrected_dm_stat = dm_stat * k
    if test_type=="one sided"
    # test whether forecasts2 are better than forecasts1
        p_value = 1 - cdf(TDist(n-1), corrected_dm_stat)
    else
        p_value = 2 * cdf(TDist(n-1), -abs(corrected_dm_stat))
    end
    corrected_dm_stat, p_value, p_value < significance_level
end
