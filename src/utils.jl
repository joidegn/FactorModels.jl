using Distributions
using DataFrames
using DataArrays
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

benchmark_forecasts(x::Array{Float64, 2}, y_index::Int64; num_predictions::Int=200) = ([mean(x[1:end-num_predictions+i, y_index]) for i in 1:num_predictions], x[end-num_predictions+1:end, y_index])

MSE(predictions, y) = sum((y-predictions).^2)/apply(*, size(y))

function factor_model_DGP(T::Int, N::Int, r::Int; model::String="Bai_Ng_2002", b=0)  # T: length of series, N: number of variables, r dimension of factors, b break size
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
        lambda(t, i) = t < break_point ? Lambda[i, :] : Lambda[i, :] .+ b
        epsilon = apply(hcat, [randn(T)*sigma[i] for i in 1:N])
        x = Float64[(f[t, :]' * lambda(t, i))[1] for t = 1:T, i in 1:N] + epsilon
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
