FactorModels
====
### Factor Models for Julia
[Factor models] (http://en.wikipedia.org/wiki/Dynamic_factor) or diffusion index models are statistical models which allow the estimation of a dependent variable using potentially very many regressors. Factor models are related to [factor analysis] (http://en.wikipedia.org/wiki/Factor_analysis).

This package has a strong focus on the econometric literature related to factor models. This is because I wrote the package for my master thesis. Resultingly the predict method refers to time-series data and estimates a factor augmented regression. If you feel something relevant is missing please feel free to open an issue or a pull request.

As soon as I my thesis is handed in I will provide a proper README (and add some tests).


## Installation
This package is not released as a Julia package (yet) as it is in an unfinished state. You have been warned. You can install it using 
```julia
julia> Pkg.clone("FactorModels")
```

## Usage
```julia
x = randn(50, 200)  # no problem to use more columns than rows, that is one of the nice features of factor models
fm = FactorModel(x)
fm = FactorModel(x, 5)  # use only 5 factors
fm = FactorModel(x, "ICp2")  # use one of the criteria defined by Bai, Ng (2002)
dfm = DynamicFactorModel((x, "ICp2"), 5)  # DynamicFactorModels take a tuple of arguments which is passend on to FactorModel and the number of factor lags used for prediction
```

predictions can be done using
```julia
predict(fm::FactorModel, y::Array{Float64, 1}, h::Int64=1, number_of_lags::Int64=5, number_of_factors::Int64=0)
```
or
```julia
predict(dfm::DynamicFactorModel, y::Array{Float64, 1}, h::Int64=1, number_of_lags::Int64=5, number_of_factors::Int64=0)
```

This estimates a linear model using number\_of\_lags lags of y, number\_of\_factors factors (and the number of lags thereof specified above in the case of dynamic factor models).


## Features
* Simulate and Estimate Dynamic factor models
* Estimate the number of Factors
* Use the estimation for prediction
* Preselect predictors using soft and hard thresholding (see Bai, Jushan, and Serena Ng. "Forecasting economic time series using targeted predictors." Journal of Econometrics 146.2 (2008): 304-317.)
