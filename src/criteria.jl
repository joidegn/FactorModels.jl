factor_residual_variance(fm::FactorModel) = sum(fm.factor_residuals.^2)/apply(*, size(fm.x))  # see page 201 of Bai Ng 2002
#factor_residual_variance(fm::FactorModel) = sum(mapslices(x->x'x/length(x), fm.factor_residuals, 1))/size(fm.x, 2)  # the same as above
# and var(fm.factor_residuals) is approximately the same as well but it probably corrects for small sample bias



function criterion_cumulative_variance(pca_result, threshold=0.95)   # simply use factors until a certain threshold of variance is reached
    pca_result.cumulative_variance .< threshold  # take threshold% of variance 
end

# PCp criteria as defined on page 201 of Bai and Ng 2002
function criterion_PCp1(fm::FactorModel; kmax=8)  # be careful! kmax is set as in Bai, Ng 2002 which might make no sense in a different setting!
    fm_unrestricted = FactorModel(fm.x, kmax)
    N_plus_T_by_NT = apply(+, size(fm.x))/apply(*, size(fm.x))
    factor_residual_variance(fm) + fm.number_of_factors*factor_residual_variance(fm_unrestricted)*(N_plus_T_by_NT)*log(N_plus_T_by_NT^-1)
end
function criterion_PCp2(fm::FactorModel; kmax=8)
    fm_unrestricted = FactorModel(fm.x, kmax)  # kmax is set as in Bai, Ng 2002
    N_plus_T_by_NT = apply(+, size(fm.x))/apply(*, size(fm.x))
    factor_residual_variance(fm) + sum(fm.number_of_factors)*factor_residual_variance(fm_unrestricted)*(N_plus_T_by_NT)*log(minimum(size(fm.x)))
end
function criterion_PCp3(fm::FactorModel; kmax=8)
    fm_unrestricted = FactorModel(fm.x, kmax)   # kmax is set as in Bai, Ng 2002
    N_plus_T_by_NT = apply(+, size(fm.x))/apply(*, size(fm.x))
    factor_residual_variance(fm) + sum(fm.number_of_factors)*factor_residual_variance(fm_unrestricted)*log(minimum(size(fm.x)))/minimum(size(fm.x))
end


# ICp criteria as defined on page 201 of Bai and Ng 2002
function criterion_ICp1(fm::FactorModel)
    N_plus_T_by_NT = apply(+, size(fm.x))/apply(*, size(fm.x))
    log(factor_residual_variance(fm)) + sum(fm.number_of_factors)*(N_plus_T_by_NT)*log(N_plus_T_by_NT^-1)
end
function criterion_ICp2(fm::FactorModel)
    N_plus_T_by_NT = apply(+, size(fm.x))/apply(*, size(fm.x))
    log(factor_residual_variance(fm)) + sum(fm.number_of_factors)*(N_plus_T_by_NT)*log(minimum(size(fm.x)))
end
function criterion_ICp3(fm::FactorModel)
    N_plus_T_by_NT = apply(+, size(fm.x))/apply(*, size(fm.x))
    log(factor_residual_variance(fm)) + sum(fm.number_of_factors)*log(minimum(size(fm.x)))/minimum(size(fm.x))
end


#  default criteria which are not quite consistent at least not when the number of factors is estimated
function criterion_BIC(fm::FactorModel)
    T, N = size(fm.x)
    factor_residual_variance(fm) + fm.number_of_factors * log(T)/T
end
