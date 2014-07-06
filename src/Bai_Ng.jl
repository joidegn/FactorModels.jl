# trying to replicate Bai and Ng 2002
using FactorModels

tuples1 = [(40, 100, 1), (60, 100, 1), (60, 200, 1), (60, 500, 1), (60, 1000, 1), (60, 2000, 1), (100, 100, 1), (100, 200, 1), (100, 500, 1), (100, 1000, 1), (100, 2000, 1), (100, 40, 1), (100, 60, 1), (200, 60, 1), (500, 60, 1), (1000, 60, 1), (2000, 60, 1), (60, 4000, 1), (100, 4000, 1), (60, 8000, 1), (100, 8000, 1), (4000, 60, 1), (4000, 100, 1), (8000, 60, 1), (8000, 100, 1), (50, 10, 1), (100, 10, 1), (100, 20, 1), (10, 100, 1), (20, 100, 1)]
tuples2 = [(40, 100, 3), (60, 100, 3), (60, 200, 3), (60, 500, 3), (60, 1000, 3), (60, 2000, 3), (100, 100, 3), (100, 200, 3), (100, 500, 3), (100, 1000, 3), (100, 2000, 3), (100, 40, 3), (100, 60, 3), (200, 60, 3), (500, 60, 3), (1000, 60, 3), (2000, 60, 3), (60, 4000, 3), (100, 4000, 3), (60, 8000, 3), (100, 8000, 3), (4000, 60, 3), (4000, 100, 3), (8000, 60, 3), (8000, 100, 3), (50, 10, 3), (100, 10, 3), (100, 20, 3), (10, 100, 3), (20, 100, 3)]
tuples3 = [(40, 100, 5), (60, 100, 5), (60, 200, 5), (60, 500, 5), (60, 1000, 5), (60, 2000, 5), (100, 100, 5), (100, 200, 5), (100, 500, 5), (100, 1000, 5), (100, 2000, 5), (100, 40, 5), (100, 60, 5), (200, 60, 5), (500, 60, 5), (1000, 60, 5), (2000, 60, 5), (60, 4000, 5), (100, 4000, 5), (60, 8000, 5), (100, 8000, 5), (4000, 60, 5), (4000, 100, 5), (8000, 60, 5), (8000, 100, 5), (50, 10, 5), (100, 10, 5), (100, 20, 5), (10, 100, 5), (20, 100, 5)]

function replicate_table(sample_tuples, num_reps=1)  # replicate main information criteria of tables 1,2,3
    criteria = ["PCp1", "PCp2", "PCp3", "ICp1", "ICp2", "ICp3"]
    results = zeros(length(sample_tuples), length(criteria))
    for criterion_idx in 1:length(criteria)
        criterion = criteria[criterion_idx]
        println("crunching criterion $criterion")
        for i in 1:length(sample_tuples)
            results_per_rep = zeros(num_reps)
            for rep in 1:num_reps
                println("tuple: ", sample_tuples[i], "\t repetition: ", rep)
                y, x, f, lambda, epsilon_x, epsilon_y = apply(factor_model_DGP, sample_tuples[i])
                x = normalize(x)
                fm = FactorModel(x, criterion)
                results_per_rep[rep] = fm.number_of_factors
            end
            results[i, criterion_idx] = mean(results_per_rep)
            println("result for tuple ", sample_tuples[i], " is: ", results[i, criterion_idx])
        end
    end
    results
end


@time results1 = replicate_table(tuples1, 100); gc()
@time results2 = replicate_table(tuples2, 100); gc()
@time results3 = replicate_table(tuples3, 100); gc()
results = vcat(results1, results2, results3)
