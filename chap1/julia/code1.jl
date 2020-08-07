using GLM
using RDatasets

data = DataFrame(
    x = log.(dataset("datasets", "Cars")[:Speed]),
    y = log.(dataset("datasets", "Cars")[:Dist])
)
result = lm(@formula(y ~ x), data)
result
