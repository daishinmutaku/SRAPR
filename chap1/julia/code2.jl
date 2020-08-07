using GLMNet
using Plots
using RDatasets
using Statistics

include("myFunc.jl")
using .MyFunc

## データの基準化
data = dataset("datasets", "LifeCycleSavings")
x = scale(Matrix{Float64}(data[3:6]))
y = data[2] .- mean(data[2])

## ラッソとリッジ回帰によるパラメータの推定
lasso = glmnet(x, y)
ridge = glmnet(x, y, alpha=0)

## 各推定法における解パスのプロット
plt = plot()
for i in 1:size(lasso.betas)[1]
  plot!(log.(lasso.lambda), lasso.betas[i, :], legend=nothing)
end
plt
## 
plt = plot()
for i in 1:size(ridge.betas)[1]
  plot!(log.(ridge.lambda), ridge.betas[i, :], legend=nothing)
end
plt
##

