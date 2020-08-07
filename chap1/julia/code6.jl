using Distributions
using GLMNet
using Plots
using Statistics

include("myFunc.jl")
using .MyFunc

## データの生成
# 変数の設定
n = 100
d = 20
s = 10
β = zeros(d)
β[1:s] = rand(Uniform(-1, 1), s)

# 説明変数ベクトルと目的変数の作成
σ = 2
tmp = MyFunc.outer(1:d, 1:d, -)
Σ = 0.5 .^ abs.(tmp)
x = rand(MvNormal(zeros(d), Σ), n)
x = MyFunc.scale(x)'
y = x * β + rand(Normal(0, σ), n)
y = y .- mean(y)

## 情報量規準によるモデル選択
fit = glmnet(x, y)
fitNulldev = sum([(y_i - mean(y))^2 for y_i in y])
fitDf = [count(beta -> !iszero(beta), fit.betas[:, i]) for i in 1:size(fit.betas)[2]]
cp = (1 .- fit.dev_ratio) * fitNulldev / n .+ 2 * σ^2 / n * fitDf

## 結果のプロット
λ = fit.lambda
plot(log.(λ), cp)

