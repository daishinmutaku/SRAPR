using Distributions
using GLM
using GLMNet
using Statistics

include("myFunc.jl")
using .MyFunc

## データの生成(n=50とした場合)
# 変数の設定
n = 50
d = 5
β = [-2., 1., -0.5, 0., 0.]

# 説明変数ベクトルと目的変数の作成
x = rand(Normal(), (n, d))
y = x * β + rand(Normal(0, 0.3), n)

x = MyFunc.scale(x)
y = y .- mean(y)
x, y

# 適応的ラッソのパラメータ
λ = 5 / n^(3 / 4)
olsFit = lm(x, y).pp.beta0

## 適応的ラッソによるパラメータ推定
alFit = glmnet(x, y, lambda=[λ], penalty_factor=1 ./ abs.(olsFit))
