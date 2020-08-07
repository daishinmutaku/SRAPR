using Distributions
using GLMNet

include("myFunc.jl")
using .MyFunc

## データの生成(d=100, rho=0.3とした場合)
# 変数の設定
n = 100
d = 100
ρ = 0.3
s = Int(d/2)

β = zeros(d)
β[1:s] .= 2

# エラスティックネットのパラメータ
λ = 4 * sqrt(2 * log(d)/n)
α = 0.5

# 説明変数ベクトルの平均と共分散行列の作成
m = zeros(d)

tmp = MyFunc.outer(1:d, 1:d, -)
Σ = ρ .^ abs.(tmp)

x = rand(MvNormal(m, Σ), n)
y = x * β .+ rand(Normal(), n)

## エラスティックネットによるパラメータの推定
enetFit = glmnet(x, y, lambda=[λ], alpha=α)
