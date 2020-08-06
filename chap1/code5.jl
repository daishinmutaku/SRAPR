using CSV
using GLMNet
using Plots
using RDatasets
using Statistics

include("myFunc.jl")
using .MyFunc

## データの読み込み
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data"
download(url, "slump.csv")
data = select!(CSV.read("slump.csv", header=true), Not(1))

## 説明変数ベクトルと目的変数の作成
x = Matrix(data[:, 1:7])
x = scale(x)
y = data[Symbol("FLOW(cm)")]
y = y .- mean(y)
x, y

## 10分割交差検証法によるCV誤差のプロット
cvLasso = glmnetcv(x, y)
plt = plot()
plot!(log.(cvLasso.lambda), cvLasso.meanloss)
savefig("10分割交差検証法のCV誤差.pdf")

## 選択したλを用いたラッソ推定値の計算
# CV誤差を最小にするλを用いた場合
λMin = cvLasso.lambda[argmin(cvLasso.meanloss)]
glmnet(x, y, lambda=[λMin]).betas

# 1標準誤差基準によるλを用いた場合
λ1Se = cul1se(cvLasso)
glmnet(x, y, lambda=[λ1Se]).betas
