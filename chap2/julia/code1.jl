using CSV
using DataFrames
using Plots
using Random
using StatsBase

## オリジナルデータをUCI ML databaseから読み込む
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
download(url, "wine.csv")
Wine = CSV.read("wine.csv", header=false)

## 使用する変数(品種, フラボノイド含有量, 色強度)を抽出
wine = Wine[:, [1, 8, 11]]
# 3変数の命名
names!(wine, [:grape, :x1, :x2])
wine[:ID] = 1:size(wine)[1]

## サンプルサイズの計算
# 品種ごと
n1 = count(grape -> (grape == 1), wine[:, 1])
n2 = count(grape -> (grape == 2), wine[:, 1])
n3 = count(grape -> (grape == 3), wine[:, 1])
# トータル
n = n1 + n2 + n3

## 乱数を用いて各品種から教師サンプルを選択(教師サンプルサイズは表2.1の設定)
# 各品種の教師サンプルサイズ
n1Train = 29
n2Train = 36
n3Train = 24
# 乱数種を指定
Random.seed!(380)
# 教師サンプル(品種1)
train1 = sample(wine[wine.grape .== 1, :].ID, n1Train, replace=false, ordered=true)
# 教師サンプル(品種2)
train2 = sample(wine[wine.grape .== 2, :].ID, n2Train, replace=false, ordered=true)
# 教師サンプル(品種3)
train3 = sample(wine[wine.grape .== 3, :].ID, n3Train, replace=false, ordered=true)
# 教師サンプル(全品種)
train = vcat((train1, train2, train3)...)

## 残りをテストサンプルとする
n1Test = n1 - n1Train
n2Test = n2 - n2Train
n3Test = n3 - n3Train
# テストサンプル(全品種)
test = setdiff(Vector(1:n), train)

## 教師データ・テストデータへ分割
wineTrain = wine[wine.ID .∈ Ref(train), :]
wineTest = wine[wine.ID .∈ Ref(test), :]

## 横軸:x1, 縦軸:x2 で散布図作成(図2.1に対応)
# x1, x2 の値域
r1 = (minimum(wine[:, 2]), maximum(wine[:, 2]))
r2 = (minimum(wine[:, 3]), maximum(wine[:, 3]))
# プロット画面を２分割
plts = Vector([])

# 教師データの散布図
plt = plot(
    title = "training; n1 = $n1Train, n2 = $n2Train, n3 = $n3Train",
    legend=:topright
)
xlims!(r1)
ylims!(r2)
xlabel!("x1")
ylabel!("x2")
for i in 1:3
    scatter!(
        wineTrain[wineTrain.grape .== i, :].x1,
        wineTrain[wineTrain.grape .== i, :].x2,
        markercolor = i,
        label = "grape$i"
    )
end
push!(plts, plt)

# 教師データの散布図
plt = plot(
    title = "test; n1 = $n1Test, n2 = $n2Test, n3 = $n3Test",
    legend=:topright
)
xlims!(r1)
ylims!(r2)
xlabel!("x1")
ylabel!("x2")
for i in 1:3
    scatter!(
        wineTest[wineTest.grape .== i, :].x1,
        wineTest[wineTest.grape .== i, :].x2,
        markercolor = i,
        label = "grape$i"
    )
end
push!(plts, plt)

## 判別境界を作図する際に使用する格子点を作成
_grid = Matrix(DataFrame(vec(collect(Iterators.product(r1[1]:(r1[2]-r1[1])/499:r1[2], r2[1]:(r2[2]-r2[1])/499:r2[2])))))