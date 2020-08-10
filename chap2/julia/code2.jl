using Statistics
using LinearAlgebra
using PrettyTables

## 事前確率を設定
π1 = 1/3
π2 = 1/3
π3 = 1 - π1 - π2
## x1, x2の平均ベクトルを品種ごとに推定
g = wineTrain[:, 1] # 品種番号の教師データ
X = Matrix(wineTrain[:, 2:3]) # x1, x2の教師データ
μ1 = describe(wineTrain[wineTrain.grape .== 1, 2:3]).mean # 品種１のx1, x2の平均ベクトル
μ2 = describe(wineTrain[wineTrain.grape .== 2, 2:3]).mean # 品種２のx1, x2の平均ベクトル
μ3 = describe(wineTrain[wineTrain.grape .== 3, 2:3]).mean # 品種３のx1, x2の平均ベクトル
M = [μ1'; μ2'; μ3']

## x1, x2の分散共分散行列は３品種とも共通と仮定し, これを推定
# x1, x2の教師データを品種ごとに中心化
centeredX = map(i -> X[i, :] - M[wineTrain[i, 1], :], 1:size(X)[1])
# 分散共分散行列を推定
Σ = cov(centeredX)
# 分散共分散行列の逆行列
Λ = inv(Σ)

## 判別ルールを関数化
function LDA(X::Matrix{Float64})
    function lda(x::Vector{Float64})
        # 品種１に対する線形判別関数
        d1 = log(π1/π3) + (x - (μ1 + μ3)/2)' * Λ * (μ1 - μ3)
        # 品種２に対する線形判別関数
        d2 = log(π2/π3) + (x - (μ2 + μ3)/2)' * Λ * (μ2 - μ3)
        if (d1 > d2 && d1 > 0)  # d1が最大かつ正なら品種１と判別
            return 1
        elseif(d2 > 0) # d2が最大かつ正なら品種２と判別
            return 2
        else  # d1, d2がいずれも0以下なら品種３と判別
            return 3
        end
    end

    [lda(X[i, :]) for i in 1:size(X)[1]]
end

## 教師データに対する判別結果
ldaTrain = LDA(Matrix(wineTrain[:, 2:3])) # 教師データを判別
function printTable(correct::Vector{Int}, discrimination::Vector{Int})
    table = zeros((3, 3))
    for i in 1:size(correct)[1]
        c = correct[i]
        d = discrimination[i]
        table[c, d] += 1
    end
    pretty_table(table)
end
printTable(Vector(wineTrain[:, 1]), ldaTrain) # 正誤表
ldaErrTrain = mean(wineTrain[:, 1] .!= ldaTrain) # 誤判別確率
print(ldaErrTrain)

## テストデータに対する判別結果
ldaTest = LDA(Matrix(wineTest[:, 2:3])) # テストデータを判別
function printTable(correct::Vector{Int}, discrimination::Vector{Int})
    table = zeros((3, 3))
    for i in 1:size(correct)[1]
        c = correct[i]
        d = discrimination[i]
        table[c, d] += 1
    end
    pretty_table(table)
end
printTable(Vector(wineTest[:, 1]), ldaTest) # 正誤表
ldaErrTest = mean(wineTest[:, 1] .!= ldaTest) # 誤判別確率
print(ldaErrTest)

## コード2.1で作成した格子点gridに対して判別を実行し, 判別境界を作成
ldaArea = LDA(_grid)

# 教師データの散布図
pltTrain = plot(
    title = "LDA (training; n1 = $n1Train, n2 = $n2Train, n3 = $n3Train) \n error rate = $ldaErrTrain",
    legend=:topright
)
contour!(
    _grid[1:500, 1], 
    _grid[1:500:250000, 2], 
    reshape(ldaArea, (500, 500))', 
    fill=(true,cgrad(:grays))
)
xlims!(r1)
ylims!(r2)
xlabel!("")
ylabel!("")
for i in 1:3
    scatter!(
        wineTrain[wineTrain.grape .== i, :].x1,
        wineTrain[wineTrain.grape .== i, :].x2,
        markercolor = i,
        label = "grape$i"
    )
end

# テストデータの散布図
pltTest = plot(
    title = "LDA (test; n1 = $n1Test, n2 = $n2Test, n3 = $n3Test) \n error rate = $ldaErrTest",
    legend=:topright
)
contour!(
    _grid[1:500, 1], 
    _grid[1:500:250000, 2], 
    reshape(ldaArea, (500, 500))', 
    fill=(true,cgrad(:grays))
)
xlims!(r1)
ylims!(r2)
xlabel!("")
ylabel!("")
for i in 1:3
    scatter!(
        wineTest[wineTest.grape .== i, :].x1,
        wineTest[wineTest.grape .== i, :].x2,
        markercolor = i,
        label = "grape$i"
    )
end
plot(pltTrain, pltTest)