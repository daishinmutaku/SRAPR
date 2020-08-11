include("myFunc.jl")
using .MyFunc

## 多項ロジステック判別を実行
function logistic(X::Matrix{Float64})
    function _logistic(x::Vector{Float64})
        function ν(μa, μb)
            Λ * (μa - μb)
        end

        function ξ(πa, πb, μa, μb)
            -1 / 2 * (μa - μb)' * Λ * (μa + μb) + log(πa / πb)
        end

        # 品種１に対する２次判別関数
        d1 = ν(μ1, μ3)' * x + ξ(π1, π3, μ1, μ3) 
        # 品種２に対する２次判別関数
        d2 = ν(μ2, μ3)' * x + ξ(π2, π3, μ2, μ3) 
        if (d1 > d2 && d1 > 0)  # d1が最大かつ正なら品種１と判別
            return 1
        elseif(d2 > 0) # d2が最大かつ正なら品種２と判別
            return 2
        else  # d1, d2がいずれも0以下なら品種３と判別
            return 3
        end
    end

    [_logistic(X[i, :]) for i in 1:size(X)[1]]
end

## 教師データに対する判別結果
# 教師データを判別
logisticTrain = logistic(Matrix(wineTrain[:, 2:3])) # 教師データを判別
MyFunc.printTable(Vector(wineTrain[:, 1]), logisticTrain) # 正誤表
logisticErrTrain = mean(wineTrain[:, 1] .!= logisticTrain) # 誤判別確率
print(logisticErrTrain)

## テストデータに対する判別結果
logisticTest = logistic(Matrix(wineTest[:, 2:3])) # テストデータを判別
MyFunc.printTable(Vector(wineTest[:, 1]), logisticTest) # 正誤表
logisticErrTest = mean(wineTest[:, 1] .!= logisticTest) # 誤判別確率
print(logisticErrTest)

## コード2.1で作成した格子点gridに対して判別を実行し, 判別境界を作成
logisticArea = logistic(_grid)

# 教師データの散布図
pltTrain = plot(
    title = "Logistic (training; n1 = $n1Train, n2 = $n2Train, n3 = $n3Train) \n error rate = $logisticErrTrain",
    legend=:topright
)
contour!(
    _grid[1:500, 1], 
    _grid[1:500:250000, 2], 
    reshape(logisticArea, (500, 500))', 
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
    title = "Logistic (test; n1 = $n1Test, n2 = $n2Test, n3 = $n3Test) \n error rate = $logisticErrTest",
    legend=:topright
)
contour!(
    _grid[1:500, 1], 
    _grid[1:500:250000, 2], 
    reshape(logisticArea, (500, 500))', 
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