include("myFunc.jl")
using .MyFunc

## 2次判別を実行
function qda(X::Matrix{Float64})
    Σ1 = cov(Matrix(wineTrain[wineTrain.grape .== 1, 2:3]))
    Σ2 = cov(Matrix(wineTrain[wineTrain.grape .== 2, 2:3]))
    Σ3 = cov(Matrix(wineTrain[wineTrain.grape .== 3, 2:3]))

    function _qda(x::Vector{Float64})
        function mahalanobisDistanceSquare(μa, μb, Σi)
            (μa - μb)' * inv(Σi) * (μa - μb)
        end

        function constantTerm(πa, πb, Σa, Σb)
            log(det(Σb)/det(Σa)) + 2 * log(πa/πb)
        end

        # 品種１に対する２次判別関数
        d1 = mahalanobisDistanceSquare(x, μ3, Σ3) - mahalanobisDistanceSquare(x, μ1, Σ1) + constantTerm(π1, π3, Σ1, Σ3)
        # 品種２に対する２次判別関数
        d2 = mahalanobisDistanceSquare(x, μ3, Σ3) - mahalanobisDistanceSquare(x, μ2, Σ2) + constantTerm(π2, π3, Σ2, Σ3)
        if (d1 > d2 && d1 > 0)  # d1が最大かつ正なら品種１と判別
            return 1
        elseif(d2 > 0) # d2が最大かつ正なら品種２と判別
            return 2
        else  # d1, d2がいずれも0以下なら品種３と判別
            return 3
        end
    end

    [_qda(X[i, :]) for i in 1:size(X)[1]]
end

## 教師データに対する判別結果
qdaTrain = qda(Matrix(wineTrain[:, 2:3])) # 教師データを判別
MyFunc.printTable(Vector(wineTrain[:, 1]), qdaTrain) # 正誤表
qdaErrTrain = mean(wineTrain[:, 1] .!= qdaTrain) # 誤判別確率
print(qdaErrTrain)

## テストデータに対する判別結果
qdaTest = qda(Matrix(wineTest[:, 2:3])) # テストデータを判別
MyFunc.printTable(Vector(wineTest[:, 1]), qdaTest) # 正誤表
qdaErrTest = mean(wineTest[:, 1] .!= qdaTest) # 誤判別確率
print(qdaErrTest)

## コード2.1で作成した格子点gridに対して判別を実行し, 判別境界を作成
qdaArea = qda(_grid)

# 教師データの散布図
pltTrain = plot(
    title = "QDA (training; n1 = $n1Train, n2 = $n2Train, n3 = $n3Train) \n error rate = $qdaErrTrain",
    legend=:topright
)
contour!(
    _grid[1:500, 1], 
    _grid[1:500:250000, 2], 
    reshape(qdaArea, (500, 500))', 
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
    title = "QDA (test; n1 = $n1Test, n2 = $n2Test, n3 = $n3Test) \n error rate = $qdaErrTest",
    legend=:topright
)
contour!(
    _grid[1:500, 1], 
    _grid[1:500:250000, 2], 
    reshape(qdaArea, (500, 500))', 
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
