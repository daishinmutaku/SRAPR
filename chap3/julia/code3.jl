using Flux: crossentropy
using Flux: @epochs
using Flux: Descent
using Flux: train!
using Flux: onecold
using Flux: throttle
using Base.Iterators: partition
using Statistics
using Plots

## メソッドの指定
loss(x, y) = crossentropy(model(x), y)
optimizer = Descent()
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

## パラメータ推定
epochs = 30
# 学習が進んでいく途中での画面表示等を設定．
evalcb = () -> @show(loss(xTrain, yTrain))
batchsize = 1
serial_iterator = partition(1:size(yTrain)[2], batchsize)
train_dataset = map(batch -> (xTrain[:, batch], yTrain[:, batch]), serial_iterator);
historyTraining = zeros(epochs)
historyTest = zeros(epochs)
for i in 1:epochs
    train!(loss, params(model), train_dataset, optimizer, cb = throttle(evalcb, 10))
    historyTraining[i] = accuracy(xTrain, yTrain)
    historyTest[i] = accuracy(xTest, yTest)
end

## 予測ラベルとエポックごとの誤差・分類精度の推移
predict = model(xTest)
predictLabel = mapslices(
    x -> argmax(x) - 1, 
    predict,
    dims=1
)
plot(plot(historyTraining), plot(historyTest))
