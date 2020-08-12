using Flux
using Flux.Data.MNIST
using Flux: onehotbatch

## トレーニングデータ
xTrain = MNIST.images(:train)
xTrain = hcat(float.(vec.(xTrain))...)
yTrain = MNIST.labels(:train)
yTrain = onehotbatch(yTrain, 0:9)

## テストデータ
xTest = MNIST.images(:test)
xTest = hcat(float.(vec.(xTest))...)
yTest = MNIST.labels(:test)
yTest = onehotbatch(yTest, 0:9)