## パッケージのインストール
library(tensorflow)
library(keras)

## データの取得と前処理
mnist = dataset_mnist()

# トレーニングデータ
x_train = mnist$train$x
x_train = array_reshape(x_train, c(nrow(x_train), 784)) / 255
y_train = to_categorical(mnist$train$y, 10)

# テストデータ
x_test = mnist$test$x
x_test = array_reshape(x_test, c(nrow(x_test), 784)) / 255
y_test = to_categorical(mnist$test$y, 10)
