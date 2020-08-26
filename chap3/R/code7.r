x = layer_input(c(28, 28, 1))
z1 = x %>% layer_conv_2d(
  filters = 64, 
  kernel_size = c(3, 3),
  padding = "same"
)
activation1 = z1 %>% layer_activation("relu")
z2 = activation1 %>% layer_max_pooling_2d(
  pool_size = c(2, 2)
)
z3 = z2 %>% layer_conv_2d(
  filters = 32,
  kernel_size = c(3, 3),
  padding = "same"
)
activation2 = z3 %>% layer_activation("relu")
z4 = activation2 %>% layer_max_pooling_2d(
  pool_size = c(2, 2)
)
flatten = z4 %>% layer_flatten()
z5 = flatten %>% layer_dense(
  units = 128,
  activation = "relu"
)
z6 = z5 %>% layer_dense(
  units = 64,
  activation = "relu"
)
y = z6 %>% layer_dense(
  units = 10,
  activation = "softmax"
)

model = keras_model(x, y)

model

# 検証
x_train = mnist$train$x
x_test = mnist$test$x
## メソッドの指定
model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "RMSProp", 
    metrics = c("accuracy")
)

## パラメータ推定
history = model %>% fit(
    x_train, 
    y_train,
    epochs = 1,
    batch_size = 128,
    validation_data = list(x_test, y_test)
)

## 予測ラベルとエポックごとの誤差・分類精度の推移
predict = model %>% predict(x_test)
predict_label = apply(predict, 1, function(x){
    which(x == max(x)) - 1
})
plot(history)