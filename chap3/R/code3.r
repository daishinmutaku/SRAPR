## メソッドの指定
model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "sgd", 
    metrics = c("accuracy")
)

## パラメータ推定
history = model %>% fit(
    x_train, 
    y_train,
    epochs = 30,
    batch_size = 1,
    validation_data = list(x_test, y_test)
)

## 予測ラベルとエポックごとの誤差・分類精度の推移
predict = model %>% predict(x_test)
predict_label = apply(predict, 1, function(x){
    which(x == max(x)) - 1
})
plot(history)