## function API によるモデルの定義
x = layer_input(784)
z1 = x %>% layer_dense(units = 256, activation="relu")
z2 = z1 %>% layer_dense(units = 128, activation="relu")
y = z2 %>% layer_dense(units = 10, activation="softmax")
model = keras_model(x, y)

## 上記のネットワークは, 次のように書くこともできる.
# model = keras_model_sequential()
# model %>%
# layer_dense(units = 256, activation="relu", input_shape = c(784)) %>%
# layer_dense(units = 128, activation="relu") %>%
# layer_dense(units = 10, activation="softmax")

model # modelの出力