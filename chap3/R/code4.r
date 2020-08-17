## スキップコネクションを持つモデルの定義
x = layer_input(784)
z1 = x %>% layer_dense(
    units = 256,
    activation = "relu"
)
z2 = z1 %>% layer_dense(
    units = 128,
    activation = "relu"
)
z3 = z2 %>% layer_dense(
    units = 256,
    activation = "relu"
)
skip = layer_add(c(z1, z3))
y = z3 %>% layer_dense(
    units = 10,
    activation = "softmax"
)
model = keras_model(x, y)

model # modelの出力
