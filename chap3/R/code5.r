x = layer_input(784)
z1 = x %>% layer_dense(units = 256, activation="relu")
dropout = z1 %>% layer_dropout(rate = 0.25)
z2 = dropout %>% layer_dense(units = 128, activation="relu")
y = z2 %>% layer_dense(units = 10, activation="softmax")
model = keras_model(x, y)