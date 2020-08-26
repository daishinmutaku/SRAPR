x = layer_input(784)
z1 = x %>% layer_dense(units = 256, activation="relu")
batch_norm = z1 %>% layer_batch_normalization()
z2 = batch_norm %>% layer_dense(units = 128, activation="relu")
y = z2 %>% layer_dense(units = 10, activation="softmax")
model = keras_model(x, y)
model