# 畳み込み層の取り出し
layer = keras_model(
  input = model$input,
  outputs = model$get_layer("conv2d")$output
)

# 畳み込み後の配列の作成
X = array_reshape(
  x_train[123, , ],
  c(1, dim(x_train[123, , ]), 1)
)
Z = layer %>% predict(X)
img = apply(Z, 4, function(x){
  matrix(
    (x - min(x)) / max(x - min(x)),
    28, 
    28
  )
})

plot(as.raster(matrix(img[, 1], 28, 28)))
