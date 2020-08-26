using Flux

z1 = Dense(784, 256)
batchNorm = BatchNorm(256, relu)
z2 = Dense(256, 128, relu)
y = Dense(128, 10)
model = Chain(
  z1, 
  batchNorm,
  z2, 
  y, 
  softmax
)
