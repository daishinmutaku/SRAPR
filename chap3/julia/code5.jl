using Flux

z1 = Dense(784, 256, relu)
dropout = Dropout(0.25)
z2 = Dense(256, 128, relu)
y = Dense(128, 10)
model = Chain(
  dropout,
  z1, 
  z2, 
  y, 
  softmax
)
