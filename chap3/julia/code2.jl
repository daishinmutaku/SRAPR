using Flux: Chain
using Flux: Dense
using NNlib: softmax
using NNlib: relu

## function API によるモデルの定義
z1 = Dense(784, 256, relu)
z2 = Dense(256, 128, relu)
y = Dense(128, 10)
model = Chain(
    z1, 
    z2, 
    y, 
    softmax
)

model