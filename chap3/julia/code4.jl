## スキップコネクションを持つモデルの定義
z1 = Dense(784, 256, relu)
z2 = Dense(256, 128, relu)
z2 = Dense(128, 256, relu)
SkipConnection(layer, connection)
y = Dense(128, 10)
model = Chain(
    z1, 
    z2, 
    z3, 
    y, 
    softmax
)