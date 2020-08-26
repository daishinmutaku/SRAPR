using Plots

## 畳み込み層の取り出し
function _layer(X)
  model[1](X)
end

## 畳み込み後の配列の作成
X = train_set[1][1][:, :, :, 123:123]
Z = _layer(X)
img = reshape(Z, (28, 28, 64))
heatmap(img[:, :, 1], color=:grays, aspect_ratio=1)