# パッケージの呼び出し
library(glmnet)

# データの読み込み(1行目はデータの番号なので, あらかじめ取り除く)
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data"
data <- read.csv(url, header=TRUE)[, -1]

# 説明変数ベクトルと目的変数の作成
x <- data[, 1:7]; x <- scale(x)
y <- data$FLOW.cm; y <- y - mean(y)

# 10分割交差検証法によるCV誤差のプロット
cv.lasso <- cv.glmnet(x, y, family="gaussian", alpha=1, nfolds = 10)
plot(cv.lasso)