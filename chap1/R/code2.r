# パッケージの呼び出し
library(glmnet)
library(plotmo)

# データの基準化
x <- scale(LifeCycleSavings[, 2:5])
y <- LifeCycleSavings[, 1] - mean(LifeCycleSavings[, 1])

# ラッソとリッジ回帰によるパラメータの推定
lasso <- glmnet(x, y, family = "gaussian", alpha=1)
ridge <- glmnet(x, y, family = "gaussian", alpha=0)

# 各推定法における解パスのプロット
plot_glmnet(lasso, xvar="lambda", label=TRUE)
plot_glmnet(ridge, xvar="lambda", label=TRUE)
