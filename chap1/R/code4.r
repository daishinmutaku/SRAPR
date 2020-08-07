# パッケージの呼び出し
library(glmnet)

# データの生成(n=50とした場合)
## 変数の設定
n <- 50; d <- 5;
beta <- c(-2, 1, -0.5, 0, 0)

## 説明変数ベクトルと目的変数の作成
x <- matrix(rnorm(n*d, 0, 1), n, d)
y <- x %*% beta + rnorm(n, 0, 0.3)

x <- scale(x); y <- y-mean(y)

## 適応的ラッソのパラメータ
lambda <- 5 / n^(3/4)
ols.fit <- lm(y ~ x)$coef[-1]

# 適応的ラッソのよるパラメータの推定
al.fit <- glmnet(x, y, family="gaussian", lambda = lambda, alpha=1, penalty.factor = 1/abs(ols.fit))
al.fit