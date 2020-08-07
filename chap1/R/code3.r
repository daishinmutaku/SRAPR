# パッケージの呼び出し
library(glmnet)
library(MASS)

# データの生成(d=100, rho=0.3とした場合)
## 変数の設定
n <- 100; d<-100; rho <- 0.3;
s <- d/2

beta <- rep(0, d); beta[1:s] <- 2

## エラスティックネットのパラメータ
lambda <- 4 * sqrt(2 * log(d)/n)
alpha <- 0.5

## 説明変数ベクトルの平均と共分散行列の作成
m <- rep(0, d)
tmp <- outer(1:d, 1:d, "-")
Sigma <- rho^{abs(tmp)}

x <- mvrnorm(n, m, Sigma)
y <- x %*% beta + rnorm(n, 0, 1)

# エラスティックネットによるパラメータの推定
enet.fit <- glmnet(x, y, family="gaussian", lambda = lambda, alpha = alpha)
enet.fit