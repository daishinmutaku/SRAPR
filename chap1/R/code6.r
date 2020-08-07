# パッケージの呼び出し
library(glmnet)
library(MASS)

# データの生成
## 変数の設定
n <- 100; d <- 20; s <- 10;
beta <- rep(0, d); beta[1:s] <- runif(s, -1, 1)

## 説明変数ヘクトルと目的変数の作成
sig <- 2; tmp <- outer(1:d, 1:d, "-"); Sig <- 0.5^{abs(tmp)}
x <- mvrnorm(n, rep(0, d), Sig); x <- scale(x)
y <- x %*% beta + rnorm(n, 0, sig); y <- y - mean(y)

# 情報量基準によるモデル選択
fit <- glmnet(x, y, family="gaussian", alpha=1)
cp <- (1 - fit$dev.ratio) * fit$nulldev / n + 2 * sig^2 / n * fit$df

# 結果のプロット
lambda <- fit$lambda 
plot(log(lambda), cp, type="b")