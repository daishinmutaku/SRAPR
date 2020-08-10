## 事前確率を設定
pi1 = 1/3; pi2 = 1/3; pi3 = 1 - pi1 - pi2;
## x1, x2の平均ベクトルを品種ごとに推定
g = wine.train[, 1] # 品種番号の教師データ
X = as.matrix(wine.train[, 2:3]) # x1, x2の教師データ
mu1 = colMeans(X[g == 1, ]) # 品種１のx1, x2の平均ベクトル
mu2 = colMeans(X[g == 2, ]) # 品種２のx1, x2の平均ベクトル
mu3 = colMeans(X[g == 3, ]) # 品種３のx1, x2の平均ベクトル
Mu = cbind(mu1, mu2, mu3)

## x1, x2の分散共分散行列は３品種とも共通と仮定し, これを推定
# x1, x2の教師データを品種ごとに中心化
centered.X = t(sapply(1:nrow(X), function(i) {X[i, ] - Mu[, g[i]]}))
# 分散共分散行列を推定
Sigma = var(centered.X)
# 分散共分散行列の逆行列
Lambda = solve(Sigma)

## 判別ルールを関数化
LDA = function(X) {
    apply(X, 1, function(x){
        # 品種１に対する線形判別関数
        d1 = log(pi1/pi3) + t(x - (mu1 + mu3)/2) %*% Lambda %*% (mu1 - mu3)
        # 品種２に対する線形判別関数
        d2 = log(pi2/pi3) + t(x - (mu2 + mu3)/2) %*% Lambda %*% (mu2 - mu3)
        if (d1 > d2 & d1 > 0) { # d1が最大かつ正なら品種１と判別
            return(as.integer(1))
        }else if(d2 > 0){ # d2が最大かつ正なら品種２と判別
            return(as.integer(2))
        }else { # d1, d2がいずれも0以下なら品種３と判別
            return(as.integer(3))
        }
    })
}

## 教師データに対する判別結果
lda.train = LDA(wine.train[, 2:3]) # 教師データを判別
print(table(正答 = wine.train[, 1], 判別 = lda.train)) # 正誤表
lda.err.train = mean(wine.train[, 1] != lda.train) # 誤判別確率
print(lda.err.train)

## テストデータに対する判別結果
lda.test = LDA(wine.test[, 2:3]) # テストデータを判別
print(table(正答 = wine.test[, 1], 判別 = lda.test)) # 正誤表
lda.err.test = mean(wine.test[, 1] != lda.test) # 誤判別確率
print(lda.err.test)

## コード2.1で作成した格子点gridに対して判別を実行し, 判別境界を作成
lda.area = LDA(grid) # 格子点に対して判別を実行
par(mfrow = c(1, 2)) # プロット画面を2分割

# 教師データの散布図
plot(
    grid, 
    # cex = 0.001, 
    col = grey(0.3 + 0.2 * lda.area),
    xlim = r1, 
    ylim = r2,
)
par(new = TRUE)
plot(
    wine.train[, 2], 
    wine.train[, 3], 
    xlim = r1,
    ylim = r2,
    xlab = "",
    ylab = "",
    col = wine.train[, 1],
    pch = wine.train[, 1],
    main = paste0(
        "LDA (training; n1 = ", n1.train, ", n2 = ", n2.train, ", n3 = ", n3.train, ") \n", "error rate = ", signif(lda.err.train, 4)
    )
)
legend(
    "topright", 
    legend = paste0(
        "grape", 1:3
    ),
    col = 1:3,
    pch = 1:3
)

# テストデータの散布図
plot(
    grid, 
    # cex = 0.001, 
    col = grey(0.3 + 0.2 * lda.area),
    xlim = r1, 
    ylim = r2
)
par(new = TRUE)
plot(
    wine.test[, 2], 
    wine.test[, 3],
    xlim = r1, 
    ylim = r2,
    xlab = "",
    ylab = "",
    col = wine.test[, 1],
    pch = wine.test[, 1],
    main = paste0(
        "LDA (test; n1 = ", n1.test, ", n2 = ", n2.test, ", n3 = ", n3.test, ") \n", "error rate = ", signif(lda.err.test, 4)
    )
)
legend(
    "topright", 
    legend = paste0(
        "grape", 1:3
    ),
    col = 1:3,
    pch = 1:3
)
par(mfrow = c(1, 1)) # プロット画面の分割を解除
