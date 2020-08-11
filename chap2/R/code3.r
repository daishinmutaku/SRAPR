## R package [Modern Applied Statistics with S] を利用
library(MASS)

## 2次判別を実行
QDA = qda(grape ~ ., data = wine.train)

## 教師データに対する判別結果
qda.train = as.integer(predict(QDA)$class) # 教師データを判別
print(table(正答 = wine.train[, 1], 判別 = qda.train)) # 正誤表
qda.err.train = mean(wine.train[, 1] != qda.train) # 誤判別確率
print(qda.err.train)

## テストデータに対する判別結果
qda.test = as.integer(predict(QDA, wine.test)$class) # テストデータを判別
print(table(正答 = wine.test[, 1], 判別 = qda.test)) # 正誤表
qda.err.test = mean(wine.test[, 1] != qda.test) # 誤判別確率
print(qda.err.test)

## コード2.1で作成した格子点gridに対して判別を実行し, 判別境界を作成
qda.area = as.integer(predict(QDA, grid)$class) # 格子点に対して判別を実行
par(mfrow = c(1, 2)) # プロット画面を2分割

# 教師データの散布図
plot(
    grid, 
    # cex = 0.001, 
    col = grey(0.3 + 0.2 * qda.area),
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
        "QDA (training; n1 = ", n1.train, ", n2 = ", n2.train, ", n3 = ", n3.train, ") \n", "error rate = ", signif(qda.err.train, 4)
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
    col = grey(0.3 + 0.2 * qda.area),
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
        "QDA (test; n1 = ", n1.test, ", n2 = ", n2.test, ", n3 = ", n3.test, ") \n", "error rate = ", signif(qda.err.test, 4)
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