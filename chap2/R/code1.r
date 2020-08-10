## オリジナルデータをUCI ML databaseから読み込む
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
Wine <- read.csv(url, header=FALSE)

## 使用する変数(品種, フラボノイド含有量, 色強度)を抽出
wine = Wine[, c(1, 8, 11)]
colnames(wine) = c("grape", "x1", "x2") # 3変数の命名

## サンプルサイズの計算
# 品種ごと
n1 = sum(wine[, 1] == 1)
n2 = sum(wine[, 1] == 2)
n3 = sum(wine[, 1] == 3)
# トータル
n = n1 + n2 + n3

## 乱数を用いて各品種から教師サンプルを選択(教師サンプルサイズは表2.1の設定)
n1.train = 29; n2.train = 36; n3.train = 24; # 各品種の教師サンプルサイズ
set.seed(380) # 乱数種を指定
# 教師サンプル(品種1)
train1 = sort(sample(which(wine[, 1] == 1), n1.train))
# 教師サンプル(品種2)
train2 = sort(sample(which(wine[, 1] == 2), n2.train))
# 教師サンプル(品種3)
train3 = sort(sample(which(wine[, 1] == 3), n3.train))
# 教師サンプル(全品種)
train = c(train1, train2, train3)

## 残りをテストサンプルとする
n1.test = n1 - n1.train
n2.test = n2 - n2.train
n3.test = n3 - n3.train
# テストサンプル(全品種)
test = (1:n)[-train]

## 教師データ・テストデータへ分割
wine.train = wine[train, ]; wine.test = wine[test, ]

## 横軸:x1, 縦軸:x2 で散布図作成(図2.1に対応)
r1 = range(wine[, 2]); r2 = range(wine[, 3]) # x1, x2 の値域
par(mfrow = c(1, 2)) # プロット画面を２分割

# 教師データの散布図
plot(
    wine.train[, 2], 
    wine.train[, 3], 
    xlim = r1, 
    ylim = r2,
    xlab = "x1", 
    ylab = "x2", 
    col = wine.train[, 1], 
    pch = wine.train[, 1], 
    main = paste0(
        "training; n1 = ", n1.train, ", n2 = ", n2.train, ", n3 =", n3.train
    )
)
legend(
    "topright", 
    legend = paste0(
        "grape", 
        1:3
    ), 
    col = 1:3, 
    pch = 1:3
)

# テストデータの散布図
plot(
    wine.test[, 2], 
    wine.test[, 3], 
    xlim = r1, 
    ylim = r2,
    xlab = "x1", 
    ylab = "x2", 
    col = wine.test[, 1], 
    pch = wine.test[, 1], 
    main = paste0(
        "test; n1 = ", n1.train, ", n2 = ", n2.train, ", n3", n3.train)
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

## 判別境界を作図する際に使用する格子点を作成
grid = data.frame(
    expand.grid(
        x1 = seq(
            r1[1], 
            r1[2], 
            length.out = 500
        ), 
        x2 = seq(
            r2[1], 
            r2[2], 
            length.out = 500
        )
    )
)
