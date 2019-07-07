# Atari_Imitation
「[The Atari Grand Challenge Dataset](http://atarigrandchallenge.com/data)」を用いて、Behavior Clonning を行うプログラム

# 使い方

- エポック数20, バッチサイズ128, 前処理あり, データセット使用率1% で学習
  - `python Atari_Imitation.py "hogehoge" --epoch 20 -b 128 -p --raito 0.01`

- エポック数20, バッチサイズ128, データセット使用率1% で学習したモデルを適用
  - `python Atari_Imitation.py "hogehoge" --epoch 20 -b 128 --raito 0.01 --test`
  
- エポック数20, バッチサイズ128, 前処理なし, ClassWaight使用, データセット使用率1% で学習
  - `python Atari_Imitation.py "hogehoge" --epoch 20 -b 128 -cw --raito 0.01`
  
```

- 第一引数 'path'　データセットのルートのパス 省略不可
- '--epoch' エポック数 整数
- '-b' '--batchsize' バッチサイズ 整数
- '-p' '--preprocess' 指定すると前処理を行う 前回の前処理後のデータがあれば省略可能
- '-cw' '--classweight' 指定すると学習時にラベルごとに重みをつける
- '--raito' データセットの使用率
- '--test' 学習済みで実行

```

実行すると `data_{データセット使用率}`のフォルダが生成される。その中に前処理後stateとactionのnumpy配列が生成される

学習済みモデルは 同一階層に `data_{データセット使用率}_{エポック数}_{バッチサイズ}_{ClassWaight有無}.h5`と保存される。

# 動作環境

- Windows 10 HOME
- anaconda==4.2.0
- Python==3.5.5
- Keras==2.1.6
- tensorflow-gpu==1.8.0
- gym==0.10.5
- cuda==v9.0
- cuDNN==7.0
- GPU: GeForce GTX 1070

# サンプル

- エポック数20, バッチサイズ128, データセット0.05% 使用、ClassWaight使用

  <img src="https://qiita-image-store.s3.amazonaws.com/0/252166/58957189-d593-4f1f-ae8c-b296b9d7a15e.gif" width=30%>
