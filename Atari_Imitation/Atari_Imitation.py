import os
import time
import argparse

import numpy as np

from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, plot_model
from keras import backend as K
from keras.models import load_model

import gym
from gym import wrappers

from Atari_Traj_Util import load_traj_prepro, preprocess, INPUT_SHAPE, FRAME_SIZE, ENV_NAME

# TensorFlowの警告を非表示に
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 軌跡データフォルダ
PATH = None

# 学習用定数
BATCH_SIZE = 128
EPOCHS = 20

# 軌道利用割合
USE_TRAJ_RATIO = 0.01

# 前処理実行
RUN_PREPROCESS = False

# ラベルごとの重み適用
USE_CLASS_WEIGHT = False

# テストのみ実行
TEST_ONLY = False

def build_cnn_model(nb_action):
    """CNNモデル構築"""
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu", input_shape=(*INPUT_SHAPE, FRAME_SIZE)))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(nb_action, activation="softmax"))

    return model

def tarin(model, nb_action, preprocess=True):
    """学習"""
    states = None
    action = None

    if preprocess:
        # 前処理済み軌跡ロード
        states, action = load_traj_prepro(PATH, nb_action, USE_TRAJ_RATIO)

        print('-----------------------------------------------------')
        # 軌跡データ保存
        dirname = 'data_' + str(USE_TRAJ_RATIO)

        if os.path.exists(dirname) == False:
            os.mkdir(dirname)
        print('Now Save Numpy')
        np.save(dirname + '/states.npy', states)
        np.save(dirname + '/action.npy', action)
        print('End Save Numpy')
    else:
        # 保存済みデータ使用
        dirname = 'data_' + str(USE_TRAJ_RATIO)

        print('Now Load Numpy')
        states = np.load(dirname + '/states.npy')
        action = np.load(dirname + '/action.npy')
        print('End Load Numpy')

    # モデルのコンパイル
    model.compile(optimizer=Adam(),           
                    loss="categorical_crossentropy",                 
                    metrics=["accuracy"])

    #モデル保存
    #plot_model(model, 'model.png', show_shapes=True)

    # テンソルボード
    tb = TensorBoard(log_dir="./logs")

    # ラベルの重み付け
    weight = None
    if USE_CLASS_WEIGHT:
        unique, count = np.unique(action, return_counts=True)
        #print(unique)
        #print(count)
        weight = np.max(count) / count
        weight = dict(zip(unique, weight))
        #print(weight)

    # 学習
    history = model.fit(states, 
                        np_utils.to_categorical(action, nb_action),
                        batch_size=BATCH_SIZE,                       
                        epochs=EPOCHS,
                        class_weight=weight,
                        callbacks=[tb])

def test(model, env):
    """テスト"""
    
    action = 0
    
    #環境初期化
    observation = env.reset()
    while True:
        state = []

        # 初期 or 前回の状態を追加
        state.append(observation)

        for i in range(1, FRAME_SIZE):
            # 描画
            env.render()

            # 行動スキップ
            observation, _, done, _ = env.step(0)

            # 終了
            if done == True:
                break

            # 配列に追加
            state.append(observation)

        # 終了
        if done == True:
            break

        # 状態前処理
        state = preprocess(state, FRAME_SIZE)

        # 行動選択
        action = model.predict_on_batch(np.array([state]))
        action = np.argmax(action[0])

        # 行動
        observation, _, done, _ = env.step(action)
            
        #終了
        if done == True:
            break
    
def main():
    """メイン関数"""
    # gym環境指定
    env = gym.make(ENV_NAME)

    # 動画保存
    env = wrappers.Monitor(env, './movie_folder_' + str(USE_TRAJ_RATIO) + '_' + str(EPOCHS) +'_' + str(BATCH_SIZE) + ('_CW' if USE_CLASS_WEIGHT else ''), video_callable=(lambda ep: True), force=True)
    
    model = None
    
    if TEST_ONLY:
        # モデルロード
        model = load_model('model_' + str(USE_TRAJ_RATIO) + '_' + str(EPOCHS) +'_' + str(BATCH_SIZE) + ('_CW' if USE_CLASS_WEIGHT else '') +'.h5')
    else:
        # モデル構築
        model = build_cnn_model(env.action_space.n)

        # モデル学習
        tarin(model, env.action_space.n, RUN_PREPROCESS)

        # モデル保存
        model.save('model_' + str(USE_TRAJ_RATIO) + '_' + str(EPOCHS) +'_' + str(BATCH_SIZE) + ('_CW' if USE_CLASS_WEIGHT else '') +'.h5')

    # テスト
    test(model, env)

    K.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('-b', '--batchsize', type=int, default=128)
    parser.add_argument('-p', '--preprocess', action="store_true")
    parser.add_argument('-cw', '--classweight', action="store_true")
    parser.add_argument('--raito', type=float, default=0.01)
    parser.add_argument('--test', action="store_true")
    args = parser.parse_args()

    PATH = args.path
    EPOCHS = args.epoch
    BATCH_SIZE = args.batchsize
    RUN_PREPROCESS = args.preprocess
    USE_CLASS_WEIGHT = args.classweight
    USE_TRAJ_RATIO = args.raito
    TEST_ONLY = args.test

    #実行時間計測
    start_time = time.time()
    
    main()

    execution_time = time.time() - start_time
    print(execution_time)