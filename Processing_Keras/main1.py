# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 
# + TensorFlow 1.8.0 インストール済み
# + keras 3.3.1 インストール済み


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

# Keras ライブラリ
import keras
from keras.models import Sequential     # Sequential model
from keras.layers import Dense          # 
from keras.layers import Activation     #
from keras import optimizers

from keras import backend as K
from keras.datasets import mnist


def main():
    """
    Keras Sequential モデルによるニューラルネットワーク
    """
    print( "Start main()" )

    # ライブラリのバージョン確認
    print( "TensorFlow version :", tf.__version__ )
    print( "Keras version :", keras.__version__ )

    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    batch_size = 128
    epoches = 10
    n_input_units = 784
    n_hidden_units = 200
    n_classes = 10

    learning_rate = 0.01

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================
    dataset_path = "C:\Data\MachineLearning_DataSet\MNIST"

    # x_train, x_test: shape (num_samples, 28, 28) の白黒画像データのuint8配列．
    # y_train, y_test: shape (num_samples,) のカテゴリラベル(0-9のinteger)のuint8配列．
    # path: データをローカルに持っていない場合 ('~/.keras/datasets/' + path) ，この位置にダウンロードされます
    (X_train, y_train),(X_test,y_test) = mnist.load_data()

    print( "X_train.shape :", X_train.shape )   # (60000, 28, 28)
    print( "y_train.shape :", y_train.shape )   # (60000,)
    print( "X_test.shape :", X_test.shape )     #
    print( "y_test.shape :", y_test.shape )     #

    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================
    # one-hot encoding のための reshape
    X_train = X_train.reshape( 60000, 784 )     # (60000, 28, 28) → (60000,784)
    X_test = X_test.reshape( 10000, 784 )

    # uint8 → float32 型に変換
    X_train = X_train.astype( "float32" )
    X_test = X_test.astype( "float32" )

    # RBG 値を 0 ~ 255 → 0.0 ~ 1.0 に変換
    X_train = X_train / 255
    X_test = X_test / 255

    # one-hot encoding
    y_train = keras.utils.to_categorical( y_train, n_classes )
    y_test = keras.utils.to_categorical( y_test, n_classes )

    #======================================================================
    # モデルの構造を定義する。
    # Define the model structure.
    #======================================================================
    # Sequential （系列）モデルは層を積み重ねてモデルを構築する。
    model = Sequential()
    #model.summary( print_fn = print() )

    #---------------------------------------------------------------
    # keras.layers.Sequential.add() メソッドで簡単にレイヤーを追加
    # この例では、パーセプトロン
    #---------------------------------------------------------------
    # keras.layers.Dense : 全結合層
    # output = activation(dot(input, kernel) + bias)
    model.add( Dense( units = n_classes, input_dim = n_input_units ) )
    model.add( Activation( "relu" ) )
    model.add( Dense( units = n_classes ) )
    model.add( Activation( "softmax" ) )

    #---------------------------------------------------------------
    # optimizer, loss を設定
    #---------------------------------------------------------------
    model.compile(
        loss = "categorical_crossentropy",                     # one-hot encoding された　cross-entropy
        optimizer = optimizers.SGD( lr = learning_rate ),      #
        metrics = ["accuracy"]
    )


    model.summary( print_fn = print() )

    #======================================================================
    # モデルの初期化と学習（トレーニング）
    #======================================================================
    # 戻り値 : History オブジェクト．
    hist = model.fit(
                x = X_train, y = y_train,
                batch_size = batch_size, epochs = epoches,
                verbose = 2,
                validation_data = ( X_test, y_test )
            )

    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================
    evals = model.evaluate( 
                x = X_test, y = y_test, 
                verbose = 1                 # 進行状況メッセージ出力モードで，0か1．
            )

    print( "loss :", evals[0] )
    print( "accuracy :", evals[1] )

    print( "hist :", hist )                         # hist  <keras.callbacks.History object at 0x000002B6D826AD68>
    print( "hist.history :", hist.history )         # 

    hist_losses = hist.history[ "loss" ]            # 学習用データでの（各エポックでの）loss 値のリスト（学習履歴）
    hist_accuracy = hist.history[ "acc" ]           # 
    hist_val_losses = hist.history[ "val_loss" ]    # 検証用データ（今の場合、テストデータ）での（各エポックでの）loss 値のリスト


    #---------------------------------------------------------
    # 損失関数を plot
    #---------------------------------------------------------
    plt.clf()
    plt.plot( 
        range( len(hist_losses) ), hist_losses,
        label = 'loss (train), learning_rate = %0.3f' % ( learning_rate ),
        linestyle = '-',
        linewidth = 0.5,
        color = 'red'
    )
    plt.plot( 
        range( len(hist_val_losses) ), hist_val_losses,
        label = 'loss (test), learning_rate = %0.3f' % ( learning_rate ),
        linestyle = '-',
        linewidth = 0.5,
        color = 'blue'
    )
    
    plt.title( "loss" )
    plt.legend( loc = 'best' )
    #plt.xlim( xmin = 0, xmax = len(ssd._losses_train) )
    plt.ylim( ymin = 0.0 )
    plt.xlabel( "epoch" )
    plt.grid()
    plt.tight_layout()
    plt.show()

    #---------------------------------------------------------
    # 正解率の算出
    #---------------------------------------------------------


    print( "Finish main()" )
    
    return


if __name__ == '__main__':
     main()
