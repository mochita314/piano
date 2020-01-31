# coding: UTF-8

import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import read
import wave
import pyaudio

frequency = 1046.502# 生成するサイン波の周波数
seconds = 0.2      # 生成する音の秒数
rate = 44100       # 出力する wav ファイルのサンプリング周波数

phases = np.cumsum(2.0 * np.pi * frequency / rate * np.ones(int(rate * seconds)))
# 波形を生成
wave = np.sin(phases)  # -1.0 〜 1.0 の値のサイン波
# import scipy.signal して、
# wave = scipy.signal.sawtooth(phases) とすると鋸歯状波、
# wave = scipy.signal.square(phases) とすると矩形波になる

# 16bit の wav ファイルに書き出す
wave = (wave * float(2 ** 15 - 1)).astype(np.int16)  # 値域を 16bit にする
wavfile.write("C6.wav", rate, wave)

