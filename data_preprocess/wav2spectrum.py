import matplotlib.pyplot as plt
import librosa.core as lc
import numpy as np
import librosa.display
from scipy.io import wavfile
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-n', type=int, required=True, 
    help="define the number of random integers")
args = parser.parse_args()

n = args.n



music_real = os.listdir('real_wav')
music_fake = os.listdir('fake_wav')


j = 0
for i in range(len(music_fake)):
  if i<n:
    continue

  path = 'fake_wav/' + music_real[i]
  fs, y_ = wavfile.read(path)
  fs = fs
  n_fft = 1024
  y, sr = librosa.load(path, sr=fs)
  mag = np.abs(lc.stft(y, n_fft=n_fft, hop_length=10, win_length=40, window='hamming'))        #进行短时傅里叶变换，并获取幅度
  D = librosa.amplitude_to_db(mag, ref=np.max)
  librosa.display.specshow(D, sr=fs, hop_length=10, x_axis=None, y_axis=None)
  path_save = 'fake_spec/' + str(i) + '.png'
  plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
  plt.margins(0, 0)
  plt.savefig(path_save, bbox_inches="tight", pad_inches=0)
  print("save fig" + str(i))
  j+=1

  if j>=7:
    break

# for i in range(len(music_fake)):
#   path = 'fake_wav/' + music_fake[i]
#   fs, y_ = wavfile.read(path)
#   fs = fs
#   n_fft = 1024
#   y, sr = librosa.load(path, sr=fs)
#   mag = np.abs(lc.stft(y, n_fft=n_fft, hop_length=10, win_length=40, window='hamming'))        #进行短时傅里叶变换，并获取幅度
#   D = librosa.amplitude_to_db(mag, ref=np.max)
#   librosa.display.specshow(D, sr=fs, hop_length=10, x_axis=None, y_axis=None)
#   path_save = 'fake_spec/' + str(i) + '.png'
#   plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#   plt.margins(0, 0)
#   plt.savefig(path_save, bbox_inches="tight", pad_inches=0)
#   print("save one fig")
#   gc.collect()