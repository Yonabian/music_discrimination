#!/usr/bin/python
# -*- coding: UTF-8 -*-
import wave
import matplotlib.pyplot as plt
import numpy as np


f = wave.open('real_wav/1.wav')

# 输出信息（声道，采样宽度，帧速率，帧数）
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]

# 读取音频，字符串格式
strData = f.readframes(nframes)

# 将字符串转化为int
waveData = np.fromstring(strData,dtype=np.int16)

# wave幅值归一化
waveData = waveData*1.0/(max(abs(waveData)))
waveData = np.reshape(waveData,[nframes,nchannels])

# 画图
time = np.arange(0,nframes)*(1.0 / framerate)
plt.figure()
plt.subplot(5,1,1)
plt.plot(time,waveData[:,0])
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("")
plt.grid(True)
plt.subplot(5,1,3)
plt.plot(time,waveData[:,1])
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("Ch-2 wavedata")
plt.grid(True)
plt.savefig('test.jpg')
f.close()
