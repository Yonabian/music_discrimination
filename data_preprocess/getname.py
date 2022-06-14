from midi2audio import FluidSynth
import os
import subprocess

music_real = os.listdir('test_pop')



for i in range(len(music_real)):
	print(i,"   ",music_real[i])