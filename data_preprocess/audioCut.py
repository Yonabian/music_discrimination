from pydub import AudioSegment
import wave
import os

real_dir = os.listdir("real_wav/");
fake_dir = os.listdir("fake_wav/");
for w in real_dir:
	path = "real_wav/" + w
	f = wave.open(path)
	rate = f.getframerate()
	frames = f.getnframes()
	duration = frames/float(rate)
	if duration<60:
		continue
	else:
		sound = AudioSegment.from_file(path,format='.wav')
		cut_wav = sound[0:60000]
		cut_wav.export(path,format='wav')

for w in fake_dir:
	path = "fake_wav/" + w
	f = wave.open(path)
	rate = f.getframerate()
	frames = f.getnframes()
	duration = frames/float(rate)
	if duration<60:
		continue
	else:
		sound = AudioSegment.from_file(path,format='.wav')
		cut_wav = sound[0:60000]
		cut_wav.export(path,format='wav')