import os
import subprocess

music_real = os.listdir('fake_midi')
# # music_fake.remove('.DS_Store')
for i in range(len(music_real)):
    path_music = 'fake_midi/' + music_real[i]
    path_save = 'fake_midi/' + 'fake_' + music_real[i]
    os.rename(path_music,path_save)