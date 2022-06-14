import os
import subprocess

music_real = os.listdir('test_pop')
# music_real.remove('.DS_Store')
# music_fake.remove('.DS_Store')
for i in range(len(music_real)):
    path_music = 'test_pop/' + music_real[i]
    path_save = 'test_pop/' + 'real_' + music_real[i]
    os.rename(path_music,path_save)