from midi2audio import FluidSynth
import os
import subprocess

music_fake = os.listdir('fake_midi')
# music_real.remove('.DS_Store')
# music_fake.remove('.DS_Store')
for i in range(len(music_fake)):
    path_music = 'fake_midi/' + music_fake[i]
    path_save = 'fake_wav/' + str(i) + '.wav'
    FluidSynth('default_sound_font.sf2').midi_to_audio(path_music, path_save)
    if i == 599:
    	break