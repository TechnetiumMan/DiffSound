import glob
import os

file_pattern = "real_data/audio/*/audio/*/cam/*.mp4"
files = glob.glob(file_pattern)
for file in files:
    os.remove(file)