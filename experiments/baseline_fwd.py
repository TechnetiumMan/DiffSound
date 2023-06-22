import sys
sys.path.append('./')
from src.diff_model import DiffSoundObj, MatSet

if __name__ == '__main__':
    mesh_dir = 'data/mesh_data/full/55/'
    audio_dir = 'data/audio_data/55/audio'
    # mesh_dir = '/data/xcx/mesh_data/full/1/'
    # audio_dir = '/data/xcx/audio_data/1/audio'
    obj = DiffSoundObj(mesh_dir, audio_dir)
    print(obj.forward().shape)
