import sys
sys.path.append('./')
from src.diff_model import DiffSoundObj, MatSet

if __name__ == '__main__':
    mesh_dir = 'data/mesh_data/full/1/'
    audio_dir = 'data/audio_data/1/audio'
    obj = DiffSoundObj(mesh_dir, audio_dir)
    target = obj.forward()
    mode_num = target.shape[1]
    obj.backward(target, mode_num)
