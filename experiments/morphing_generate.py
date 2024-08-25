# generate meshes with different morphing coeffecient between two initial meshes
# the morphing coeffecient determines the morphing mesh is closer to which initial mesh
# we just interpolate the sdf of two initial meshes with the morphing coeffecient, at every vertices in DMTet background mesh

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"                             
import argparse
import json
from tqdm import tqdm
import sys
sys.path.append("src/dmtet/")
from geometry.dmtet_interpolate import DMTetGeometry
from render import obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file') 
    FLAGS = parser.parse_args()
    FLAGS.without_tensorboard = True

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    print("Config / Flags:")
    print("---------")
    for key in FLAGS.__dict__.keys():
        print(key, FLAGS.__dict__[key])
    print("---------")

    os.makedirs(FLAGS.out_mesh_dir, exist_ok=True)

    # generate target mesh with specific morphing coef
    for interp_coef in FLAGS.morphing_list:
        target_geometry = DMTetGeometry(128, FLAGS.mesh_scale, FLAGS)
        target_geometry.apply_sdf2(FLAGS.init_mesh_dir + FLAGS.mesh_name1 + ".obj", 
                                   FLAGS.init_mesh_dir + FLAGS.mesh_name2 + ".obj")
        target_triangle_mesh = target_geometry.getMesh(return_triangle=True, interp_coef=interp_coef)
        
        os.makedirs(os.path.join(FLAGS.out_mesh_dir, f"{FLAGS.mesh_name1}_{FLAGS.mesh_name2}"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_mesh_dir, f"{FLAGS.mesh_name1}_{FLAGS.mesh_name2}"), target_triangle_mesh, name=f"morphing{interp_coef}.obj")
