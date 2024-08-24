# generate meshes with different thicknesses for thickness inference experiment
# initial mesh is the solid mesh, then we transfer it to a hollow mesh with specifit thickness using DMTetGeometry

# parameters in generate_thickness.json:
# init_mesh_dir: initial solid mesh directory
# mesh_name: mesh filename before ".obj"
# out_dir: output directory
# thickness_list: list of thicknesses to generate
# mesh_scale: used for scale the mesh (better to be a little larger to the edge length of a smallest cube containing the mesh)

import os                         
import argparse
import json
import sys
sys.path.append("src/dmtet/")

# Import topology / geometry trainers
from geometry.dmtet_thickness import DMTetGeometry
from render import obj

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

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

    # generate target mesh with thickness=thickness_coef
    for thickness_coef in FLAGS.thickness_list:
        target_geometry = DMTetGeometry(128, FLAGS.mesh_scale, FLAGS)
        target_geometry.apply_sdf(FLAGS.init_mesh_dir + FLAGS.mesh_name + ".obj", FLAGS)
        target_triangle_mesh = target_geometry.getMesh(return_triangle=True, thickness_coef=thickness_coef)
        
        os.makedirs(os.path.join(FLAGS.out_mesh_dir, f"{FLAGS.mesh_name}"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_mesh_dir, f"{FLAGS.mesh_name}"), target_triangle_mesh, name=f"thickness{thickness_coef}.obj")

#----------------------------------------------------------------------------
