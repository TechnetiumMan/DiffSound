#pragma once
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/script.h>
#include "array3D.h"
#include "macro.h"
#include <iostream>
#include <memory>
using namespace ModalSound;

__global__ void compute_mass_matrix_kernel(GArr<float> values, GArr<int> rows, GArr<int> cols, GArr<float> vertices, GArr<int> tets, float d, GArr<float> m, int order);
float get_shape_func_value(int order, int i, float x, float y, float z, float w);
void compute_elememt_mass_matrix(CArr<float>& m, int order, int vnum);
void assemble_mass_matrix(const torch::Tensor &vertices_, const torch::Tensor &tets_,
                          torch::Tensor &values_, torch::Tensor &rows_, torch::Tensor &cols_, torch::Tensor &element_mm, const float density, const int order);


