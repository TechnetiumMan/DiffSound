#include <torch/extension.h>
#include "array3D.h"

using namespace ModalSound;

__global__ void compute_mass_matrix_kernel(GArr<float> values,
                                           GArr<int> rows,
                                           GArr<int> cols,
                                           GArr<float> vertices,
                                           GArr<int> tets,
                                           float d)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tets.size() / 4)
        return;
    int *tets_ptr = tets.begin() + idx * 4;
    float x[4], y[4], z[4];
    for (int i = 0; i < 4; i++)
    {
        x[i] = vertices[tets_ptr[i] * 3];
        y[i] = vertices[tets_ptr[i] * 3 + 1];
        z[i] = vertices[tets_ptr[i] * 3 + 2];
    }
    int vid[12] = {
        tets_ptr[0] * 3,     tets_ptr[0] * 3 + 1, tets_ptr[0] * 3 + 2, tets_ptr[1] * 3,
        tets_ptr[1] * 3 + 1, tets_ptr[1] * 3 + 2, tets_ptr[2] * 3,     tets_ptr[2] * 3 + 1,
        tets_ptr[2] * 3 + 2, tets_ptr[3] * 3,     tets_ptr[3] * 3 + 1, tets_ptr[3] * 3 + 2,
    };
    float V = ((x[1] - x[0]) * ((y[2] - y[0]) * (z[3] - z[0]) - (y[3] - y[0]) * (z[2] - z[0])) +
               (y[1] - y[0]) * ((x[3] - x[0]) * (z[2] - z[0]) - (x[2] - x[0]) * (z[3] - z[0])) +
               (z[1] - z[0]) * ((x[2] - x[0]) * (y[3] - y[0]) - (x[3] - x[0]) * (y[2] - y[0]))) /
              6;
    V = abs(V);
    float m[] = {2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0, 0,
                 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0,
                 1, 0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0,
                 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0,
                 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 2};

    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
            m[i * 12 + j] *= (d / 20) * V;

    int offset = idx * 12 * 12;
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
        {
            values[offset + i * 12 + j] = m[i * 12 + j];
            rows[offset + i * 12 + j] = vid[i];
            cols[offset + i * 12 + j] = vid[j];
        }
}

__global__ void compute_stiffness_matrix_kernel(GArr<float> values,
                                                GArr<int> rows,
                                                GArr<int> cols,
                                                GArr<float> vertices,
                                                GArr<int> tets,
                                                float E0,
                                                float v)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= tets.size() / 4)
        return;
    int *tets_ptr = tets.begin() + idx * 4;
    float x[4], y[4], z[4];
    for (int i = 0; i < 4; i++)
    {
        x[i] = vertices[tets_ptr[i] * 3];
        y[i] = vertices[tets_ptr[i] * 3 + 1];
        z[i] = vertices[tets_ptr[i] * 3 + 2];
    }
    int vid[12] = {
        tets_ptr[0] * 3,     tets_ptr[0] * 3 + 1, tets_ptr[0] * 3 + 2, tets_ptr[1] * 3,
        tets_ptr[1] * 3 + 1, tets_ptr[1] * 3 + 2, tets_ptr[2] * 3,     tets_ptr[2] * 3 + 1,
        tets_ptr[2] * 3 + 2, tets_ptr[3] * 3,     tets_ptr[3] * 3 + 1, tets_ptr[3] * 3 + 2,
    };
    float a[4], b[4], c[4], V;
    a[0] = y[1] * (z[3] - z[2]) - y[2] * (z[3] - z[1]) + y[3] * (z[2] - z[1]);
    a[1] = -y[0] * (z[3] - z[2]) + y[2] * (z[3] - z[0]) - y[3] * (z[2] - z[0]);
    a[2] = y[0] * (z[3] - z[1]) - y[1] * (z[3] - z[0]) + y[3] * (z[1] - z[0]);
    a[3] = -y[0] * (z[2] - z[1]) + y[1] * (z[2] - z[0]) - y[2] * (z[1] - z[0]);
    b[0] = -x[1] * (z[3] - z[2]) + x[2] * (z[3] - z[1]) - x[3] * (z[2] - z[1]);
    b[1] = x[0] * (z[3] - z[2]) - x[2] * (z[3] - z[0]) + x[3] * (z[2] - z[0]);
    b[2] = -x[0] * (z[3] - z[1]) + x[1] * (z[3] - z[0]) - x[3] * (z[1] - z[0]);
    b[3] = x[0] * (z[2] - z[1]) - x[1] * (z[2] - z[0]) + x[2] * (z[1] - z[0]);
    c[0] = x[1] * (y[3] - y[2]) - x[2] * (y[3] - y[1]) + x[3] * (y[2] - y[1]);
    c[1] = -x[0] * (y[3] - y[2]) + x[2] * (y[3] - y[0]) - x[3] * (y[2] - y[0]);
    c[2] = x[0] * (y[3] - y[1]) - x[1] * (y[3] - y[0]) + x[3] * (y[1] - y[0]);
    c[3] = -x[0] * (y[2] - y[1]) + x[1] * (y[2] - y[0]) - x[2] * (y[1] - y[0]);
    V = ((x[1] - x[0]) * ((y[2] - y[0]) * (z[3] - z[0]) - (y[3] - y[0]) * (z[2] - z[0])) +
         (y[1] - y[0]) * ((x[3] - x[0]) * (z[2] - z[0]) - (x[2] - x[0]) * (z[3] - z[0])) +
         (z[1] - z[0]) * ((x[2] - x[0]) * (y[3] - y[0]) - (x[3] - x[0]) * (y[2] - y[0]))) /
        6;
    V = abs(V);
    float e[] = {1 - v, v, v, 0,        0, 0, v, 1 - v, v, 0, 0,        0, v, v, 1 - v, 0, 0, 0,
                 0,     0, 0, 0.5f - v, 0, 0, 0, 0,     0, 0, 0.5f - v, 0, 0, 0, 0,     0, 0, 0.5f - v};
    for (int i = 0; i < 6 * 6; i++)
        e[i] = e[i] * (E0 / (1 + v) / (1 - 2 * v));

    float be[] = {a[0], 0,    0,    a[1], 0,    0,    a[2], 0,    0,    a[3], 0,    0,    0,    b[0], 0,
                  0,    b[1], 0,    0,    b[2], 0,    0,    b[3], 0,    0,    0,    c[0], 0,    0,    c[1],
                  0,    0,    c[2], 0,    0,    c[3], b[0], a[0], 0,    b[1], a[1], 0,    b[2], a[2], 0,
                  b[3], a[3], 0,    0,    c[0], b[0], 0,    c[1], b[1], 0,    c[2], b[2], 0,    c[3], b[3],
                  c[0], 0,    a[0], c[1], 0,    a[1], c[2], 0,    a[2], c[3], 0,    a[3]};
    for (int i = 0; i < 6 * 12; i++)
        be[i] = be[i] / (6 * V);

    float ke1[12 * 6];
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 6; j++)
        {
            ke1[i * 6 + j] = 0;
            for (int k = 0; k < 6; k++)
                ke1[i * 6 + j] += be[k * 12 + i] * e[k * 6 + j];
        }
    float ke2[12 * 12];

    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
        {
            ke2[i * 12 + j] = 0;
            for (int k = 0; k < 6; k++)
                ke2[i * 12 + j] += ke1[i * 6 + k] * be[k * 12 + j];
            ke2[i * 12 + j] *= V;
        }

    int offset = idx * 12 * 12;
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
        {
            values[offset + i * 12 + j] = ke2[i * 12 + j];
            rows[offset + i * 12 + j] = vid[i];
            cols[offset + i * 12 + j] = vid[j];
        }
}

__global__ void compute_stiffness_B_D_matrix_kernel(GArr2D<float> B,
                                                    GArr2D<float> D,
                                                    GArr<float> vertices,
                                                    GArr<int> tets,
                                                    float E0,
                                                    float v)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= tets.size() / 4)
        return;
    int *tets_ptr = tets.begin() + idx * 4;
    float x[4], y[4], z[4];
    for (int i = 0; i < 4; i++)
    {
        x[i] = vertices[tets_ptr[i] * 3];
        y[i] = vertices[tets_ptr[i] * 3 + 1];
        z[i] = vertices[tets_ptr[i] * 3 + 2];
    }
    float a[4], b[4], c[4], V;
    a[0] = y[1] * (z[3] - z[2]) - y[2] * (z[3] - z[1]) + y[3] * (z[2] - z[1]);
    a[1] = -y[0] * (z[3] - z[2]) + y[2] * (z[3] - z[0]) - y[3] * (z[2] - z[0]);
    a[2] = y[0] * (z[3] - z[1]) - y[1] * (z[3] - z[0]) + y[3] * (z[1] - z[0]);
    a[3] = -y[0] * (z[2] - z[1]) + y[1] * (z[2] - z[0]) - y[2] * (z[1] - z[0]);
    b[0] = -x[1] * (z[3] - z[2]) + x[2] * (z[3] - z[1]) - x[3] * (z[2] - z[1]);
    b[1] = x[0] * (z[3] - z[2]) - x[2] * (z[3] - z[0]) + x[3] * (z[2] - z[0]);
    b[2] = -x[0] * (z[3] - z[1]) + x[1] * (z[3] - z[0]) - x[3] * (z[1] - z[0]);
    b[3] = x[0] * (z[2] - z[1]) - x[1] * (z[2] - z[0]) + x[2] * (z[1] - z[0]);
    c[0] = x[1] * (y[3] - y[2]) - x[2] * (y[3] - y[1]) + x[3] * (y[2] - y[1]);
    c[1] = -x[0] * (y[3] - y[2]) + x[2] * (y[3] - y[0]) - x[3] * (y[2] - y[0]);
    c[2] = x[0] * (y[3] - y[1]) - x[1] * (y[3] - y[0]) + x[3] * (y[1] - y[0]);
    c[3] = -x[0] * (y[2] - y[1]) + x[1] * (y[2] - y[0]) - x[2] * (y[1] - y[0]);
    V = ((x[1] - x[0]) * ((y[2] - y[0]) * (z[3] - z[0]) - (y[3] - y[0]) * (z[2] - z[0])) +
         (y[1] - y[0]) * ((x[3] - x[0]) * (z[2] - z[0]) - (x[2] - x[0]) * (z[3] - z[0])) +
         (z[1] - z[0]) * ((x[2] - x[0]) * (y[3] - y[0]) - (x[3] - x[0]) * (y[2] - y[0]))) /
        6;
    V = abs(V);
    float e[] = {1 - v, v, v, 0,        0, 0, v, 1 - v, v, 0, 0,        0, v, v, 1 - v, 0, 0, 0,
                 0,     0, 0, 0.5f - v, 0, 0, 0, 0,     0, 0, 0.5f - v, 0, 0, 0, 0,     0, 0, 0.5f - v};
    for (int i = 0; i < 6 * 6; i++)
        D(idx, i) = e[i] * (E0 / (1 + v) / (1 - 2 * v));

    float be[] = {a[0], 0,    0,    a[1], 0,    0,    a[2], 0,    0,    a[3], 0,    0,    0,    b[0], 0,
                  0,    b[1], 0,    0,    b[2], 0,    0,    b[3], 0,    0,    0,    c[0], 0,    0,    c[1],
                  0,    0,    c[2], 0,    0,    c[3], b[0], a[0], 0,    b[1], a[1], 0,    b[2], a[2], 0,
                  b[3], a[3], 0,    0,    c[0], b[0], 0,    c[1], b[1], 0,    c[2], b[2], 0,    c[3], b[3],
                  c[0], 0,    a[0], c[1], 0,    a[1], c[2], 0,    a[2], c[3], 0,    a[3]};
    for (int i = 0; i < 6 * 12; i++)
        B(idx, i) = be[i] / (6 * V);
}

#define CUDA_BLOCK_NUM 64
void assemble_mass_matrix(const torch::Tensor &vertices_,
                          const torch::Tensor &tets_,
                          torch::Tensor &values_,
                          torch::Tensor &rows_,
                          torch::Tensor &cols_,
                          const float density)
{
    GArr<float> vertices((float *)vertices_.data_ptr(), vertices_.size(0));
    GArr<int> tets((int *)tets_.data_ptr(), tets_.size(0));
    GArr<float> values((float *)values_.data_ptr(), values_.size(0));
    GArr<int> rows((int *)rows_.data_ptr(), rows_.size(0));
    GArr<int> cols((int *)cols_.data_ptr(), cols_.size(0));
    int tet_num = tets.size() / 4;
    cuExecute(tet_num, compute_mass_matrix_kernel, values, rows, cols, vertices, tets, density);
}

void assemble_stiffness_matrix(const torch::Tensor &vertices_,
                               const torch::Tensor &tets_,
                               torch::Tensor &values_,
                               torch::Tensor &rows_,
                               torch::Tensor &cols_,
                               const float E,
                               const float nu)
{
    GArr<float> vertices((float *)vertices_.data_ptr(), vertices_.size(0));
    GArr<int> tets((int *)tets_.data_ptr(), tets_.size(0));
    GArr<float> values((float *)values_.data_ptr(), values_.size(0));
    GArr<int> rows((int *)rows_.data_ptr(), rows_.size(0));
    GArr<int> cols((int *)cols_.data_ptr(), cols_.size(0));
    int tet_num = tets.size() / 4;
    cuExecute(tet_num, compute_stiffness_matrix_kernel, values, rows, cols, vertices, tets, E, nu);
}

void stiffness_B_D_matrix(const torch::Tensor &vertices_,
                          const torch::Tensor &tets_,
                          torch::Tensor &B_,
                          torch::Tensor &D_,
                          const float E,
                          const float nu)
{
    GArr<float> vertices((float *)vertices_.data_ptr(), vertices_.size(0));
    GArr<int> tets((int *)tets_.data_ptr(), tets_.size(0));
    GArr2D<float> B((float *)B_.data_ptr(), B_.size(0), B_.size(1));
    GArr2D<float> D((float *)D_.data_ptr(), D_.size(0), D_.size(1));
    int tet_num = tets.size() / 4;
    cuExecute(tet_num, compute_stiffness_B_D_matrix_kernel, B, D, vertices, tets, E, nu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("assemble_mass_matrix", &assemble_mass_matrix, "Assemble mass matrix for tet mesh");
    m.def("assemble_stiffness_matrix", &assemble_stiffness_matrix, "Assemble stiffness matrix for tet mesh");
    m.def("stiffness_B_D_matrix", &stiffness_B_D_matrix, "Compute B matrix and D matrix within stiffness for tet mesh");
}
