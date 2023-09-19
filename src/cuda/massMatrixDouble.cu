#include "massMatrixDouble.h"

__global__ void compute_mass_matrix_kernel(GArr<double> values, GArr<int> rows, GArr<int> cols, GArr<double> vertices, GArr<int> tets, double d, GArr<double> m, int order)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vnum; // vertice num of this order
    if(order == 1)  vnum = 4;
    else if(order == 2) vnum = 10;
    else if(order == 3) vnum = 20;
    if (idx >= tets.size() / vnum)
        return;

    int msize = vnum * 3;
    assert(m.size() == msize * msize);

    int* tets_ptr = tets.begin() + idx * vnum;

    double x[4], y[4], z[4];
    if(order == 1) 
    {
        for (int i = 0; i < vnum; i++)
        {
            x[i] = vertices[tets_ptr[i] * 3];
            y[i] = vertices[tets_ptr[i] * 3 + 1];
            z[i] = vertices[tets_ptr[i] * 3 + 2];
        }
    }
    else if(order == 2)
    {
        x[0] = vertices[tets_ptr[0] * 3];
        y[0] = vertices[tets_ptr[0] * 3 + 1];
        z[0] = vertices[tets_ptr[0] * 3 + 2];
        x[1] = vertices[tets_ptr[2] * 3];
        y[1] = vertices[tets_ptr[2] * 3 + 1];
        z[1] = vertices[tets_ptr[2] * 3 + 2];
        x[2] = vertices[tets_ptr[4] * 3];
        y[2] = vertices[tets_ptr[4] * 3 + 1];
        z[2] = vertices[tets_ptr[4] * 3 + 2];
        x[3] = vertices[tets_ptr[9] * 3];
        y[3] = vertices[tets_ptr[9] * 3 + 1];
        z[3] = vertices[tets_ptr[9] * 3 + 2];
    }
    else if(order == 3)
    {
        x[0] = vertices[tets_ptr[0] * 3];
        y[0] = vertices[tets_ptr[0] * 3 + 1];
        z[0] = vertices[tets_ptr[0] * 3 + 2];
        x[1] = vertices[tets_ptr[3] * 3];
        y[1] = vertices[tets_ptr[3] * 3 + 1];
        z[1] = vertices[tets_ptr[3] * 3 + 2];
        x[2] = vertices[tets_ptr[6] * 3];
        y[2] = vertices[tets_ptr[6] * 3 + 1];
        z[2] = vertices[tets_ptr[6] * 3 + 2];
        x[3] = vertices[tets_ptr[16] * 3];
        y[3] = vertices[tets_ptr[16] * 3 + 1];
        z[3] = vertices[tets_ptr[16] * 3 + 2];
    }
    
    double V = ((x[1] - x[0]) * ((y[2] - y[0]) * (z[3] - z[0]) - (y[3] - y[0]) * (z[2] - z[0])) + (y[1] - y[0]) * ((x[3] - x[0]) * (z[2] - z[0]) - (x[2] - x[0]) * (z[3] - z[0])) + (z[1] - z[0]) * ((x[2] - x[0]) * (y[3] - y[0]) - (x[3] - x[0]) * (y[2] - y[0]))) / 6;
    V = abs(V) * 6;

    int vid[60];
    for (int i = 0; i < vnum; i++)
    {
        vid[i * 3] = tets_ptr[i] * 3;
        vid[i * 3 + 1] = tets_ptr[i] * 3 + 1;
        vid[i * 3 + 2] = tets_ptr[i] * 3 + 2;
    }

    int offset = idx * msize * msize;
    for (int i = 0; i < msize; i++)
        for (int j = 0; j < msize; j++)
        {
            values[offset + i * msize + j] = m[i * msize + j] * d * V;
            rows[offset + i * msize + j] = vid[i];
            cols[offset + i * msize + j] = vid[j];
        }
}

double get_shape_func_value(int order, int i, double x, double y, double z, double w)
{
    if(order == 1)
    {
        switch(i)
        {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            case 3: return w;
        }
    }
    else if(order == 2)
    {
        switch(i)
        {
            case 0: return x * (2 * x - 1);
            case 1: return 4 * x * y;
            case 2: return y * (2 * y - 1);
            case 3: return 4 * y * z;
            case 4: return z * (2 * z - 1);
            case 5: return 4 * z * x;
            case 6: return 4 * x * w;
            case 7: return 4 * y * w;
            case 8: return 4 * z * w;
            case 9: return w * (2 * w - 1);
        }
    }
    else if(order == 3)
    {
        switch(i)
        {
            case 0: return 1/2*(3*x-1)*(3*x-2)*x;
            case 1: return 9/2*x*y*(3*x-1);
            case 2: return 9/2*x*y*(3*y-1);
            case 3: return 1/2*(3*y-1)*(3*y-2)*y;
            case 4: return 9/2*y*z*(3*y-1);
            case 5: return 9/2*y*z*(3*z-1);
            case 6: return 1/2*(3*z-1)*(3*z-2)*z;
            case 7: return 9/2*z*x*(3*z-1);
            case 8: return 9/2*z*x*(3*x-1);
            case 9: return 27*x*y*z;
            case 10: return 9/2*x*w*(3*x-1);
            case 11: return 9/2*y*w*(3*y-1);
            case 12: return 9/2*z*w*(3*z-1);
            case 13: return 9/2*x*w*(3*w-1);
            case 14: return 9/2*y*w*(3*w-1);
            case 15: return 9/2*z*w*(3*w-1);
            case 16: return 1/2*(3*w-1)*(3*w-2)*w;
            case 17: return 27*y*z*w;
            case 18: return 27*x*z*w;
            case 19: return 27*x*y*w;
        }
    }
    return 0;
}

#define CUDA_BLOCK_NUM 64
void assemble_mass_matrix(const torch::Tensor &vertices_, const torch::Tensor &tets_,
                          torch::Tensor &values_, torch::Tensor &rows_, torch::Tensor &cols_, torch::Tensor &element_mm, const double density, const int order)
{

    GArr<double> vertices((double*)vertices_.data_ptr(), vertices_.size(0));
    GArr<int> tets((int*)tets_.data_ptr(), tets_.size(0));
    GArr<double> values((double*)values_.data_ptr(), values_.size(0));
    GArr<int> rows((int*)rows_.data_ptr(), rows_.size(0));
    GArr<int> cols((int*)cols_.data_ptr(), cols_.size(0));

    int vnum;
    if(order == 1) vnum = 4;
    else if(order == 2) vnum = 10;
    else if(order == 3) vnum = 20;

    GArr<double> m((double*)element_mm.data_ptr(), element_mm.size(0));

    int tet_num = tets.size() / vnum;
    cuExecute(tet_num, compute_mass_matrix_kernel,
              values, rows, cols, vertices, tets, density, m, order);
}
