#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
// #include "massMatrix.h"
#include "massMatrixDouble.h"

// using namespace diffRender;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("assemble_mass_matrix", &assemble_mass_matrix, "");
}
