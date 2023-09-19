import numpy as np
import torch
from torch.utils.cpp_extension import load as load_cuda
import os
from glob import glob

class CUDA_MODULE:
    _module = None

    @staticmethod
    def get(name):
        if CUDA_MODULE._module is None:
            CUDA_MODULE.load()
        return getattr(CUDA_MODULE._module, name)

    @staticmethod
    def load(Debug=False, MemoryCheck=False, Verbose=False):
        src_dir = os.path.dirname(os.path.abspath(
            __file__)) + "/cuda"
        os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(src_dir, 'build')
        cflags = ''
        if Debug:
            cflags += '-G -g'
            cflags += ' -DDEBUG'
        else:
            cflags += '-O3'
            cflags += ' -DNDEBUG'
        if MemoryCheck:
            cflags += ' -DMEMORY_CHECK'
        cuda_files = [os.path.join(src_dir, 'massMatrixDouble.cu'), os.path.join(src_dir, 'bind.cu')]
        
        include_paths = [src_dir, src_dir.replace('cuda', 'include')] 
        CUDA_MODULE._module = load_cuda(name='diffFEM',
                                        sources=cuda_files,
                                        extra_include_paths=include_paths,
                                        extra_cuda_cflags=[cflags], 
                                        verbose=Verbose)
        return CUDA_MODULE._module

CUDA_MODULE.load(Debug=True, MemoryCheck=True, Verbose=True)
mass_matrix_assembler = CUDA_MODULE.get('assemble_mass_matrix')
