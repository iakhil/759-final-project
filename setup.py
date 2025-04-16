from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os
import sysconfig

# Get Python include path
python_include = sysconfig.get_paths()['include']

setup(
    name='matmul_cuda',
    ext_modules=[
        CUDAExtension(
            'matmul_cuda',
            sources=['matmul_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-ccbin=/usr/bin/gcc-11',  # or the actual path from `which gcc` after loading module
                    f'-I{python_include}',
                    '--compiler-options', "'-fPIC'",
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
