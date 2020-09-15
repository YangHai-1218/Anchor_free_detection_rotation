from setuptools import setup,find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch
import os

def make_cuda_ext(name, module, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)




if __name__ == '__main__':
    setup(
        name='Anchor_free_detection_rotation',
        version= '0.0.1',
        description='Anchor free detection for rotation bbox',
        ext_modules=[
            make_cuda_ext(
                name='rnms_ext',
                module='ops.nms',
                sources=['src/rnms_ext.cpp', 'src/rcpu/rnms_cpu.cpp'],
                sources_cuda=[
                    'src/rcuda/rnms_cuda.cpp', 'src/rcuda/rnms_kernel.cu'
                ]),
            make_cuda_ext(
                name='polygon_geo_cpu',
                module='ops.polygon_geo',
                sources=['src/polygon_geo_cpu.cpp']),
            make_cuda_ext(
                name='nms_ext',
                module='ops.nms',
                sources=['src/nms_ext.cpp', 'src/cpu/nms_cpu.cpp'],
                sources_cuda=[
                    'src/cuda/nms_cuda.cpp', 'src/cuda/nms_kernel.cu'
                ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)