from distutils.core import setup, Extension

import numpy as np

_qpSWIFT = Extension("qpSWIFT",
                     sources=[
                         "pyqpSWIFT.c",
                         "../src/amd_1.c",
                         "../src/amd_2.c",
                         "../src/amd_aat.c",
                         "../src/amd_control.c",
                         "../src/amd_defaults.c",
                         "../src/amd_dump.c",
                         "../src/amd_global.c",
                         "../src/amd_info.c",
                         "../src/amd_order.c",
                         "../src/amd_post_tree.c",
                         "../src/amd_postorder.c",
                         "../src/amd_preprocess.c",
                         "../src/amd_valid.c",
                         "../src/ldl.c",
                         "../src/timer.c",
                         "../src/Auxilary.c",
                         "../src/qpSWIFT.c"
                     ],
                     include_dirs=["../include/",
                                   np.get_include(),
                                   ],
                     #    extra_compile_args=["-O3"
                     #    ]
                     )


def main():
    setup(
        name="giskard_qpswift",
        version="1.0.0",
        description="Python interface for qpSWIFT",
        long_description='qpSWIFT is light-weight sparse Quadratic Programming solver targetted for embedded and robotic applications. It employs Primal-Dual Interioir Point method with Mehrotra Predictor corrector step and Nesterov Todd scaling. For solving the linear system of equations, sparse LDL\' factorization is used along with approximate minimum degree heuristic to minimize fill-in of the factorizations. This is a fork of the official qpSWIFT project (https://github.com/qpSWIFT/qpSWIFT)',
        url='https://github.com/SemRoCo/qpSWIFT',
        author="Simon Stelter",
        setup_requires=["numpy >= 1.6"],
        install_requires=["numpy >= 1.6"],
        ext_modules=[_qpSWIFT],
        license='GPLv3',
        license_files=('LICENSE',),
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
            'Operating System :: POSIX :: Linux',
            'Intended Audience :: Science/Research',
            "Programming Language :: Python",
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.12',
        ],
    )


if __name__ == "__main__":
    main()
