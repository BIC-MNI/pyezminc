#  Copyright 2013, Haz-Edine Assemlal

#  This file is part of PYEZMINC.
# 
#  PYEZMINC is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, version 2.
# 
#  PYEZMINC is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with PYEZMINC.  If not, see <http://www.gnu.org/licenses/>.


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy
import os

# hack to allow user to specify location of minc installation
import sys

# default location of minc library
# find location of minc-toolkit
MINCDIR=os.environ.get('MINC_TOOLKIT',"/opt/minc/1.9.15")
source_path=os.path.join(os.path.dirname(__file__), "../../src")

MINCLIBS= ['minc2','z','m', 'minc_io'] 

# A hack to allow user to specify location of minc
# unfortunately, making it proper by extending build and build_ext and install 
# commands is not documented
# 
if '--mincdir' in sys.argv:
    index = sys.argv.index('--mincdir')
    sys.argv.pop(index)  # Removes the '--mincdir'
    MINCDIR = sys.argv.pop(index)  # Returns the element after the '--mincdir'



ext_modules=[ Extension(
                    "pyezminc",                                              # name of extension
                    ["pyezminc.pyx", 'pyezminc.pxd','minc_1_iterators.cpp'], # our Cython source
                    libraries=MINCLIBS,
                    include_dirs = [os.path.join(MINCDIR,'include'),
                                    numpy.get_include()],
                    library_dirs = [os.path.join(MINCDIR,'lib')],
                    runtime_library_dirs = [os.path.join(MINCDIR,'lib')],    # RPATH settings
                    #extra_objects = [os.path.join(MINCDIR,'libminc_io.a')], # Use this if using static link
                    language="c++")]  # causes Cython to create C++ source

setup(
    name = 'pyezminc',
    version = '1.2',
    url = 'https://github.com/BIC-MNI/pyezminc',
    author = 'Haz-Edine Assemlal',
    author_email = 'haz-edine@assemlal.com',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: PyPy",
        "License :: OSI Approved :: BSD License",
    ],
    cmdclass={'build_ext': build_ext },
    py_modules = ['minc'],
    ext_modules = ext_modules,
    license = 'GNU GPL v2'
)

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
