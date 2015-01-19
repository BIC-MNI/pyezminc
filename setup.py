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

#HOME = os.path.expanduser('~')

# TODO: locate minc library
MINCDIR = '/projects/souris/vfonov/1.9.07'

# TODO: determine if volume_io is still availabel as separate library
MINCLIBS= ['minc2','z','m', 'minc_io'] 


ext_modules=[Extension(
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
    version = '1.0',
    url = 'https://github.com/BIC-MNI/pyezminc',
    author = 'Haz-Edine Assemlal',
    author_email = 'haz-edine@assemlal.com',
    cmdclass={'build_ext': build_ext},
    py_modules = ['minc'],
    ext_modules = ext_modules,
    license = 'GNU GPL v2'
)

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
