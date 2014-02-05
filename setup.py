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

import numpy
import os

HOME = os.path.expanduser('~')

ext_modules=[Extension(
                   "pyezminc",                 # name of extension
                   ["pyezminc.pyx", 'pyezminc.pxd'], #  our Cython source
                   libraries=['volume_io2', 'minc2','z','m', 'minc_io'],
                   #extra_objects = ['/hydra/home/hassemlal/local/lib/libminc_io.a'],
                   include_dirs = [os.path.join(HOME, 'local','include'),
                                   '/opt/minc/include',
                                   numpy.get_include()],
                   library_dirs = [os.path.join(HOME, 'local', 'lib'),
                                   '/opt/minc/lib'],
                   runtime_library_dirs = ['/opt/minc/lib'],
                   language="c++")]  # causes Cython to create C++ source

setup(
      name = 'pyezminc',
      cmdclass={'build_ext': build_ext},
      ext_modules = ext_modules
)
