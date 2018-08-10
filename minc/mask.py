#!/usr/bin/env python

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

from __future__ import division

import numpy as np
from . import Image


class Mask(Image):
    def __init__(self, fname=None, data=None, autoload=True, dtype=np.int32, mnc2np=True, *args, **kwargs):
        self.mnc2np = mnc2np
        super(Mask, self).__init__(fname=fname, data=data, autoload=autoload, dtype=dtype, *args, **kwargs)
        if mnc2np and data!=None and not fname:
            self.data = self._minc_to_numpy(self.data)
    
    def _minc_to_numpy(self, data):
        return np.logical_not(data.astype(np.bool)).astype(np.dtype)
    
    def _numpy_to_minc(self, data):
        return np.logical_not(data.astype(np.bool)).astype(self.dtype)
        
    def _load(self, name, *args, **kwargs):
        '''Load a file'''
        super(Mask, self)._load(name, *args, **kwargs)
        if self.mnc2np:
            self.data = self._minc_to_numpy(self.data)
        
    def _save(self, name=None, imitate=None, *args, **kwargs):
        old_data = self.data
        try:
            self.data = self._numpy_to_minc(self.data)
            super(Mask, self)._save(name=name, imitate=imitate, *args, **kwargs)
        finally:
            self.data = old_data

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
