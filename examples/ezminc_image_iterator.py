#! /usr/bin/env python


import minc
import sys
import os
import pyezminc


DATA_PATH = '/opt/minc/share/icbm152_model_09c'


if __name__ == "__main__":
    
    s=0.0
    for i in pyezminc.input_iterator_real(os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c.mnc')):
        s+=i
    print "sum of all voxels in T1w template={}".format(s)
    
    s=0.0
    for i in pyezminc.input_iterator_int(os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c_mask.mnc')):
        s+=i
    print "sum of all voxels in T1w template mask={}".format(s)
  
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
