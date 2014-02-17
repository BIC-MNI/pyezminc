#! /usr/bin/env python


import minc
import sys
import os
import pyezminc
import numpy as np

#DATA_PATH = '/opt/minc/share/icbm152_model_09c'
DATA_PATH = '/extra/mni'

if __name__ == "__main__":
    
    inp=pyezminc.parallel_input_iterator()
    out=pyezminc.parallel_output_iterator()
    
    inp.open([os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c.mnc'),
                os.path.join(DATA_PATH, 'mni_icbm152_t2_tal_nlin_sym_09c.mnc')],
                os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c_mask.mnc'))

    out.open(["/tmp/test1.mnc","/tmp/test2.mnc"],os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c_mask.mnc'))
    inp.begin()

    #out.begin()
    s=np.zeros(shape=[2],dtype=np.float64,order='C')
    qqq=np.empty_like(s)
    try:
        while True:
            #out.value(inp.value())
            qqq=inp.value(qqq)
            #out.value(qqq)

            s=s+qqq
            inp.next()
            out.next()
    except StopIteration:
        pass

    del inp
    del out
    print "parallel input sum={}".format(str(s))

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
