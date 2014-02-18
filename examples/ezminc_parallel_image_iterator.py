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

    inputs=20
    outputs=2

    inp.open([os.path.join(DATA_PATH,   'mni_icbm152_t1_tal_nlin_sym_09c.mnc') for i in xrange(inputs)],
             os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c_mask.mnc'))

    out.open(["/tmp/test_{}.mnc".format(str(i)) for i in xrange(outputs)],os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c_mask.mnc'))
    inp.begin()
    out.begin()
    
    # allocate sum
    s=np.zeros(shape=[inputs],dtype=np.float64,order='C') 
    # allocate work space
    qqq=np.empty_like(s)
    
    try:
        while True:
            qqq=inp.value(qqq)
            res=qqq[0:outputs]
            
            out.value(res)

            s=s+qqq

            inp.next()
            out.next()

    except StopIteration:
        pass

    del inp
    del out
    print "parallel input sum={}".format(str(s))
    
    # removing output files
    for i in xrange(outputs):
        os.unlink("/tmp/test_{}.mnc".format(str(i)))

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
