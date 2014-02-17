#! /usr/bin/env python


import minc
import sys
import os
import pyezminc


DATA_PATH = '/opt/minc/share/icbm152_model_09c'


if __name__ == "__main__":
    
    sum1=0.0
    for i in pyezminc.input_iterator_real(os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c.mnc')):
        sum1+=i
    print "sum of all voxels in T1w template={}".format(sum1)

    sum2=0.0
    for i in pyezminc.input_iterator_int(os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c_mask.mnc')):
        sum2+=i
    print "sum of all voxels in T1w template mask={}".format(sum2)
    
    print("Copying contents...")

    # copy file contents
    i=pyezminc.input_iterator_real(os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c.mnc'))
    o=pyezminc.output_iterator_real("/tmp/test.mnc",os.path.join(DATA_PATH, 'mni_icbm152_t1_tal_nlin_sym_09c.mnc'))

    try:
        while True:
            o.value(i.value())
            i.next()
            o.next()
    except StopIteration:
        pass
    del i
    del o
    
    sum3=0.0
    for i in pyezminc.input_iterator_real("/tmp/test.mnc"):
        sum3+=i
    print "sum of all voxels in copy of T1w template={}".format(sum3)
    os.unlink("/tmp/test.mnc")
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
