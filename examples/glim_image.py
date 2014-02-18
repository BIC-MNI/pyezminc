#! /usr/bin/env python

import minc
import sys
import os
import pyezminc
import numpy as np
import csv

import rpy2.robjects as ro

from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr


def load_csv(csv_file):
    '''Load csv file into a dictionary'''
    data={}
    # load CSV file 
    with open(input_csv,'r') as f:
        for r in csv.DictReader(f):
            for k in r.iterkeys():
                try:
                    data[k].append(r[k])
                except KeyError:
                    data[k]=[r[k]]
    return data


if __name__ == "__main__":
    
    # setup automatic conversion for numpy
    ro.conversion.py2ri = numpy2ri
    # import R objects
    stats = importr('stats')
    base = importr('base')

    # read the input data
    input_csv='brain.csv'
    mask_file='mask.mnc'
    
    # load CSV file
    data=load_csv(input_csv)
    # data['jacobian'] now contains file names
    # and data['cohort'] - group
    
    # setup R objects for performing linear modelling
    cohort = ro.FactorVector(data['cohort'])
    
    inp=pyezminc.parallel_input_iterator()
    out=pyezminc.parallel_output_iterator()

    inp.open(data['jacobian'],mask_file )
    out.open(["output_slope.mnc","output_std_error.mnc","output_t_stat.mnc","output_p_val.mnc"],mask_file)

    # allocate space for input
    jacobian=np.zeros(shape=[inp.dim()],dtype=np.float64,order='C')

    # allocate space for output
    result=np.zeros(shape=[out.dim()],dtype=np.float64,order='C')

    # allocate R formula, saves time for interpreter
    fmla = ro.Formula('jacobian ~ cohort')

    # assign cohort variable - it stays the same 
    fmla.environment["cohort"] = cohort

    # start iteration, not really needed
    inp.begin()
    out.begin()

    l=None
    try:
        while True:

            if inp.value_mask():
                jacobian=inp.value(jacobian)

                # update jacobian vairable
                fmla.environment["jacobian"] = jacobian
                
                # run linear model
                l=stats.lm(fmla)
                
                # extract coeffecients
                s = base.summary(l).rx2('coefficients')

                result[0]=s.rx(2,1)[0] # this is estimate
                result[1]=s.rx(2,2)[0] # this is standard error
                result[2]=s.rx(2,3)[0] # this is t-value
                result[3]=s.rx(2,4)[0] # this is p-value
            else:
                # we are passing-by voxels outside of the mask,
                # assign default value
                result[0]=result[1]=result[2]=0
                result[3]=1.0

            # set the value
            out.value(result)
            
            # move to the next voxel
            inp.next()
            out.next()

    except StopIteration:
        pass

    # free up memory, not really needed 
    del inp
    del out


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
