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
from rpy2.rinterface import RRuntimeError

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
    base  = importr('base')
    nlme  = importr('nlme')

    # read the input data
    input_csv='longitudinal_roi.csv'
    mask_file='mask_roi.mnc'
    
    # load CSV file
    data=load_csv(input_csv)
    # columns:
    # Subject,Visit,Filename,Age,Gender,Scale
    # 
    
    # setup R objects for performing linear modelling
    Subject = ro.FactorVector(data['Subject'])
    Visit   = ro.FactorVector(data['Visit'])
    Age     = ro.FloatVector(data['Age'])
    Gender  = ro.FactorVector(data['Gender'])
    Scale   = ro.FloatVector(data['Scale'])
    
    inp=pyezminc.parallel_input_iterator()
    out=pyezminc.parallel_output_iterator()

    
    # allocate R formula, saves time for interpreter
    fixed_effects = ro.Formula('Jacobian ~ I(Age^2) + Gender:I(Age^2) + Age + Gender:Age + Gender')
    random_effects = ro.Formula('~1|Subject')
    
    inp.open(data['Filename'],mask_file )
    out.open(["output_roi_Intercept.mnc","output_roi_Age2.mnc","output_roi_Gender_Age2.mnc","output_roi_Age.mnc","output_roi_Gender_Age.mnc","output_roi_Gender.mnc",
    "output_roi_Intercept_t.mnc","output_roi_Age2_t.mnc","output_roi_Gender_Age2_t.mnc","output_roi_Age_t.mnc","output_roi_Gender_Age_t.mnc","output_roi_Gender_t.mnc"
    ],mask_file)

    # allocate space for input
    jacobian=np.zeros(shape=[inp.dim()],dtype=np.float64,order='C')

    # allocate space for output
    zero=np.zeros(shape=[out.dim()],dtype=np.float64,order='C')
    result=np.zeros(shape=[out.dim()],dtype=np.float64,order='C')

    
    # assign ariables - they stays the same 
    random_effects.environment["Subject"] = Subject
    fixed_effects.environment["Visit"]  = Visit
    fixed_effects.environment["Age"]    = Age
    fixed_effects.environment["Gender"] = Gender
    #fmla.environment["Scale"] = Scale

    # start iteration, not really needed
    inp.begin()
    out.begin()

    l=None
    try:
        while True:

            result=np.zeros(shape=[out.dim()],dtype=np.float64,order='C')
            if inp.value_mask():
                Jacobian=ro.FloatVector(inp.value())

                # update jacobian vairable
                fixed_effects.environment["Jacobian"] = Jacobian
                
                try:
                    # run linear model
                    l = base.summary(nlme.lme(fixed_effects,random=random_effects,method="ML"))
                    # extract coeffecients
                    result[0:6]  = l.rx2('coefficients').rx2('fixed')[:]
                    
                    # extract t-values
                    result[6:12] = l.rx2('tTable').rx(True,4)[:]
                    
                except RRuntimeError:
                    # probably model didn't converge
                    pass
            out.value(result)
            
            # move to the next voxel
            inp.next()
            out.next()
            # display something on the screen....
            sys.stdout.write(".")
            sys.stdout.flush()

    except StopIteration:
        pass

    # free up memory, not really needed 
    del inp
    del out


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
