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

# using SCOOP https://code.google.com/p/scoop/
from scoop import futures, shared


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


# define R objects globally, so that we don't have to transfer them between instances
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

# assign ariables - they stays the same 

# allocate R formula, saves time for interpreter
random_effects = ro.Formula('~1|Subject')

zero=np.zeros(shape=[12],dtype=np.float64,order='C')

def run_nlme(jacobian):
    fixed_effects = ro.Formula('Jacobian ~ I(Age^2) + Gender:I(Age^2) + Age + Gender:Age + Gender')

    fixed_effects.environment["Subject"] = Subject
    fixed_effects.environment["Visit"]   = Visit
    fixed_effects.environment["Age"]     = Age
    fixed_effects.environment["Gender"]  = Gender
    
    # update jacobian variable
    fixed_effects.environment["Jacobian"] = ro.FloatVector(jacobian)

    # allocate space for output
    result=np.zeros(shape=[12],dtype=np.float64,order='C')

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

    return result

if __name__ == "__main__":
    
    # setup automatic conversion for numpy

    inp=pyezminc.parallel_input_iterator()
    
    inp.open(data['Filename'],mask_file )
    
    # allocate space for input
    

    #fmla.environment["Scale"] = Scale

    # start iteration, not really needed
    inp.begin()

    l=None
    masked=[]
    results=[]
    # submit executions
    try:
        while True:
            if inp.value_mask():
                masked.append(True)
                #jacobian=np.zeros(shape=[inp.dim()],dtype=np.float64,order='C')
                
                results.append(futures.submit(run_nlme,inp.value()))
            else:
                # we are passing-by voxels outside of the mask,
                # assign default value
                masked.append(False)
            # move to the next voxel
            inp.next()
    except StopIteration:
        pass
    
    del inp
    
    out=pyezminc.parallel_output_iterator()
    
    out.open(["output_Intercept.mnc","output_Age2.mnc","output_Gender_Age2.mnc","output_Age.mnc","output_Gender_Age.mnc","output_Gender.mnc",
    "output_Intercept_t.mnc","output_Age2_t.mnc","output_Gender_Age2_t.mnc","output_Age_t.mnc","output_Gender_Age_t.mnc","output_Gender_t.mnc"
    ],mask_file)
    
    # collect results
    out.begin()
    k=0
    try:
        for i in masked:
            if i :
                # get result of processing
                out.value(results[k].result())
                k+=1
            else:
                # it was masked away
                out.value(zero)
            out.next()
    except StopIteration:
        pass
    # free up memory, not really needed 
    del out

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
