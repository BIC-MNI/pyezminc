#! /usr/bin/env python

import string
import os
import numpy as np
import subprocess as sp
import argparse
import minc

from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble


def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Run tissue classifier ')
    
    parser.add_argument('prior',help="classification prior")
    
    parser.add_argument('image',help="Run classifier on a set of given images",nargs='+')
    
    parser.add_argument('output',help="Output image")
    
    parser.add_argument('--mask', 
                    help="Use this mask" )
                    
    parser.add_argument('--method',
                    choices=['SVM','lSVM','nuSVM','NN','RanForest','AdaBoost'],
                    default='lSVM',
                    help='Classification algorithm')
                    
    parser.add_argument('-n',type=int,help="nearest neighbors",default=15)
    
    parser.add_argument('--debug', action="store_true",
                    dest="debug",
                    default=False,
                    help='Print debugging information' )
                    
    parser.add_argument('--coord', action="store_true",
                    dest="coord",
                    default=False,
                    help='Use image coordinates as additional features' )
    
    options = parser.parse_args()
    
    return options

if __name__ == "__main__":
    options = parse_options()
    
    #print(repr(options))
    
    # load prior and input image
    if options.prior is not None and options.image is not None:
        print("Loading images...")
        prior=minc.Label(options.prior)
        images= [ minc.Image(i).data for i in options.image ]
        
        if options.coord:
            # add features dependant on coordinates
            c=np.mgrid[0:images[0].shape[0] , 0:images[0].shape[1] , 0:images[0].shape[2]]
            images.append(c[0])
            images.append(c[1])
            images.append(c[2])
            
        mask=None
        if options.mask is not None:
            mask=minc.Label(options.mask)
        print("Done")
        
        
        labels=list(np.unique(prior.data))
        counts=list(np.bincount(np.ravel(prior.data)))
        
        if 0 in labels:
            print("Label 0 will be discarded...")
            labels.remove(0)
            counts.pop(0) # assume it's first
        
        print("Available labels:{} counts: {} available images:{}".format(repr(labels),repr(counts),len(images)))
        
        
        print("Creating training dataset for classifier")
        training_X= np.column_stack( tuple( np.ravel(j[prior.data>0]) for j in images  ) )
        training_Y= np.ravel( prior.data[prior.data>0])
        
        print(training_X.shape)
        print(training_Y.shape)
        
        print("Fitting...")
        
        if options.method=="SVM":
            clf = svm.SVC()
        elif options.method=="nuSVM":
            clf = svm.NuSVC()
        elif options.method=='NN':
            clf = neighbors.KNeighborsClassifier(options.n)
        elif options.method=='RanForest':
            clf = ensemble.RandomForestClassifier(n_estimators=options.n)
        elif options.method=='AdaBoost':
            clf = ensemble.AdaBoostClassifier(n_estimators=options.n)
        else:
            clf = svm.LinearSVC()
        
        
        clf.fit(training_X,training_Y)
        
        print(clf)
        
        print("Classifying...")
        
        out_cls=None
        
        if mask is not None:
            print("Using mask")
            out_cls=np.empty_like(prior.data)
            out_cls[mask.data>0]=clf.predict( np.column_stack( tuple( np.ravel( j[ mask.data>0 ] ) for j in images  ) ) )
        else:
            out_cls=clf.predict( np.column_stack( tuple( np.ravel( j ) for j in images  ) ) )
        
        print("Saving output...")
        out=minc.Label(data=out_cls)
        out.save(name=options.output,imitate=options.prior)
    else:
        print "Error in arguments"