#! /usr/bin/env python

# standard library
import string
import os
import argparse
import pickle
import sys
# minc
import minc

# numpy
import numpy as np

# scikit-learn
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn import tree

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Run tissue classifier ')
    
    parser.add_argument('prior',help="classification prior")
    
    parser.add_argument('image',help="Run classifier on a set of given images",nargs='+')
    
    parser.add_argument('--output',help="Output image")
    
    parser.add_argument('--mask', 
                    help="Mask output results, set to 0 outside" )
                    
    parser.add_argument('--trainmask', 
                    help="Training mask" )

    parser.add_argument('--debug', action="store_true",
                    dest="debug",
                    default=False,
                    help='Print debugging information' )
    
    parser.add_argument('--coord', action="store_true",
                    dest="coord",
                    default=False,
                    help='Use image coordinates as additional features' )

    parser.add_argument('--save',help='Save training results in a file')
    parser.add_argument('--load',help='Load training results from a file')
    
    options = parser.parse_args()
    
    return options

if __name__ == "__main__":
    history=minc.format_history(sys.argv)
    
    options = parse_options()
    #print(repr(options))
    
    # load prior and input image
    if (options.prior is not None or options.load is not None) and options.image is not None:
        if options.debug: print("Loading images...")
        
        images= [ minc.Image(i).data for i in options.image ]

        if options.coord:
            # add features dependant on coordinates
            c=np.mgrid[0:images[0].shape[0] , 0:images[0].shape[1] , 0:images[0].shape[2]]
        
            # use with center at 0 and 1.0 at the edge, could have used preprocessing 
            images.append( ( c[0]-images[0].shape[0]/2.0)/ (images[0].shape[0]/2.0) )
            images.append( ( c[1]-images[0].shape[1]/2.0)/ (images[0].shape[1]/2.0) )
            images.append( ( c[2]-images[0].shape[2]/2.0)/ (images[0].shape[1]/2.0) )

        mask=None
        if options.mask is not None:
            mask=minc.Label(options.mask)
        if options.debug: print("Done")
        
        clf=None
        
        
        if options.load is not None:
            with open(options.load, 'rb') as f:
                clf = pickle.load(f)
        else:
            prior=minc.Label(options.prior)
            
            labels=list(np.unique(prior.data))
            counts=list(np.bincount(np.ravel(prior.data)))
        
            if 0 in labels:
                if options.debug: print("Label 0 will be discarded...")
                labels.remove(0)
                counts.pop(0) # assume it's first
        
            if options.debug: 
              print("Available labels:{} counts: {} available images:{}".format(repr(labels),repr(counts),len(images)))
              print("Creating training dataset for classifier")
        
            if options.trainmask is not None:
                trainmask = minc.Label(options.trainmask)
            
                training_X = np.column_stack( tuple( np.ravel( j[ np.logical_and(prior.data>0 , trainmask.data>0 ) ] ) for j in images  ) )
                training_Y = np.ravel( prior.data[ np.logical_and(prior.data>0 , trainmask.data>0 ) ] )
            else:
                training_X = np.column_stack( tuple( np.ravel( j[ prior.data>0 ] ) for j in images  ) )
                training_Y = np.ravel( prior.data[ prior.data>0 ] )
        
        
            if options.debug: print("Fitting...")
        
            # TODO: run tensor-flow style training here
            pass
            
        if options.debug: print(clf)
        
        if options.save is not None:
            # TODO: save classification results to file
            pass
        
        if options.output is not None:
            if options.debug: print("Classifying...")
        
            out_cls=None
        
            if mask is not None:
                if options.debug: print("Using mask")
                out_cls=np.empty_like(images[0], dtype=np.int32 )
                #out_cls[mask.data>0]=clf.predict( np.column_stack( tuple( np.ravel( j[ mask.data>0 ] ) for j in images  ) ) )
                # TODO: apply classification here
            else:
                #out_cls=clf.predict( np.column_stack( tuple( np.ravel( j ) for j in images  ) ) )
                # TODO: apply classification here
        
            if options.debug: print("Saving output...")
            
            out=minc.Label(data=out_cls)
            out.save(name=options.output, imitate=options.image[0],history=history)
    else:
        print "Error in arguments"
