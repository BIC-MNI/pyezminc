#! /usr/bin/env python

# standard library
import string
import os
import argparse
import pickle
import sys
import json
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
                                 description='Perform error-correction learning and application')
    
    parser.add_argument('--train',help="Training library in json format")
    
    parser.add_argument('--input',help="Method to be corrected")
    
    parser.add_argument('--output',help="Output image")
    
    parser.add_argument('--mask', 
                    help="Region for correction" )
    
    parser.add_argument('--method',
                    choices=['SVM','lSVM','nuSVM','NN','RanForest','AdaBoost','tree'],
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
    
    parser.add_argument('--random', type=int,
                    dest="random",
                    help='Provide random state if needed' )
    
    parser.add_argument('--save',help='Save training results in a file')
    
    parser.add_argument('--load',help='Load training results from a file')
    
    parser.add_argumetn('image',help='Input images',nargs='+')
    
    options = parser.parse_args()
    
    return options

if __name__ == "__main__":
    history=minc.format_history(sys.argv)
    
    options = parse_options()
    
    # load training images
    if (options.train is not None or options.save is not None) :
        
        if options.debug: print("Loading training images...")
        
        with open(options.train,'rb') as f:
            train=json.load(f)
        
        training_images=[]
        training_err=[]
        
        #go over training samples
        clf=None
        
        for inp in train:
            
            mask=minc.Label(inp[-3]).data
            ground=minc.Label(inp[-2]).data
            auto=minc.Label(inp[-1]).data
            
            images=[ minc.Image(i).data for i in inp[0:-3] ]

            if options.coord:
                # add features dependant on coordinates
                c=np.mgrid[0:images[0].shape[0] , 0:images[0].shape[1] , 0:images[0].shape[2]]
                # use with center at 0 and 1.0 at the edge, could have used preprocessing 
                images.append( ( c[0]-images[0].shape[0]/2.0)/ (images[0].shape[0]/2.0) )
                images.append( ( c[1]-images[0].shape[1]/2.0)/ (images[0].shape[1]/2.0) )
                images.append( ( c[2]-images[0].shape[2]/2.0)/ (images[0].shape[1]/2.0) )
                
            images.append(auto) # add auto labelling as a feature
            # TODO add more features here

            # extract only what's needed
            training_images.append( [ i[mask>0] for i in images ] )
            training_err.append( logical_xor(ground[mask>0],auto[mask>0] )

        if options.debug: print("Done")

        clf=None

        training_X = np.hstack( tuple( np.column_stack( tuple( j for j in  training_images[i] ) for (i,k) in enumerate(training_images)   ) ) )
        training_Y = np.ravel( np.concatenate( tuple(j for j in training_err ) ) )

        if options.debug: print("Fitting...")

        if options.method=="SVM":
            clf = svm.SVC()
        elif options.method=="nuSVM":
            clf = svm.NuSVC()
        elif options.method=='NN':
            clf = neighbors.KNeighborsClassifier(options.n)
        elif options.method=='RanForest':
            clf = ensemble.RandomForestClassifier(n_estimators=options.n,random_state=options.random)
        elif options.method=='AdaBoost':
            clf = ensemble.AdaBoostClassifier(n_estimators=options.n,random_state=options.random)
        elif options.method=='tree':
            clf = tree.DecisionTreeClassifier(random_state=options.random)
        else:
            clf = svm.LinearSVC()
        
        clf.fit(training_X,training_Y)
        
        if options.debug: print(clf)
        
        if options.save is not None:
            with open(options.save,'wb') as f:
                pickle.dump(clf, f)
                
    elif options.load is not None and options.input is not Nona and options.image is not None:
        
        with open(options.load, 'rb') as f:
            clf = pickle.load(f)
        
        if options.debug: print(clf)
        
        if options.output is not None:
            if options.debug: print("Classifying...")
        
            out_cls=None
        
            if mask is not None:
                if options.debug: print("Using mask")
                out_cls=np.empty_like(images[0], dtype=np.int32 )
                out_cls[mask.data>0]=clf.predict( np.column_stack( tuple( np.ravel( j[ mask.data>0 ] ) for j in images  ) ) )
            else:
                out_cls=clf.predict( np.column_stack( tuple( np.ravel( j ) for j in images  ) ) )
        
            if options.debug: print("Saving output...")
            
            out=minc.Label(data=out_cls)
            out.save(name=options.output, imitate=options.image[0],history=history)
    else:
        print "Error in arguments"