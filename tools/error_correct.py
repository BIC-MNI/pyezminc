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

def prepare_features(options, images, auto_seg, mask):
    
    if options.coord:
        # add features dependant on coordinates
        c=np.mgrid[0:images[0].shape[0] , 0:images[0].shape[1] , 0:images[0].shape[2]]
        # use with center at 0 and 1.0 at the edge, could have used preprocessing 
        images.append( ( c[0]-images[0].shape[0]/2.0)/ (images[0].shape[0]/2.0) )
        images.append( ( c[1]-images[0].shape[1]/2.0)/ (images[0].shape[1]/2.0) )
        images.append( ( c[2]-images[0].shape[2]/2.0)/ (images[0].shape[1]/2.0) )
        
    images.append( auto_seg ) # add auto labelling as a feature
    # TODO add more features here

    # extract only what's needed
    return [  i[ mask>0 ] for i in images ] 
    
def convert_image_list(images):
    s=[]
    for (i,k) in enumerate(images):
        s.append(np.column_stack( tuple( np.ravel( j ) for j in k ) ) )
        print s[-1].shape
    return np.vstack( tuple( i for i in s ) )

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Perform error-correction learning and application')
    
    parser.add_argument('--train',
                    help="Training library in json format")
    
    parser.add_argument('--input',
                    help="Automatic seg to be corrected")
    
    parser.add_argument('--output',
                    help="Output image, required for application of method")
    
    parser.add_argument('--mask', 
                    help="Region for correction, required for application of method" )
                        
    parser.add_argument('--method',
                    choices=['SVM','lSVM','nuSVM','NN','RanForest','AdaBoost','tree'],
                    default='lSVM',
                    help='Classification algorithm')
    
    parser.add_argument('-n',
                    type=int,
                    help="nearest neighbors",
                    default=15)
    
    parser.add_argument('--debug', 
                    action="store_true",
                    dest="debug",
                    default=False,
                    help='Print debugging information' )
    
    parser.add_argument('--coord', 
                    action="store_true",
                    dest="coord",
                    default=False,
                    help='Use image coordinates as additional features' )
    
    parser.add_argument('--random', 
                    type=int,
                    dest="random",
                    help='Provide random state if needed' )
    
    parser.add_argument('--save', 
                    help='Save training results in a file')
    
    parser.add_argument('--load', 
                    help='Load training results from a file')
    
    parser.add_argument('image',
                    help='Input images', nargs='*')
    
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
        
        for (i,inp) in enumerate(train):
            
            mask=minc.Label(   inp[-3] ).data
            ground=minc.Label( inp[-2] ).data
            auto=minc.Label(   inp[-1] ).data
            
            images=[ minc.Image(k).data for k in inp[0:-3] ]
            
            training_images.append( prepare_features( options, images, auto, mask ) )
            training_err.append( np.logical_xor( ground[mask>0], auto[mask>0] ) )
            
            print "....{}".format(i)
            
            if i == 0:
                print "Dumping images..."
                for (j,k) in enumerate( training_images[-1] ):
                    test=np.zeros_like( images[0] )
                    #print test.shape
                    #print k.shape
                    test[ mask>0 ]=k
                    out=minc.Image( data=test )
                    out.save( name="dump_{}.mnc".format(j), imitate=inp[0] )
        
        if options.debug: print("Done")

        clf=None

        training_X = convert_image_list( training_images )  
        print training_X
          #np.hstack( tuple( np.column_stack( tuple( j for j in  training_images[i] ) for (i,k) in enumerate(training_images)   ) ) )
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
        
        with open(options.save,'wb') as f:
            pickle.dump(clf, f)
                
    elif options.load  is not None and \
         options.input is not None and \
         options.image is not None and \
         options.mask  is not None and \
         options.output is not None:
        
        if options.debug: print("Runnin error-correction...")
        
        with open(options.load, 'rb') as f:
            clf = pickle.load(f)
        
        if options.debug: print(clf)
        
        if options.debug: print("Loading input images...")
        
        images=[ minc.Image( i ).data for i in options.image ]
        
        mask=minc.Label( options.mask ).data
        auto=minc.Label( options.input ).data
        
        out_cls=None
        test_x=convert_image_list ( [ prepare_features( options, images, auto, mask ) ] ) 
        print test_x
        #out_cls=np.copy( auto )
        out_cls=np.zeros_like( auto )
        
        if options.debug: print("Running classifier...")
        
        out_cls[mask>0] = clf.predict( test_x ) # np.logical_xor(clf.predict( test_x ), auto[mask>0] )
    
        if options.debug: print("Saving output...")
        
        out=minc.Label( data=out_cls )
        out.save(name=options.output, imitate=options.input, history=history)
    else:
        print "Error in arguments"
