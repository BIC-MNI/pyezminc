#! /usr/bin/env python

# standard library
import string
import os
import argparse
import pickle
import cPickle 
import sys
import json
import csv
# minc
import minc

# numpy
import numpy as np

# scikit-learn
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn import tree
from sklearn import cross_validation
from sklearn import preprocessing

def prepare_features(options, images, coords, auto_seg, mask):
    '''Convert list of input imates into numpy array to be used as input to classifier'''
    
    #if options.coord:
    # add features dependant on coordinates
    c=np.mgrid[ 0:images[0].shape[0] , 
                0:images[0].shape[1] , 
                0:images[0].shape[2] ]
    
    image_no=len(images)
    # use with center at 0 and 1.0 at the edge, could have used preprocessing 
    
    if options.coord and coords is None:
        images.append( ( c[0]-images[0].shape[0]/2.0)/ (images[0].shape[0]/2.0) )
        images.append( ( c[1]-images[0].shape[1]/2.0)/ (images[0].shape[1]/2.0) )
        images.append( ( c[2]-images[0].shape[2]/2.0)/ (images[0].shape[1]/2.0) )
    else: # assume we have three sets of coords
        images.append( coords[0] )
        images.append( coords[1] )
        images.append( coords[2] )
    
    # assume binary labelling here
    aa=auto_seg-0.5
    
    # add auto labelling as a feature
    images.append( aa ) 
    
    # add apparance and context images
    for x in range(-1,2) :
        for y in range (-1,2) :
            for z in range(-1,2) :
                if x!=0 or y!=0 or z!=0 :
                    images.append( np.roll( np.roll( np.roll( images[0], shift=x, axis=0 ), shift=y, axis=1), shift=z, axis=2 ) )
                    # add more context
                    #images.append( np.roll( np.roll( np.roll( aa,        shift=x, axis=0 ), shift=y, axis=1), shift=z, axis=2 ) )
    
    app_features=len(images)-image_no-3 # 3 spatial features
    
    # add joint features
    if options.joint and options.coord:
        for i in range(app_features):
            # multiply apparance features by coordinate features 
            images.append( images[0] * images[i+image_no+3] ) 
            for j in range(3):
                # multiply apparance features by coordinate features 
                images.append( images[j+image_no] * images[i+image_no+3] ) 
    
    # extract only what's needed
    return [  i[ mask>0 ] for i in images ] 


def convert_image_list(images):
    '''convert list of images into np array'''
    s=[]
    for (i,k) in enumerate(images):
        s.append(np.column_stack( tuple( np.ravel( j ) for j in k ) ) )
        print s[-1].shape
    
    return np.vstack( tuple( i for i in s ) )


def parse_options():
    '''parse command-line options'''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Perform error-correction learning and application')
    
    parser.add_argument('--train',
                    help="Training library in json format (array of) image1,[image2,...imageN,]mask,ground-truth,auth")
                    
    parser.add_argument('--train_csv',
                    help="Training library in CSV format: image1,[image2,...imageN,]mask,ground-truth,auth")
    
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
                    
    parser.add_argument('--dump', 
                    action="store_true",
                    dest="dump",
                    default=False,
                    help='Dump first sample features (for debugging)' )
    
    parser.add_argument('--coord', 
                    action="store_true",
                    dest="coord",
                    default=False,
                    help='Use image coordinates as additional features' )
                    
    parser.add_argument('--joint', 
                    action="store_true",
                    dest="joint",
                    default=False,
                    help='Produce joint features between appearance and coordinate' )
    
    parser.add_argument('--normalize', 
                    action="store_true",
                    dest="normalize",
                    default=False,
                    help='Normalized input images to have zero mean and unit variance' )
    
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
    if ( (options.train     is not None or \
          options.train_csv is not None )  and \
          options.save      is not None ) :
        
        if options.debug: print("Loading training images...")
        
        train=None
        if options.train is not None:
            with open(options.train,'rb') as f:
                train=json.load(f)
        else:
            with open(options.train_csv,'rb') as f:
                train=list(csv.reader(f))
        
        training_images=[]
        training_output=[]
        training_err=[]
        
        #go over training samples
        clf=None
        
        #scaler=preprocessing.StandardScaler().fit(X)
        
        for (i,inp) in enumerate(train):
            mask  =minc.Label(  inp[-3] ).data
            ground=minc.Label(  inp[-2] ).data
            auto  =minc.Label(  inp[-1] ).data
            
            # normalize input features to zero mean and unit std
            if options.normalize:
              images=[ preprocessing.scale(minc.Image(k).data) for k in inp[0:-3] ]
            else:
              images=[ minc.Image(k).data for k in inp[0:-3] ]
            
            # store training data
            training_images.append( prepare_features( options, images, None, auto, mask ) )
            
            # perform direct learning right now
            training_output.append( ground[mask>0] )
            
            # dump first dataset for debugging
            if i == 0 and options.dump:
                print "Dumping feature images..."
                for (j,k) in enumerate( training_images[-1] ):
                    test=np.zeros_like( images[0] )
                    test[ mask>0 ]=k
                    out=minc.Image( data=test )
                    out.save( name="dump_{}.mnc".format(j), imitate=inp[0] )
                    
        if options.debug: print("Done")

        clf=None
        
        # convert into  a large array
        training_X = convert_image_list( training_images )  
        training_Y = np.ravel( np.concatenate( tuple(j for j in training_output ) ) )

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
        
        #scores = cross_validation.cross_val_score(clf, training_X, training_Y)
        #print scores
        
        clf.fit( training_X, training_Y )
        
        #print(clf.score(training_X,training_Y))
        
        if options.debug: print( clf )
        
        with open(options.save,'wb') as f:
            cPickle.dump(clf, f, -1)
                
    elif options.load  is not None and \
         options.input is not None and \
         options.image is not None and \
         options.mask  is not None and \
         options.output is not None:
        
        if options.debug: print( "Runnin error-correction..." )
        
        with open(options.load, 'rb') as f:
            clf = cPickle.load(f)
        
        if options.debug: print( clf )
        
        if options.debug: print( "Loading input images..." )
        
        if options.normalize:
          images=[ preprocessing.scale( minc.Image( i ).data ) for i in options.image ]
        else:
          images=[ minc.Image( i ).data for i in options.image ]

        mask=minc.Label( options.mask ).data
        auto=minc.Label( options.input ).data

        out_cls=None
        test_x=convert_image_list ( [ prepare_features( options, images, None, auto, mask ) ] ) 
        #print test_x

        out_cls=np.copy( auto ) # use input data 
        #out_cls=np.zeros_like( auto )
        
        if options.debug: print("Running classifier...")
        
        out_cls[ mask>0 ] = clf.predict( test_x ) # np.logical_xor(clf.predict( test_x ), auto[mask>0] )
    
        if options.debug: print("Saving output...")
        
        out=minc.Label( data=out_cls )
        out.save(name=options.output, imitate=options.input, history=history)
    else:
        print "Error in arguments, run with --help"


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80
