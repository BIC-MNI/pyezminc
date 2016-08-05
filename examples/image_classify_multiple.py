#! /usr/bin/env python

# standard library
import string
import os
import argparse
import pickle
import sys
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

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Run tissue classifier ')
    
    parser.add_argument('--train',help="Training library, in csv format <img1>[,<img2>,...<imgN>],<labels>")
    
    parser.add_argument('images',help="Run classifier on a set of given image(s)",nargs='+')
    
    parser.add_argument('--priors',help="Add set of priors",nargs='*')
    
    parser.add_argument('--output',help="Output image")
    
    parser.add_argument('--mask', 
                    help="Mask output results, set to 0 outside" )
    
    parser.add_argument('--trainmask', 
                    help="Mask traing library, set to 0 outside" )
    
    parser.add_argument('--method',
                    choices=['SVM',
                             'lSVM',
                             'nuSVM',
                             'NN',
                             'RanForest',
                             'AdaBoost',
                             'tree'],
                    default='lSVM',
                    help='Classification algorithm')
    
    parser.add_argument('-n',type=int,help="nearest neighbors",default=15)
    
    parser.add_argument('-j','--jobs',type=int,help="Number of jobs, -1 - maximum",default=1)
    
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
    
    options = parser.parse_args()
    
    return options

if __name__ == "__main__":
    history=minc.format_history(sys.argv)
    
    options = parse_options()
    
    
    # load prior and input image
    clf=None
    
    if options.train is not None :
        prefix=os.path.dirname(options.train)
        with open(options.train,'r') as f:
            train_library=list(csv.reader(f))
    
        n_i=-1
        for i in train_library:
            if n_i==-1:
                n_i=len(i)-1
            elif n_i!=(len(i)-1):
                raise "Inconsistent number of images:{}".format(repr(i))
        
        if n_i==-1:
            raise "No input images!"
        
        
        if options.debug: 
            print("Loading {} images ...".format(n_i*len(train_library)))
        
        
        images = [ [ minc.Image(os.path.join(prefix,j[i])).data for i in range(n_i) ] for j in train_library ]
        segs   = [   minc.Label(os.path.join(prefix,j[n_i])).data                     for j in train_library ]
        
        priors = [ ]
        # TODO: check shape of all images for consistency
        _shape = images[0][0].shape
        
        if options.coord:
            # add features dependant on coordinates
            c=np.mgrid[ 0:_shape[0] , 0:_shape[1] ,0:_shape[2] ]
            # use with center at 0 and 1.0 at the edge, could have used preprocessing
            priors.append( ( c[0]-_shape[0]/2.0)/ (_shape[0]/2.0) )
            priors.append( ( c[1]-_shape[1]/2.0)/ (_shape[1]/2.0) )
            priors.append( ( c[2]-_shape[2]/2.0)/ (_shape[1]/2.0) )
        
        if options.priors is not None:
            for i in options.priors:
                priors.append(minc.Image(i).data)
        
        # append priors
        for i,j in enumerate(images):
            images[i].extend(priors)
        
        #for i,j in enumerate(images):
            #sh=[ repr(k.shape) for k in j ]
            #print("{} - {} - {}".format(i,':'.join(sh),segs[i].shape))
        # 
        
        if options.debug: print("Creating training dataset for classifier")
        
        if options.trainmask is not None:
            trainmask = minc.Label(options.trainmask)
        
            training_X = np.concatenate( tuple( np.column_stack( tuple( np.ravel( j[ trainmask.data>0 ] ) for j in i  ) ) for i in images ) )
            training_Y = np.concatenate( tuple( np.ravel( s[ trainmask.data>0 ] ) for s in segs ) )
        else:
            training_X = np.concatenate( tuple( np.column_stack( tuple( np.ravel( j ) for j in i  ) ) for i in images ) )
            training_Y = np.concatenate( tuple( np.ravel( s ) for s in segs ) )
        
        # try to cleanup space
        
        print("training_X.shape={}".format(training_X.shape))
        print("training_Y.shape={}".format(training_Y.shape))
        
        labels=list(np.unique(training_Y))
        counts=list(np.bincount(np.ravel(training_Y)))
    
        #if 0 in labels:
            #if options.debug: print("Label 0 will be discarded...")
            #labels.remove(0)
            #counts.pop(0) # assume it's first
    
        if options.debug: print("Available labels:{} counts: {} ".format(repr(labels),repr(counts)))
    
        if options.debug: print("Fitting...")
    
        if options.method=="SVM":
            clf = svm.SVC()
        elif options.method=="nuSVM":
            clf = svm.NuSVC()
        elif options.method=='NN':
            clf = neighbors.KNeighborsClassifier(options.n)
        elif options.method=='RanForest':
            clf = ensemble.RandomForestClassifier(n_estimators=options.n,random_state=options.random,n_jobs=options.jobs)
        elif options.method=='GradBoost':
            clf = ensemble.GradientBoostingClassifier(n_estimators=options.n,random_state=options.random,n_jobs=options.jobs)
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
        
    if options.load is not None:
        with open(options.load, 'rb') as f:
            clf = pickle.load(f)

    if options.debug:
        if hasattr(clf, 'n_classes_'): print("n_classes={}".format(clf.n_classes_))
        if hasattr(clf, 'feature_importances_'): print("importance={}".format(repr(clf.feature_importances_)))
    
    if (options.output is not None) and (options.images is not None):
        
        images= [ minc.Image(i).data for i in options.images ]
        
        _shape = images[0].shape
        
        priors = [ ]
        # TODO: check shape of all images for consistency
        
        if options.coord:
            # add features dependant on coordinates
            c=np.mgrid[ 0:_shape[0] , 0:_shape[1] ,0:_shape[2] ]
            # use with center at 0 and 1.0 at the edge, could have used preprocessing
            priors.append( ( c[0]-_shape[0]/2.0)/ (_shape[0]/2.0) )
            priors.append( ( c[1]-_shape[1]/2.0)/ (_shape[1]/2.0) )
            priors.append( ( c[2]-_shape[2]/2.0)/ (_shape[1]/2.0) )
        
        if options.priors is not None:
            for i in options.priors:
                priors.append(minc.Image(i).data)

        mask=None
        
        if options.mask is not None:
            mask=minc.Label(options.mask)
        
        # append priors
        images.extend(priors)
        
        out_cls=None
        if options.debug: print("Classifying...")
    
        if mask is not None:
            if options.debug: print("Using mask")
            out_cls=np.zeros_like(images[0], dtype=np.int32 )
            out_cls[mask.data>0]=clf.predict( np.column_stack( tuple( np.ravel( j[ mask.data>0 ] ) for j in images  ) ) )
        else:
            out_cls=clf.predict( np.column_stack( tuple( np.ravel( j ) for j in images  ) ) )
        
        if options.debug: print("Saving output...")
        
        out=minc.Label(data=out_cls)
        out.save(name=options.output, imitate=options.images[0],history=history)
    else:
        print "Error in arguments"
        