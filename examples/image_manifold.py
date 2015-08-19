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
from sklearn import manifold, datasets

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Run tissue classifier ')
        
    parser.add_argument('image',help="Run classifier on a set of given images",nargs='+')
    
    parser.add_argument('--output',help="Output image (s)")
    
    parser.add_argument('--mask', 
                    help="Mask images, set to 0 outside" )

    parser.add_argument('--method',
                    choices=['LLE','Spectral'],
                    default='LLE',
                    help='Algorithm')
    
    parser.add_argument('-n',type=int,help="nearest neighbors",default=10)
    
    parser.add_argument('-c',type=int,help="Components",default=1)
    
    parser.add_argument('--patch',type=int,help="Patch radius, 0 = disable", default=0)
    
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
    
    options = parser.parse_args()
    
    return options

if __name__ == "__main__":
    history=minc.format_history(sys.argv)
    
    options = parse_options()
    
    if options.image is not None:
        if options.debug: print("Loading images...")
        
        images= [ minc.Image(i).data for i in options.image ]
        
        if options.patch>0:
            input_images=len(images)
            
            for i in range(input_images) :
                for x in range(-options.patch,options.patch+1) :
                    for y in range (-options.patch,options.patch+1) :
                        for z in range(-options.patch,options.patch+1) :
                            if not (x==0 and y==0 and z==0):
                                images.append( np.roll( np.roll( np.roll( images[i], shift=x, axis=0 ), shift=y, axis=1), shift=z, axis=2 ) )
            
        
        if options.coord:
            # add features dependant on coordinates
            c=np.mgrid[0:images[0].shape[0] , 0:images[0].shape[1] , 0:images[0].shape[2]]
        
            # use with center at 0 and 1.0 at the edge, could have used preprocessing 
            images.append( ( c[0]-images[0].shape[0]/2.0)/(images[0].shape[0]/2.0)*100 )
            images.append( ( c[1]-images[0].shape[1]/2.0)/(images[0].shape[1]/2.0)*100 )
            images.append( ( c[2]-images[0].shape[2]/2.0)/(images[0].shape[1]/2.0)*100 )

        mask=None
        
        if options.debug: print("Done")
        
        man=None
        training_X=None
        Y=None
        
        if options.debug: print("Creating training dataset for classifier")
    
        if options.mask is not None:
            mask=minc.Label(options.mask)
            training_X = np.column_stack( tuple( np.ravel( j[ mask.data>0] ) for j in images  ) )
        else:
            training_X = np.column_stack( tuple( np.ravel( j ) for j in images  ) )
            
        if options.debug: 
            print("Fitting {}, dataset size:{} ...".format(options.method,training_X.shape))
    
        if options.method=="LLE":
            man =  manifold.LocallyLinearEmbedding(
                                n_components=options.c,
                                n_neighbors=options.n,
                                eigen_solver='auto',
                                method='standard')
            Y=man.fit_transform(training_X)
                                        
        else: #if options.method=="Spectral":
            man = manifold.SpectralEmbedding(
                                n_components=options.c,
                                n_neighbors=options.n)
            Y=man.fit_transform(training_X)
            
        if options.debug: print(man)
        
        if options.output is not None:
            if options.debug: print("Writing output...")
        
            out_man=None
            
            for i in range(options.c):
                #
                if mask is not None:
                    out_man = np.zeros_like( images[0] )
                    out_man[ mask.data>0 ]=Y[:,i]
                else:
                    out_man = np.empty_like( images[0] )
                    out_man [:] =Y[:,i]
                    
                if options.debug: print("Saving output...")

                out=minc.Image(data=out_man)
                out.save(name="{}_{:d}.mnc".format(options.output,i), imitate=options.image[0], history=history)
    else:
        print "Error in arguments"

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80
