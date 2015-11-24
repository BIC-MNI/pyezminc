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

# tensor-flow
import tensorflow as tf

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
    
    parser.add_argument('--neighbours', 
                    dest="neighbours",
                    default=None,
                    type=int,
                    help='Use neighbours as additional features' )

    parser.add_argument('--save',help='Save training results in a file')
    parser.add_argument('--load',help='Load training results from a file')
    
    options = parser.parse_args()
    
    return options

if __name__ == "__main__":
    history=minc.format_history(sys.argv)
    
    options = parse_options()
    #print(repr(options))
    patch_size=options.neighbours
    # load prior and input image
    if (options.prior is not None or options.load is not None) and options.image is not None:
        if options.debug: print("Loading images...")
        # convert to float as we go
        images= [ minc.Image(i).data.astype(np.float32)  for i in options.image ]
        input_images=len(images)
        
        if options.neighbours is not None:
          for i in range(input_images) :
              for x in range(-patch_size,patch_size+1) :
                  for y in range (-patch_size,patch_size+1) :
                      for z in range(-patch_size,patch_size+1) :
                          if not (x==0 and y==0 and z==0): # skip the central voxel
                            images.append( np.roll( np.roll( np.roll( images[i], shift=x, axis=0 ), shift=y, axis=1), shift=z, axis=2 ) )
        if options.coord:
            # add features dependant on coordinates
            c=np.mgrid[0:images[0].shape[0] , 0:images[0].shape[1] , 0:images[0].shape[2]].astype(np.float32)
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
            #TODO: load classifications 
            pass
        else:
            prior=minc.Label(options.prior)
            
            labels=list(np.unique(prior.data))
            counts=list(np.bincount(np.ravel(prior.data)))
            
            if 0 in labels:
                if options.debug: print("Label 0 will be discarded...")
                labels.remove(0)
                counts.pop(0) # assume it's first
            
            features=len(images)
            outputs= len(labels)
            
            if options.debug: 
              print("Available labels:{} counts: {} available images:{}".format(repr(labels),repr(counts),len(images)))
              print("Creating training dataset for classifier")
            
            if options.trainmask is not None:
                trainmask  = minc.Label(options.trainmask)
                training_X = np.column_stack( tuple( np.ravel( j[  np.logical_and(prior.data>0 , trainmask.data>0 ) ] ) for j in images ) )
                training_Y = np.eye(outputs) [ np.ravel( prior.data[  np.logical_and(prior.data>0 , trainmask.data>0 ) ] -1 ) ].astype(np.float32)
            else:
                training_X = np.column_stack( tuple( np.ravel( j[prior.data>0] ) for j in images  ) )
                training_Y = np.eye(outputs) [ np.ravel( prior.data[prior.data>0] -1 ) ].astype(np.float32)
            
            if options.debug: 
              print("Fitting...")
              print(training_Y)
            
            W = tf.Variable( tf.zeros([features,outputs]) )
            b = tf.Variable( tf.zeros([outputs]) )
            
            # TODO: run tensor-flow style training here
            x = tf.placeholder("float32", [None, features])
            y = tf.nn.softmax(tf.matmul(x,W) + b)
            
            y_ = tf.placeholder("float32", [None,outputs])
            cross_entropy = -tf.reduce_sum(y_*tf.log(y))
            train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
            
            init = tf.initialize_all_variables()
            sess = tf.Session()
            sess.run(init)
            
            sess.run(train_step, feed_dict={x: training_X, y_: training_Y})
            
            correct_prediction = tf.equal(tf.argmax(training_Y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            
            print sess.run(accuracy, feed_dict={x: training_X, y_: training_Y})
            print sess.run(W), sess.run(b)
        if options.debug: 
          #TODO: print W and b
          pass
        
        if options.save is not None:
            # TODO: save classification results to file
            pass
        
        if options.output is not None:
            if options.debug: print("Classifying...")
            
            out_cls=None
            
            if mask is not None:
                if options.debug: print("Using mask")
                out_cls=np.empty_like(images[0], dtype=np.int32 )
                inp=np.column_stack( tuple( np.ravel( j[ mask.data>0 ] ) for j in images  ) )
                # fit the data
                out = tf.argmax(tf.nn.softmax( tf.matmul(inp,W) + b ) , 1 )
                out_cls[mask.data>0] = sess.run(out) + 1 
            else:
                inp=np.column_stack( tuple( np.ravel( j ) for j in images  ) )
                out = tf.argmax(tf.nn.softmax( tf.matmul(inp,W) + b ) , 1 )
                out_cls = (sess.run(out) + 1 ).astype(np.int32)
            
            if options.debug: 
              print("Saving output...")
            
            out=minc.Label(data=out_cls)
            #out=minc.Image(data=out_cls.astype(np.float64))
            out.save(name=options.output, imitate=options.image[0],history=history)
    else:
        print "Error in arguments"
