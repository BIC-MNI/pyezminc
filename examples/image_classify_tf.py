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

        image_ranges=[]
        
        for i in range(len(images)):
          image_ranges.append(np.ptp(images[i]))

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

            num_features=len(images)
            num_classes= len(labels)

            if options.debug: 
              print("Available labels:{} counts: {} available images:{}".format(repr(labels),repr(counts),len(images)))
              print("Creating training dataset for classifier")

            if options.trainmask is not None:
                trainmask  = minc.Label(options.trainmask)
                training_X = np.column_stack( tuple( np.ravel( j[  np.logical_and(prior.data>0 , trainmask.data>0 ) ] ) for j in images ) )
                training_Y = np.ravel( prior.data[  np.logical_and(prior.data>0 , trainmask.data>0 ) ] -1 )
            else:
                training_X = np.column_stack( tuple( np.ravel( j[prior.data>0] ) for j in images  ) )
                training_Y = np.ravel( prior.data[prior.data>0] - 1 )

            if options.debug: 
              print("Fitting...")
              #np.set_printoptions(threshold='nan')
              #print( training_Y )
              
            x  = tf.placeholder("float32", [None, num_features] )
            y_ = tf.placeholder("int32",   [None] )

            # W = tf.Variable( tf.zeros([num_features,outputs]) )
            # b = tf.Variable( tf.zeros([outputs]) )

            with tf.name_scope('gaussian') as scope:
                # initialize mean and covariance matrix
                batch_size = tf.size(y_)
                _labels  = tf.expand_dims(y_, 1)
                _indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
                
                concated = tf.concat(1, [_indices, _labels])
                
                onehot_labels = tf.to_float(tf.sparse_to_dense(
                    concated, tf.pack([batch_size, num_classes]), 1.0, 0.0))
                
                # going to iterate over all outputs, because I don't know how to do it easier
                out=[]
                
                #mean  = tf.Variable( tf.zeros(         [ num_classes, num_features]   ) , name="means")
                # create uniformly distributed means
                sigma = tf.Variable( tf.concat(0,      [ tf.expand_dims( tf.diag( tf.ones( [ num_features] ) ), 0) for k in range(num_classes)] ), name="covariance" )
                
                for i in range( num_classes ):
                  
                  M     = tf.squeeze(tf.split(0,num_classes, mean )[i])
                  s     = tf.squeeze(tf.split(0,num_classes, sigma)[i])
                    
                  sdet=1.0
                  
                  if num_features==1: # special case with single modality
                    m2    = tf.squeeze(tf.mul(tf.div(tf.sub(x,M),s),tf.sub(x,M)))
                    sdet  = s
                  else:
                    # calculate probabilities of class membership (multivariate gaussian)
                    d     = tf.sub( x , M  )
                    s_inv = tf.matrix_inverse(s)
                    
                    m1    = tf.matmul( d,  s_inv )
                    m2    = tf.matmul( m1, d, transpose_b=True)
                    sdet  = tf.matrix_determinant(s)
                    
                  out.append( tf.expand_dims( tf.exp( -0.5*m2 )/sdet, -1 ))# should be columns
                
                # align columns
                y=tf.nn.softmax(tf.concat(1, out))
                
                cross_entropy = -tf.reduce_sum( onehot_labels * tf.log( y ) )
                
                loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
                opt  = tf.train.GradientDescentOptimizer(learning_rate=0.1)
                train_step = opt.minimize(loss)

            init = tf.initialize_all_variables()
            sess = tf.Session()
            sess.run(init)
            
            #print(sess.run(tf.matrix_inverse(s1)))
            #print(sess.run(M1))
            #print( y.get_shape())
            #print( sess.run( tf.slice(y,[0,0],[1,2]), feed_dict={x: training_X, y_: training_Y}))
            print( sess.run( onehot_labels, feed_dict={x: training_X, y_: training_Y})) 
            print( sess.run( y, feed_dict={x: training_X, y_: training_Y})) 
            print( sess.run(cross_entropy, feed_dict={x: training_X, y_: training_Y}))
            
            for step in xrange(0, 200):
                sess.run(train_step, feed_dict={x: training_X, y_: training_Y} )
                #opt.run()
                #if step % 20 == 0:
                print step, sess.run(cross_entropy, feed_dict={x: training_X, y_: training_Y})

            print "mean=",sess.run(mean)
            print "sigma=",sess.run(sigma),
            
            correct_prediction = tf.equal(training_Y , tf.to_int32(tf.argmax(y,1)))
            
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print "Accuracy=",sess.run(accuracy, feed_dict={x: training_X, y_: training_Y})

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
