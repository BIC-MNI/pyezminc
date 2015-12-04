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
# for timing
import time
def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Run intensity normalization to minimize spread within each tissue class ')
    
    parser.add_argument('prior',help="classification prior")
    
    parser.add_argument('image',help="Run classifier on a set of given images")
    
    parser.add_argument('--output',help="Output image")
    
    parser.add_argument('--trainmask', 
                    help="Training mask" )
    
    parser.add_argument('--iter', 
                    default=500,
                    type=int,
                    help="Number of iterations" )

    parser.add_argument('--debug', action="store_true",
                    dest="debug",
                    default=False,
                    help='Print debugging information' )
    
    parser.add_argument('--coord', action="store_true",
                    dest="coord",
                    default=False,
                    help='Use image coordinates as additional features' )
    
    parser.add_argument('--log', 
                    dest="log",
                    default=None,
                    help='Otput tensorboard log into this directory' )
    
    parser.add_argument('--neighbours', 
                    dest="neighbours",
                    default=None,
                    type=int,
                    help='Use neighbours as additional features' )
    
    parser.add_argument('--subsample',
                        default=None,
                        type=float,
                        help='randomly subsample training data')
    
    #parser.add_argument('--product',
                        #dest='product',
                        #default=False,
                        #action="store_true",
                        #help='Use product as additional feature')

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
        
        #images= [ minc.Image(i).data.astype(np.float32)  for i in options.image ]
        image=minc.Image(options.image).data.astype(np.float32)
        
        # add features dependant on coordinates
        c=np.mgrid[0:image.shape[0] , 0:image.shape[1] , 0:image.shape[2]].astype(np.float32)
        
        # normalized spatial coordinates
        _extents=max(image.shape)
        cx= c[0)/( _extents/2.0) )
        cy= c[1]/( _extents/2.0) )
        cz= c[2]/( _extents/2.0) )
        
        # generate basis functions
        
        _basis=()
        
        M=10 # number of basis functions
        # for now initialize in numpy
        # TODO: maybe move to tensor-flow 
        #
        for i in range(M):
            cosx=np.cos(np.pi*(cx+0.5)*i/M)
            for j in range(M):
                cosy=np.cos(np.pi*(cy+0.5)*j/M)
                for k in range(M):
                    cosz=np.cos(np.pi*(cz+0.5)*k/M)
                    _basis.append(cosx*cosy*cosz)
        
        num_basis=len(_basis)
        
        # initial coeffecients for normalization
        init_coeff=np.zeros([num_basis]).astype(np.float32)
        init_coeff[0]=1.0 
        
        image_ranges=[]
        
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

            num_classes = len(labels)

            if options.debug: 
              print("Available labels:{} counts: {} available images:{}".format(repr(labels),repr(counts),len(images)))
              print("Creating training dataset for classifier")

            if options.trainmask is not None:
                trainmask  = minc.Label(options.trainmask)
                training_X = np.ravel( image.data[  np.logical_and(prior.data>0 , trainmask.data>0 ) ]    ) 
                training_Y = np.ravel( prior.data[  np.logical_and(prior.data>0 , trainmask.data>0 ) ] -1 )
                basis=np.column_stack( tuple( np.ravel( j[np.logical_and(prior.data>0 , trainmask.data>0 )] ) for j in _basis  ) )
            else:
                training_X = np.ravel( image.data[prior.data>0]     )
                training_Y = np.ravel( prior.data[prior.data>0] - 1 )
                basis=np.column_stack( tuple( np.ravel( j[ prior.data>0 ] ) for j in _basis  ) )
            # 
            # 
            # 
            if options.debug: 
              print("Fitting...")
            
            x      = tf.placeholder("float32", [None] )
            y_     = tf.placeholder("int32",   [None] )
            basis_ = tf.placeholder("float32", [None, num_basis] )
            coeff  = tf.Variable(init_coeff , name="basis_coeff")
            
            # normalization field, normalized to have unit sum
            normalization = tf.reduce_sum( tf.mul( coeff, basis_ ), 0)/tf.reduce_sum(coeff)
            
            with tf.name_scope('correct') as scope:
                
                #means = tf.Variable( tf.zeros([ num_classes ], float32 ))
                #sds =   tf.Variable( tf.zeros([ num_classes ], float32 ))
                
                batch_size = tf.size( y_ )
                corr_x = tf.mul( x, normalization )
                # calculate standard deviations inside each class
                
                sigma = tf.Variable(0 , name="covariance" )
                
                for i in range( num_classes ):
                  
                  means = tf.segment_mean( corr_x, y_ )
                  s     = tf.squeeze(tf.split(0,num_classes, sigma)[i])
                    
                  if num_features==1: 
                    # special case with single modality
                    d     = tf.sub( x , M  )
                    m2    = tf.mul(tf.div(d,s),d)
                  else:
                    # calculate probabilities of class membership (multivariate gaussian)
                    d     = tf.sub( x , M  )
                    m1    = tf.matmul( d,  tf.matrix_inverse(s) )
                    m2    = tf.expand_dims( tf.reduce_sum( tf.mul( m1, d ), 1), -1) # have to replace matrix multiplication with this
                    
                  out.append( -0.5*m2 )# should be columns
                
                # align columns
                _out=tf.concat(1, out)
                y = tf.nn.softmax(_out) + _epsilon # to avoid taking log of 0
                
                cross_entropy = -tf.reduce_sum( onehot_labels * tf.log( y ) )
                loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
                accuracy = tf.reduce_mean(tf.cast(tf.equal(y_ , tf.to_int32(tf.argmax(y,1))), "float"))

                #opt  = tf.train.GradientDescentOptimizer(learning_rate=0.1)
                opt  = tf.train.AdagradOptimizer(learning_rate=0.1)
                #opt = tf.train.AdamOptimizer()
                train_step = opt.minimize(loss)# ,[mean,sigma]
                
                #w_hist = tf.histogram_summary("means", mean)
                #b_hist = tf.histogram_summary("sigmas", sigma)
                #y_hist = tf.histogram_summary("y", y)
                if options.log is not None:
                    y_hist = tf.histogram_summary("y", y)
                    cross_entropy_summary = tf.scalar_summary("cross_entropy", cross_entropy)
                    accuracy_hist_summary = tf.histogram_summary("accuracy_hist", tf.cast(tf.equal(y_ , tf.to_int32(tf.argmax(y,1))), "float"))
                    accuracy_summary = tf.scalar_summary("accuracy", accuracy)
                    
            if options.log is not None:
                summary_op = tf.merge_all_summaries()
                
            init = tf.initialize_all_variables()
            sess = tf.Session()
            
            if options.log is not None:
                print("Writing log to {}".format(options.log))
                writer = tf.train.SummaryWriter(options.log, sess.graph_def)
                
            sess.run(init)

            print("Initial values:")
            (initial_entropy,initial_mean,initial_sigma,initial_accuracy)= \
              sess.run([cross_entropy,
                        mean,
                        sigma,
                        accuracy], 
                        feed_dict={x: training_X, y_: training_Y} )
            print("entropy={},accuracy={}".format(initial_entropy,initial_accuracy))
            t0 = time.time()
            sub_iters=100
            epochs=options.iter/sub_iters
            training_X_=training_X
            training_Y_=training_Y
            for step in xrange(0, epochs):
                # get a new random subsample if needed 
                if options.subsample is not None:
                    subset=np.random.choice(num_datasets,size=num_datasets*options.subsample,replace=False)
                    training_Y_=training_Y[subset]
                    training_X_=training_X[subset,:]
                
                for sstep in xrange(sub_iters):
                    if options.log is not None:
                        (dummy,summary_str,_entropy,_mean,_sigma,_accuracy)= \
                            sess.run([train_step,
                                    summary_op,
                                    cross_entropy,
                                    mean,
                                    sigma,
                                    accuracy], 
                                feed_dict={x: training_X_, y_: training_Y_} )
                        writer.add_summary(summary_str, step*sub_iters+sstep)
                    else:
                        (dummy,_entropy,_mean,_sigma,_accuracy)= \
                            sess.run([train_step,
                                    cross_entropy,
                                    mean,
                                    sigma,
                                    accuracy], 
                                feed_dict={x: training_X_, y_: training_Y_} )

                print("{} - {},{}".format(step*sub_iters,_entropy,_accuracy))

            t1 = time.time()      
            (final_entropy,final_mean,final_sigma,final_accuracy)= \
              sess.run([cross_entropy,
                        mean,
                        sigma,
                        accuracy], 
                        feed_dict={x: training_X, y_: training_Y} )
            
            print("final means=",final_mean)
            print("initial means=",initial_mean)
            
            print("final sigmas=",final_sigma)
            print("initial sigmas=",initial_sigma)
            
            print("final accuracy=",final_accuracy)
            print("initial accuracy=",initial_accuracy)
            
            print("Elapsed time={}".format(t1-t0))

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
                out_cls=np.empty_like(image, dtype=np.int32 )
                inp=np.column_stack( tuple( np.ravel( j[ mask.data>0 ] ) for j in images  ) )
                
                out = sess.run( tf.argmax(y , 1)+1, feed_dict={x: inp} )
                out_cls[mask.data>0] = out.astype(np.int32)
            else:
                inp=np.column_stack( tuple( np.ravel( j ) for j in images  ) )
                out = sess.run( tf.argmax(y , 1)+1, feed_dict={x: inp} )
                out_cls = out.astype(np.int32)
            
            if options.debug: 
              print("Saving output...")
            
            out=minc.Label(data=out_cls)
            #out=minc.Image(data=out_cls.astype(np.float64))
            out.save(name=options.output, imitate=options.image[0],history=history)
    else:
        print "Error in arguments"

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on
