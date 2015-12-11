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
    
    parser.add_argument('reference',help="Reference Image")
    
    parser.add_argument('image',help="Input image")
    
    parser.add_argument('output',help="Output image")
    
    parser.add_argument('--mask', 
                    help="Training mask" )
    
    parser.add_argument('--refmask', 
                    help="Training mask" )
    
    parser.add_argument('--iter', 
                    default=100,
                    type=int,
                    help="Number of iterations" )
    
    parser.add_argument('--sub_iter', 
                    default=10,
                    type=int,
                    help="Number of sub-iterations" )

    parser.add_argument('--debug', action="store_true",
                    dest="debug",
                    default=False,
                    help='Print debugging information' )
    
    parser.add_argument('-O', 
                    dest="O",
                    default=3,
                    type=int,
                    help='Order' )
    
    parser.add_argument('--bins',
                        default=100,
                        type=int,
                        help="Number of histogram bins")
    
    parser.add_argument('--bw',
                        default=1.0,
                        type=float,
                        help="Gaussian kernel BW")
    
    parser.add_argument('--log', 
                    dest="log",
                    default=None,
                    help='Otput tensorboard log into this directory' )
    
    parser.add_argument('--alpha', 
                    dest="alpha",
                    default=0.5,
                    type=float,
                    help='Use neighbours as additional features' )
    
    parser.add_argument('--subsample',
                        default=0.01,
                        type=float,
                        help='randomly subsample training data')
    
    parser.add_argument('--save',help='Save training results in a file')
    parser.add_argument('--load',help='Load training results from a file')
    
    options = parser.parse_args()
    
    return options

def tf_naive_histogram(hist_bins,values,bw):
    return tf.reduce_sum( tf.exp(tf.square( 
        tf.sub( tf.expand_dims( hist_bins, 1), tf.expand_dims( values, 0) ) )
     *(-1.0/bw))/np.sqrt(np.pi),1)/tf.to_float(tf.size(values))

def tf_apply_corr(vals,coeff):
    #ret=coeff[0]
    ret=vals*coeff[1]+coeff[0]
    for i in range(2,coeff.get_shape()[0]):
        ret=ret+tf.pow(vals,i)*coeff[i]
    return ret


if __name__ == "__main__":
    history=minc.format_history(sys.argv)
    
    options = parse_options()
    # load prior and input image
    if options.debug: print("Loading images...")
    # convert to float as we go
    
    ref_image=minc.Image(options.reference).data.astype(np.float32)
    image=minc.Image(options.image).data.astype(np.float32)
    
    if options.debug: print("Done")
    
    mm=(image>0)
    if options.mask is not None:
        mask  = minc.Label(options.mask)
        mm = np.logical_and(image>0 , mask.data>0 )
        
    rmm=(ref_image>0)
    if options.refmask is not None:
        refmask  = minc.Label(options.refmask)
        rmm = np.logical_and(ref_image>0 , refmask.data>0 )
    
    print ref_image[rmm]
    rmin=np.amin(ref_image[rmm])
    rmax=np.amax(ref_image[rmm])
    
    #num_basis=len(_basis)

    # initial coeffecients for normalization:
    coeff_count=options.O

    # initial linear fit
    init_coeff=np.zeros([coeff_count]).astype(np.float32)
    init_coeff[0]=0.0
    init_coeff[1]=1.0
    
    #basis=np.column_stack( tuple( j for j in _basis  ) )
    training_X = np.ravel( image[ mm ] ) 
    training_Y = np.ravel( ref_image[ rmm ] )
    
    if options.debug: 
        print("Fitting...")
        print("Number of unknowns:{}".format(coeff_count))
    
    x             = tf.placeholder("float32", [None] )
    y             = tf.placeholder("float32", [None] )  
    bw            = tf.constant(options.bw)  
    coeff         = tf.Variable(init_coeff ,  name="coeff")
    hist_bins     = tf.linspace(rmin,rmax,options.bins, "hist_bins")
    
    hist_ref  = tf_naive_histogram(hist_bins,y,bw)
    
    x_corr    = tf_apply_corr(x,coeff)
    
    hist_corr = tf_naive_histogram(hist_bins,x_corr,bw)+1e-20
    
    loss      = tf.reduce_sum(hist_ref*tf.log(hist_ref/hist_corr+1e-20))

    #opt  = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    opt  = tf.train.AdagradOptimizer(learning_rate=0.1)
    #opt = tf.train.AdamOptimizer()
    train_step = opt.minimize(loss)# ,[mean,sigma]
    
    if options.log is not None:
        loss_summary = tf.scalar_summary("loss", loss)

    if options.log is not None:
        summary_op = tf.merge_all_summaries()
        
    init = tf.initialize_all_variables()
    sess = tf.Session()
    
    if options.log is not None:
        print("Writing log to {}".format(options.log))
        writer = tf.train.SummaryWriter(options.log, sess.graph_def)
        
    sess.run(init)

    t0 = time.time()
    
    #np.savetxt("hist.csv",sess.run(hist_ref))
    #exit(1)
    
    #sub_iters=100
    epochs=options.iter/options.sub_iter
    
    training_X_=training_X
    training_Y_=training_Y
    num_datasets=training_X.shape[0]
    num_ref_datasets=training_Y.shape[0]
    
    (_loss)= \
        sess.run([loss],  
            feed_dict={x: training_X_,
                       y: training_Y_} )
    print("Initial loss:{} ".format(_loss))
    print("Using random subsample of {} for fitting".format(int(num_datasets*options.subsample)))
    
    for step in xrange(0, epochs):
        # get a new random subsample if needed 
        if options.subsample is not None:
            subset=np.random.choice(num_datasets, size=int(num_datasets*options.subsample), replace=False)
            training_X_=training_X[subset]
            rsubset=np.random.choice(num_ref_datasets, size=int(num_ref_datasets*options.subsample), replace=False)
            training_Y_=training_Y[subset]
        
        # now we have to sort samples in order for segment_mean to work properly 
        for sstep in xrange(options.sub_iter):
            if options.log is not None:
                (dummy,summary_str,_loss)= \
                    sess.run([train_step, summary_op, loss], 
                        feed_dict={x: training_X_,y: training_Y_},
                            )
                writer.add_summary(summary_str, step*options.sub_iter+sstep)
            else:
                (dummy,_loss)= \
                    sess.run([train_step, loss],  
                        feed_dict={x: training_X_, y: training_Y_} )
            print("{}".format(_loss))
            
        print("{} - {}".format((step+1)*options.sub_iter,_loss))
    
    t1 = time.time()
    print("Elapsed time={}".format(t1-t0))
    
    #srt=np.argsort(training_Y)
    #training_Y_srt=training_Y[srt]
    #training_X_srt=training_X[srt]
    #basis_srt=basis[srt,:]
    
    final_coeff= sess.run(coeff)
    #print("final loss=",final_loss)
    print("Final coeffecients=",final_coeff)

    if options.output is not None:
        if options.debug: print("Saving...")

        # TODO: reshape to match minc
        #_normalization/=np.sum(final_coeff)
        
        out=np.zeros_like(image)
        for i in range(coeff_count):
            out+=final_coeff[i]*np.power(image,i)
        
        out=minc.Image(data=out.astype(np.float64))
        #out=minc.Image(data=out_cls.astype(np.float64))
        out.save(name=options.output, imitate=options.image,history=history)

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on
