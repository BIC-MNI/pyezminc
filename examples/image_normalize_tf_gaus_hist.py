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
    
    parser.add_argument('ref',help="reference image")
    
    parser.add_argument('image',help="Run classifier on a set of given images")
    
    parser.add_argument('--output',help="Output image")
    
    parser.add_argument('--mask', 
                    help="Training mask" )
    
    parser.add_argument('--refmask', 
                    help="Ref mask" )
    
    parser.add_argument('--iter', 
                    default=500,
                    type=int,
                    help="Number of iterations" )
    
    parser.add_argument('--sub_iter', 
                    default=100,
                    type=int,
                    help="Number of sub-iterations" )

    parser.add_argument('--debug', action="store_true",
                    dest="debug",
                    default=False,
                    help='Print debugging information' )

    parser.add_argument('-D',
                    dest="D",
                    default=50,
                    type=float,
                    help='Knot Distance' )

    parser.add_argument('--log', 
                    dest="log",
                    default=None,
                    help='Otput tensorboard log into this directory' )
    
   
    parser.add_argument('--subsample',
                        default=0.01,
                        type=float,
                        help='randomly subsample training data')
    
    parser.add_argument('--bw',
                        default=None,
                        type=float,
                        help='Kernel bandwidth')
    
    parser.add_argument('--bins',
                        default=20,
                        type=int,
                        help='Histogram bins')
    
    parser.add_argument('--threads',
                        dest="threads",
                        default=None,
                        type=int,
                        help='Max number of threads')
    
    #parser.add_argument('--product',
                        #dest='product',
                        #default=False,
                        #action="store_true",
                        #help='Use product as additional feature')

    parser.add_argument('--save',help='Save training results in a file')
    parser.add_argument('--load',help='Load training results from a file')
    
    options = parser.parse_args()
    
    return options

def tf_gauss_kernel_histogram(hist_bins,values,bw):
    with tf.name_scope('histogram') as scope:
        r=tf.reduce_sum( tf.exp(tf.square( 
        tf.sub( tf.expand_dims( hist_bins, 1), tf.expand_dims( values, 0) ) )
            *(-1.0/(bw*bw)))/np.sqrt(np.pi),1)/tf.to_float(tf.size(values))
    # gaussian kernel function
        return r


if __name__ == "__main__":
    history=minc.format_history(sys.argv)
    
    options = parse_options()
    
    if options.threads is not None:
        tf_config=tf.ConfigProto(inter_op_parallelism_threads=options.threads,
                             intra_op_parallelism_threads=options.threads)
    else:
        tf_config=tf.ConfigProto()
        
    # load prior and input image
    if (options.ref is not None or options.load is not None) and options.image is not None:
        if options.debug: print("Loading images...")
        # convert to float as we go
        image=minc.Image(options.image).data.astype(np.float32)
        
        if options.debug: print("Done")
        
        clf=None
        
        if options.load is not None:
            #TODO: load classifications 
            pass
        else:
            ref_image=minc.Image(options.ref).data.astype(np.float32)
            rmm=(ref_image>0)
            mm=(image>0)

            if options.mask is not None:
                mask  = minc.Label(options.mask)
                mm = np.logical_and(image>0 , mask.data>0 )
                
            if options.refmask is not None:
                refmask  = minc.Label(options.refmask)
                rmm = np.logical_and(ref_image>0 , refmask.data>0 )

            rmin=np.amin(ref_image[rmm])
            rmax=np.amax(ref_image[rmm])
            print("Ref Range {} - {}".format(rmin,rmax))
            imin=np.amin(image[mm])
            imax=np.amax(image[mm])
            print("Image Range {} - {}".format(imin,imax))
            
            if options.bw is not None:
                bw=options.bw
            else:
                bw=(rmax-rmin)/options.bins*2.0
            
            # add features dependant on coordinates
            c=np.mgrid[0:image.shape[0] , 
                       0:image.shape[1] , 
                       0:image.shape[2]].astype(np.float32)
            
            c=np.column_stack(( np.ravel(c[0][mm] ), np.ravel(c[1][mm] ),np.ravel(c[2][mm] )))
            # TODO: use step size here to convert to physical units?
            
            knots_x=np.floor( image.shape[0]/options.D )+1
            knots_y=np.floor( image.shape[1]/options.D )+1
            knots_z=np.floor( image.shape[2]/options.D )+1
            
            num_basis=knots_z*knots_y*knots_x
            
            spatial_bw=options.D*2.0
            
            # knots coordinates - uniformly sampling space
            knots=(np.mgrid[ 0:knots_x,
                            0:knots_y, 
                            0:knots_z ].astype(np.float32))*options.D+options.D/2
            
            knots = np.column_stack(( np.ravel(knots[0] ),
                                      np.ravel(knots[1] ),
                                      np.ravel(knots[2] ) ))
            
            # initial coeffecients for normalization
            init_coeff=np.ones([num_basis]).astype(np.float32)/(spatial_bw*np.sqrt(2*np.pi))
            
            #basis=np.column_stack( tuple( j for j in _basis  ) )
            training_X = np.ravel( image[ mm ]  ) 
            training_Y = np.ravel( ref_image[ rmm ] )
            # 
            if options.debug: 
              print("Fitting...")
              print("Number of unknowns:{}".format(num_basis))
            
            x        = tf.placeholder("float32", [None] )
            y        = tf.placeholder("float32", [None] )
            #basis_   = tf.placeholder("float32", [None, num_basis] )
            coord_   = tf.placeholder("float32", [None, 3] )
            
            # this is the only trainable variable
            coeff    = tf.Variable(init_coeff,   name="basis_coeff", trainable=True)
            
            
            # normalization field
            # calculate EQ^2 distance from every coord to every knot
            # then sum over them
            dist_sq_element = tf.square( tf.sub( tf.expand_dims(coord_,1), tf.expand_dims(knots,0)))
            dist_sq = tf.reduce_sum( dist_sq_element, 2)
            kernels=tf.exp((-1.0/(spatial_bw*spatial_bw)) * dist_sq )
            
            normalization = tf.reduce_sum( tf.expand_dims(coeff, 0 ) * kernels, 1 , name='Normalization_Field')
            
            with tf.name_scope('correct') as scope:
                hist_bins  = tf.linspace(rmin,rmax,options.bins, name="hist_bins")
                corr_x     = tf.mul(x,normalization,name='Corrected_Image') #tf.exp( tf.add( x, normalization ) )
                # calculate standard deviations inside each class
                
                hist_ref  = tf.clip_by_value(tf_gauss_kernel_histogram(hist_bins,y,bw),1e-10,1.0,name='Ref_histogram')
                hist_corr = tf.clip_by_value(tf_gauss_kernel_histogram(hist_bins,corr_x,bw),1e-10,1.0,name='Corr_histogram')
                
                # K-L divergence
                loss      = tf.reduce_sum(hist_ref * tf.log(  tf.clip_by_value(hist_ref/hist_corr,1e-10,1e10) ),name='K-L_divergence')

                global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = 0.1
                learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                        300, 0.96, staircase=True)

                #opt  = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                opt  = tf.train.AdagradOptimizer(learning_rate=learning_rate)
                #opt = tf.train.AdamOptimizer()
                train_step = opt.minimize(loss,var_list=[coeff],global_step=global_step)# ,[mean,sigma]
                
                if options.log is not None:
                    coeff_hist = tf.histogram_summary("coeff", coeff)
                    loss_summary = tf.scalar_summary("loss", loss)
                    learning_rate_summary = tf.scalar_summary("learning_rate",learning_rate)

            if options.log is not None:
                summary_op = tf.merge_all_summaries()
                
            init = tf.initialize_all_variables()
            sess = tf.Session(config=tf_config)
            
            if options.log is not None:
                print("Writing log to {}".format(options.log))
                writer = tf.train.SummaryWriter(options.log, sess.graph_def)
                
            sess.run(init)

            t0 = time.time()
            #sub_iters=100
            epochs=options.iter/options.sub_iter
            training_X_=training_X
            training_Y_=training_Y
            num_datasets=training_X.shape[0]
            r_num_datasets=training_Y.shape[0]
            
            
            for step in xrange(0, epochs):
                # get a new random subsample if needed 
                if options.subsample<1.0:
                    subset=np.random.choice(num_datasets, size=    int(num_datasets*options.subsample), replace=False)
                    subset_r=np.random.choice(r_num_datasets, size=int(r_num_datasets*options.subsample), replace=False)
                else:
                    subset=np.random.choice(num_datasets, size=    int(options.subsample), replace=False)
                    subset_r=np.random.choice(r_num_datasets, size=int(options.subsample), replace=False)
                training_Y_=training_Y[subset_r]
                training_X_=training_X[subset]
                
                coord=c[subset,:]
                
                for sstep in xrange(options.sub_iter):
                    if options.log is not None:
                        (dummy,summary_str,_loss)= \
                            sess.run([train_step, summary_op, loss], 
                                feed_dict={x: training_X_, 
                                           y: training_Y_,
                                           coord_: coord},
                                    )
                        writer.add_summary(summary_str,step*options.sub_iter+sstep)
                    else:
                        (dummy,_loss)= \
                            sess.run([train_step, loss], 
                                feed_dict={x: training_X_, 
                                           y: training_Y_,
                                           coord_: coord} )
                    print("{}".format(_loss))
                print("{} - {} ".format(step*options.sub_iter,_loss))
            
            t1 = time.time()
            print("Elapsed time={}".format(t1-t0))
            
            final_coeff= sess.run(coeff)
            #print("final loss=",final_loss)
            print("Final coeffecients=",final_coeff)
        
        
            if options.output is not None:
                if options.debug: print("Saving...")
                
                downsampled=np.array(image.shape)
                #
                print("Generating correction field:")
                #
                _normalization=np.zeros_like(image)
                #
                for i in range(downsampled[2]):
                    c=np.mgrid[ 0:downsampled[0] ,
                                0:downsampled[1] ,
                                i:(i+1)].astype(np.float32)
                    
                    c=np.column_stack( ( np.ravel(c[0] ),np.ravel(c[1] ),np.ravel(c[2] )))
                    
                    _normalization[:,:,i]=sess.run(
                        normalization, feed_dict={coord_: c} 
                        ).reshape((downsampled[0],downsampled[1]))
                # 
                print("Done")
                # TODO: interpolate here?
                
                #print(cx.shape,cy.shape,cz.shape,image.shape,final_coeff.shape)
                
                #options.M=5 # number of basis functions
                # for now initialize in numpy
                # TODO: maybe move to tensor-flow 
                __i=0
                
                #_normalization=np.exp(_normalization)
                out=minc.Image(data=_normalization.astype(np.float64))
                #out=minc.Image(data=out_cls.astype(np.float64))
                out.save(name=options.output, imitate=options.image, history=history)
                
    else:
        print "Error in arguments"

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on
