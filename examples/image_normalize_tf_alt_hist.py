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
    
    parser.add_argument('-M', 
                    dest="M",
                    default=5,
                    type=int,
                    help='Order' )
    
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
    
    parser.add_argument('--sq',
                        default=False,
                        action="store_true",
                        help='Use square kernel')
    
    parser.add_argument('--bins',
                        default=40,
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
                *(-1.0/(2.0*bw*bw)))/(np.sqrt(np.pi*2.0)*bw),1)/tf.to_float(tf.size(values))
    # gaussian kernel function
        return r

def tf_sq_histogram(hist_bins,values,bw):
    with tf.name_scope('histogram') as scope:
        r=tf.reduce_sum( tf.clip_by_value( 1.0-tf.abs(
            tf.sub( tf.expand_dims( hist_bins, 1), tf.expand_dims( values, 0) ) )/bw
                ,0.0,1.0))/tf.to_float(tf.size(values))
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
                if options.sq:
                    bw=(rmax-rmin)/options.bins
                else:
                    bw=(rmax-rmin)*1.6/options.bins
            
            # add features dependant on coordinates
            c=np.mgrid[0:image.shape[0] , 
                       0:image.shape[1] , 
                       0:image.shape[2]].astype(np.float32)
            
            # normalized spatial coordinates
            #_extents=max(image.shape)

            cx= np.ravel(c[0][mm])
            cy= np.ravel(c[1][mm])
            cz= np.ravel(c[2][mm])
            
            #coord=np.column_stack(cx,cy,cz)
            # generate basis functions
            #print(cx)
            
            #_basis=[]
            
            #options.M=5 # number of basis functions per dimension
            # for now initialize in numpy
            # TODO: maybe move to tensor-flow 
            #for i in range(options.M):
                #cosx=np.cos(np.pi*(cx+0.5)*i/image.shape[0])
                #for j in range(options.M):
                    #cosy=np.cos(np.pi*(cy+0.5)*j/image.shape[1])
                    #cosxy=cosx*cosy
                    #for k in range(options.M):
                        #cosz=np.cos(np.pi*(cz+0.5)*k/image.shape[2])
                        #_basis.append(cosxy*cosz)
                        #if options.debug: print("{} {} {}".format(i,j,k))
                        
            num_basis=options.M*options.M*options.M

            # initial coeffecients for normalization
            init_coeff=np.zeros([num_basis]).astype(np.float32)
            init_coeff[0]=1.0
            
            #basis=np.column_stack( tuple( j for j in _basis  ) )
            training_X = np.ravel( image[ mm ]  ) 
            training_Y = np.ravel( ref_image[ rmm ] )
            # 
            if options.debug: 
              print("Fitting...")
              print("Number of unknowns:{}".format(num_basis))
            
            x        = tf.placeholder("float32", [None] )
            y        = tf.placeholder("float32", [None] )
            basis_   = tf.placeholder("float32", [None, num_basis] )
            #coord_   = tf.placeholder("float32", [None, 3] )
            
            # this is the only trainable variable
            coeff    = tf.Variable(init_coeff,   name="basis_coeff", trainable=True)
            
            #args     = (coord_+0.5)*np.pi/shape
            
                
            # normalization field, normalized to have unit sum
            normalization = tf.reduce_sum( tf.mul(coeff, basis_), 1) # - tf.reduce_sum(coeff)
            
            
            with tf.name_scope('correct') as scope:
                hist_bins  = tf.to_float(tf.linspace(rmin-bw,rmax+bw,options.bins+3, name="hist_bins"))
                corr_x     = tf.mul(x,normalization,name='Corrected_Image') #tf.exp( tf.add( x, normalization ) )
                # calculate standard deviations inside each class
                
                if options.sq:
                    hist_ref  = tf.clip_by_value(tf_sq_histogram(hist_bins,y,bw),     1e-10,1.0,name='Ref_histogram')
                    hist_corr = tf.clip_by_value(tf_sq_histogram(hist_bins,corr_x,bw),1e-10,1.0,name='Corr_histogram')
                else:
                    hist_ref  = tf.clip_by_value(tf_gauss_kernel_histogram(hist_bins,y,bw),     1e-10,1.0,name='Ref_histogram')
                    hist_corr = tf.clip_by_value(tf_gauss_kernel_histogram(hist_bins,corr_x,bw),1e-10,1.0,name='Corr_histogram')
                    
                s_ref    = tf.reduce_sum(hist_ref)
                s_corr   = tf.reduce_sum(hist_corr)
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
                
                cx_=cx[subset]
                cy_=cy[subset]
                cz_=cz[subset]

# calculate basis only for the points used to conserve mempory, TODO: move this into TF graph?
                _basis=[]
                
                print("Recalculating basis...")
                for i in range(options.M):
                    cosx=np.cos(np.pi*(cx_+0.5)*i/image.shape[0])
                    for j in range(options.M):
                        cosy=np.cos(np.pi*(cy_+0.5)*j/image.shape[1])
                        cosxy=cosx*cosy
                        for k in range(options.M):
                            cosz=np.cos(np.pi*(cz_+0.5)*k/image.shape[2])
                            _basis.append(cosxy*cosz)
                basis=np.column_stack( tuple( j for j in _basis  ) )
                print("Done")
                
                for sstep in xrange(options.sub_iter):
                    if options.log is not None:
                        (dummy,summary_str,_loss,_s_ref,_s_corr)= \
                            sess.run([train_step, summary_op, loss,s_ref,s_corr], 
                                feed_dict={x: training_X_, 
                                           y: training_Y_,
                                           basis_: basis},
                                    )
                        writer.add_summary(summary_str, step*options.sub_iter+sstep)
                    else:
                        (dummy,_loss,_s_ref,_s_corr)= \
                            sess.run([train_step, loss,s_ref,s_corr], 
                                feed_dict={x: training_X_, 
                                           y: training_Y_,
                                           basis_: basis} )
                    print("{} - {} {}".format(_loss,_s_ref,_s_corr))
                print("{} - {} ".format((step+1)*options.sub_iter,_loss))
            
            t1 = time.time()
            print("Elapsed time={}".format(t1-t0))
            
            final_coeff= sess.run(coeff)
            #print("final loss=",final_loss)
            #print("Final coeffecients=",final_coeff)
        
        
            if options.output is not None:
                if options.debug: print("Saving...")
                
                c=np.mgrid[0:image.shape[0] , 
                           0:image.shape[1] , 
                           0:image.shape[2]].astype(np.float32)
                
                cx= c[0]
                cy= c[1]
                cz= c[2]
                
                _normalization=np.zeros_like(image)
                #print(cx.shape,cy.shape,cz.shape,image.shape,final_coeff.shape)
                
                #options.M=5 # number of basis functions
                # for now initialize in numpy
                # TODO: maybe move to tensor-flow 
                __i=0
                print("Generating correction field:")
                for i in range(options.M):
                    cosx=np.cos(np.pi*(cx+0.5)*i/image.shape[0])
                    print("{}%".format(i*100/options.M))
                    for j in range(options.M):
                        cosxy=cosx*np.cos(np.pi*(cy+0.5)*j/image.shape[1])
                        for k in range(options.M):
                            cosz=np.cos(np.pi*(cz+0.5)*k/image.shape[2])
                            _normalization+=cosz*cosxy*final_coeff[__i]
                            __i+=1
                #
                #_normalization=np.exp(_normalization)
                out=minc.Image(data=_normalization.astype(np.float64))
                #out=minc.Image(data=out_cls.astype(np.float64))
                out.save(name=options.output, imitate=options.image,history=history)
                
    else:
        print "Error in arguments"

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on
