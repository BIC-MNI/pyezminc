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
    
    parser.add_argument('-O', 
                    dest="O",
                    default=3,
                    type=int,
                    help='Order' )
    
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
    
    parser.add_argument('--bw',
                        default=None,
                        type=float,
                        help='Kernel bandwidth')
    
    parser.add_argument('--bins',
                        default=20,
                        type=int,
                        help='Histogram bins')

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
    # gaussian kernel function
    return tf.reduce_sum( tf.exp(tf.square( 
        tf.sub( tf.expand_dims( hist_bins, 1), tf.expand_dims( values, 0) ) )
     *(-1.0/(bw*bw)))/np.sqrt(np.pi),1)/tf.to_float(tf.size(values))

if __name__ == "__main__":
    history=minc.format_history(sys.argv)
    
    options = parse_options()
    # load prior and input image
    if (options.ref is not None or options.load is not None) and options.image is not None:
        if options.debug: print("Loading images...")
        # convert to float as we go
        
        image=minc.Image(options.image).data.astype(np.float32)
        
        nvoxels=image.shape[0]*image.shape[1]*image.shape[2]
       
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
                bw=(rmax-rmin)/options.bins*2.0*np.sqrt(2.0*np.log(2.0))

            
            # add features dependant on coordinates
            c=np.mgrid[0:image.shape[0] , 0:image.shape[1] , 0:image.shape[2]].astype(np.float32)
            
            # normalized spatial coordinates
            _extents=max(image.shape)

            cx= np.ravel(c[0][mm]/_extents )
            cy= np.ravel(c[1][mm]/_extents )
            cz= np.ravel(c[2][mm]/_extents )
            
            # generate basis functions
            #print(cx)
            
            #_basis=[]
            
            ##options.M=5 # number of basis functions
            ## for now initialize in numpy
            ## TODO: maybe move to tensor-flow 
            #for i in range(options.M):
                #cosx=np.cos(np.pi*(cx+0.5)*i/options.M)
                #for j in range(options.M):
                    #cosy=np.cos(np.pi*(cy+0.5)*j/options.M)
                    #cosxy=np.multiply(cosx,cosy)
                    #for k in range(options.M):
                        #cosz=np.cos(np.pi*(cz+0.5)*k/options.M)
                        #_basis.append(np.multiply(cosxy,cosz))
                        ##if options.debug: print("{} {} {}".format(i,j,k))
                        
            #num_basis=len(_basis)

            # initial coeffecients for normalization:
            coeff_count=1
            
            if options.O>=1:coeff_count+=3
            if options.O>=2:coeff_count+=6
            if options.O>=3:coeff_count+=10
            if options.O>=4:coeff_count+=15
                            
            init_coeff=np.zeros([coeff_count]).astype(np.float32)
            init_coeff[0]=1.0
            
            #basis=np.column_stack( tuple( j for j in _basis  ) )
            training_X = np.ravel( image[ mm ]  ) 
            training_Y = np.ravel( ref_image[ rmm ]  )
            # 
            if options.debug: 
              print("Fitting...")
              print("Number of unknowns:{}".format(coeff_count))
            
            x        = tf.placeholder("float32", [None] )
            y        = tf.placeholder("float32", [None] )
            
            cx_      = tf.placeholder("float32", [None] )
            cy_      = tf.placeholder("float32", [None] )
            cz_      = tf.placeholder("float32", [None] )
            
            hist_bins     = tf.linspace(rmin,rmax,options.bins, "hist_bins")
            coeff    = tf.Variable(init_coeff , name="basis_coeff")
            tf_alpha = tf.constant(options.alpha,name='alpha')
            
            # normalization field, normalized to have unit sum
            
            nvoxels_ = tf.constant(nvoxels,dtype='float32')
            basis_   = [tf.ones_like(cx_)]
            integral_= [tf.constant(1.0)]
            
            if options.O >= 1:
                basis_.extend([cx_, cy_, cz_])
                integral_.extend([tf.constant(0.5), tf.constant(0.5), tf.constant(0.5)])
            
            if options.O >= 2:
                basis_.extend([ cx_*cx_, cy_*cy_, cz_*cz_, 
                                cx_*cy_, cx_*cz_, cy_*cz_])
                integral_.extend(
                    [   tf.constant(1/3.0)  ,tf.constant(1/3.0  ),tf.constant(1/3.0  ),
                        tf.constant(0.5*0.5),tf.constant(0.5*0.5),tf.constant(0.5*0.5)]
                    )
                    
            if options.O >= 3:
                basis_.extend( [ cx_*cx_*cx_, cy_*cy_*cy_, cz_*cz_*cz_, 
                                 cx_*cx_*cy_, cx_*cx_*cz_,
                                 cy_*cy_*cx_, cy_*cy_*cz_,
                                 cz_*cz_*cy_, cz_*cz_*cx_,
                                 cx_*cy_*cz_
                                ] )
                integral_.extend(
                    [   tf.constant(0.25),   tf.constant(0.25), tf.constant(0.25),
                        tf.constant(0.5/3.0),tf.constant(0.5/3.0),
                        tf.constant(0.5/3.0),tf.constant(0.5/3.0),
                        tf.constant(0.5/3.0),tf.constant(0.5/3.0),
                        tf.constant(0.5*0.5*0.5)]
                    )
            
            if options.O >= 4:
                basis_.extend( [ cx_*cx_*cx_*cx_, cy_*cy_*cy_*cy_, cz_*cz_*cz_*cz_, 
                                
                                cx_*cx_*cx_*cy_, cx_*cx_*cx_*cz_,
                                cy_*cy_*cy_*cx_, cy_*cy_*cy_*cz_,
                                cz_*cz_*cz_*cx_, cz_*cz_*cz_*cx_,
                                
                                cx_*cx_*cy_*cy_, cx_*cx_*cz_*cz_, cy_*cy_*cz_*cz_,
                                cx_*cx_*cy_*cz_, cy_*cy_*cx_*cz_, cz_*cz_*cx_*cy_
                                ] )
                integral_.extend(
                    [   tf.constant(0.2),     tf.constant(0.2),     tf.constant(0.2),
                        tf.constant(0.25*0.5),tf.constant(0.25*0.5),
                        tf.constant(0.25*0.5),tf.constant(0.25*0.5),
                        tf.constant(0.25*0.5),tf.constant(0.25*0.5),

                        tf.constant(1.0/9.0),tf.constant(1.0/9.0),tf.constant(1.0/9.0),
                        tf.constant(0.5*0.5/3.0),tf.constant(0.5*0.5/3.0),tf.constant(0.5*0.5/3.0)
                    ]
                    )

            basis     = tf.concat(1, [tf.expand_dims(b,1)  for b in basis_])
            #integral  = tf.concat(0, [tf.expand_dims(b,0)  for b in integral_])
            
            #nrm  = tf.reduce_sum( tf.mul( integral, coeff ) ) #/ nvoxels_
            
            # make sure it's all normalized to 1
            #normalization = tf.sigmoid(tf.reduce_sum( tf.mul( basis, coeff),1 ))*2.0
            normalization = tf.reduce_sum( tf.mul( basis, coeff),1 ) # /nrm
            
            corr_x = tf.mul( x, normalization )
            # calculate standard deviations inside each class
            
            hist_ref  = tf.clip_by_value(tf_gauss_kernel_histogram(hist_bins,y,bw),1e-10,1.0)
            hist_corr = tf.clip_by_value(tf_gauss_kernel_histogram(hist_bins,corr_x,bw),1e-10,1.0)
            
            # K-L divergence
            loss      = tf.reduce_sum(hist_ref * tf.log(  tf.clip_by_value(hist_ref/hist_corr,1e-10,1e10) ))

            #opt  = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            opt  = tf.train.AdagradOptimizer(learning_rate=0.1)
            #opt = tf.train.AdamOptimizer()
            train_step = opt.minimize(loss)# ,[mean,sigma]
            
            if options.log is not None:
                coeff_hist = tf.histogram_summary("coeff", coeff)
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
            #sub_iters=100
            epochs=options.iter/options.sub_iter
            
            training_X_=training_X
            training_Y_=training_Y
            cx_subset=cx
            cy_subset=cy
            cz_subset=cz
            num_datasets=training_X.shape[0]
            r_num_datasets=training_Y.shape[0]

            #(_loss,_means)= \
                #sess.run([loss, means],  
                    #feed_dict={ x: training_X, 
                                #y_: training_Y, 
                                #cx_: cx_subset,
                                #cy_: cy_subset,
                                #cz_: cz_subset} )
#            print("Initial loss:{} means:{}".format(_loss,_means))
                  
            for step in xrange(0, epochs):
                # get a new random subsample if needed 
                subset=np.random.choice(num_datasets, size=num_datasets*options.subsample, replace=False)
                subset_r=np.random.choice(r_num_datasets, size=r_num_datasets*options.subsample, replace=False)
                
                training_Y_=training_Y[subset_r]
                training_X_=training_X[subset]
                cx_subset=cx[subset]
                cy_subset=cy[subset]
                cz_subset=cz[subset]
                
                for sstep in xrange(options.sub_iter):
                    if options.log is not None:
                        (dummy,summary_str,_loss)= \
                            sess.run([train_step, summary_op, loss], 
                                feed_dict={x: training_X_, 
                                           y: training_Y_,
                                           cx_: cx_subset,
                                           cy_: cy_subset,
                                           cz_: cz_subset},
                                    )
                        writer.add_summary(summary_str, step*options.sub_iter+sstep)
                    else:
                        (dummy,_loss)= \
                            sess.run([train_step, loss],  
                                feed_dict={x: training_X_, 
                                           y: training_Y_,
                                           cx_: cx_subset,
                                           cy_: cy_subset,
                                           cz_: cz_subset} )
                print("{} - {} ".format((step+1)*options.sub_iter,_loss))
            
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
                
                c=np.mgrid[0:image.shape[0] , 0:image.shape[1] , 0:image.shape[2]].astype(np.float32)
                
                cx= np.ravel(c[0]/ _extents  )
                cy= np.ravel(c[1]/ _extents  )
                cz= np.ravel(c[2]/ _extents  )
                
                __i=0
                print("Generating correction field:")
                _normalization=sess.run(normalization,
                                      feed_dict={cx_: cx,cy_: cy, cz_: cz})
                #
                # TODO: reshape to match minc
                #_normalization/=np.sum(final_coeff)
                out=minc.Image(data=np.reshape(_normalization.astype(np.float64),image.shape))
                #out=minc.Image(data=out_cls.astype(np.float64))
                out.save(name=options.output, imitate=options.image,history=history)
                
    else:
        print "Error in arguments"

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on
