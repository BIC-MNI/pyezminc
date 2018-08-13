/* ----------------------------- MNI Header -----------------------------------
@NAME       : xfm_param.h
@DESCRIPTION: helper headers for easy conversion between 3d transformation parameters and 4D matrix
@METHOD     : Contains routines :
             linear_param_to_matrix
@CALLS      :
@COPYRIGHT  :
              Copyright 2018 Vladimir S. FONOV, McConnell Brain Imaging Centre,
              Montreal Neurological Institute, McGill University.
              Permission to use, copy, modify, and distribute this
              software and its documentation for any purpose and without
              fee is hereby granted, provided that the above copyright
              notice appear in all copies.  The author and McGill University
              make no representations about the suitability of this
              software for any purpose.  It is provided "as is" without
              express or implied warranty.
*/


int matrix_extract_linear_param( double *in_matrix,
                                 double *center,
                                 double *translations,
                                 double *scales,
                                 double *shears,
                                 double *rotations);

int linear_param_to_matrix(double *out_matrix,
                           double *center,
                           double *translations,
                           double *scales,
                           double *shears,
                           double *rotations);
