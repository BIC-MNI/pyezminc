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


#include <volume_io.h>

/*these functions are defined in minc2-matrix-ops*/
/* ----------------------------- MNI Header -----------------------------------
@NAME       : build_transformation_matrix
@INPUT      : center, translations, scales, rotations
@OUTPUT     : *lt->mat - a linear transformation matrix
@RETURNS    : nothing
@DESCRIPTION: mat = (T)(C)(S)(SH)(R)(-C)
               the matrix is to be  PREmultiplied with a column vector (mat*colvec)
               when used in the application
---------------------------------------------------------------------------- */
void build_transformation_matrix(VIO_Transform *trans,
                                  double *center,
                                  double *translations,
                                  double *scales,
                                  double *shears,
                                  double *rotations);

/* extract parameters from linear transform
   trans = [scale][shear][rot]
         = [scale][shear][rz][ry][rx];
                                  */
VIO_BOOL extract2_parameters_from_matrix(VIO_Transform *trans,
                                         double *center,
                                         double *translations,
                                         double *scales,
                                         double *shears,
                                         double *rotations);



int matrix_extract_linear_param( double *in_matrix,
                                 double *center,
                                 double *translations,
                                 double *scales,
                                 double *shears,
                                 double *rotations)
{
  int i,j;
  VIO_Transform lin;

  /* convert 4D raw matrix into MINC world */
  for(i=0;i<4;i++) {
    for(j=0;j<4;j++) {
        lin.m[j][i]=in_matrix[i*4+j];
      }}

    return extract2_parameters_from_matrix(&lin,
                                    center,
                                    translations,
                                    scales,
                                    shears,rotations)?0:-1;
}


int linear_param_to_matrix(double *out_matrix,
                           double *center,
                           double *translations,
                           double *scales,
                           double *shears,
                           double *rotations)
{
  int i,j;
  VIO_Transform lin;
  memset(&lin, 0, sizeof(VIO_Transform));

  build_transformation_matrix(&lin,
                              center, translations,
                              scales, shears, rotations);

  for(i=0;i<4;i++) {
    for(j=0;j<4;j++) {
        out_matrix[i*4+j]=lin.m[j][i];
        }}

  return 0;
}
