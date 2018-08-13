/* ----------------------------- MNI Header -----------------------------------
@NAME       : minc2-matrix-ops.c
@DESCRIPTION: File containing routines for doing basic matrix calculations, based on 
              mni_autoreg/minctracc/Numerical/make_rots.c
              mni_autoreg/minctracc/Numerical/matrix_basics.c
              mni_autoreg/minctracc/Numerical/rotmat_to_ang.c
              mni_autoreg/minctracc/Numerical/Numerical/quaternion.c
@METHOD     : Contains routines :
                 printmatrix
                 calc_centroid
                 translate
                 transpose
                 matrix_multiply
                 trace
                 matrix_scalar_multiply
                 invertmatrix
@CALLS      : 
@COPYRIGHT  :
              Copyright 1993 Peter Neelin, McConnell Brain Imaging Centre, 
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


#define  FILL_NR_COLVEC( vector, x, y, z ) \
            { \
                vector[1][1] = (x); \
                vector[2][1] = (y); \
                vector[3][1] = (z); \
                vector[4][1] = 1.0; \
            }

#define  ADD_NR_COLVEC( vector, v1, v2 ) \
            { \
                vector[1][1] = v1[1][1] + v2[1][1]; \
                vector[2][1] = v1[2][1] + v2[2][1]; \
                vector[3][1] = v1[3][1] + v2[3][1]; \
                vector[4][1] = 1.0; \
            }

#define  SUB_NR_COLVEC( vector, v1, v2 ) \
            { \
                vector[1][1] = v1[1][1] - v2[1][1]; \
                vector[2][1] = v1[2][1] - v2[2][1]; \
                vector[3][1] = v1[3][1] - v2[3][1]; \
                vector[4][1] = 1.0; \
            }

#define  DOT_NR_COLVEC( vector, v1, v2 ) \
            { \
                vector[1][1] = v1[1][1]*v2[1][1]; \
                vector[2][1] = v1[2][1]*v2[2][1]; \
                vector[3][1] = v1[3][1]*v2[3][1]; \
                vector[4][1] = 1.0; \
            }

#define  SCALAR_MULT_NR_COLVEC( vector, v1, sc ) \
            { \
                vector[1][1] = v1[1][1]*sc; \
                vector[2][1] = v1[2][1]*sc; \
                vector[3][1] = v1[3][1]*sc; \
                vector[4][1] = 1.0; \
            }

#define  DOTSUM_NR_COLVEC( v1, v2 ) \
                (v1[1][1] * v2[1][1] + \
                v1[2][1] * v2[2][1] + \
                v1[3][1] * v2[3][1]) 

#define  MAG_NR_COLVEC( v1 ) \
            ( sqrt( v1[1][1] * v1[1][1] + \
                v1[2][1] * v1[2][1] + \
                v1[3][1] * v1[3][1] ) )


            
void extract_quaternions(float **m, double *quat);
void build_rotmatrix(float **m, double *quat);

            
void make_rots(float **xmat, float rot_x, float rot_y, float rot_z);

VIO_BOOL rotmat_to_ang(float **rot, float *ang);


void build_rotmatrix(float **m, double *quat);
void extract_quaternions(float **m, double *quat);


void printmatrix(int rows, int cols, float **the_matrix);

void calc_centroid(int npoints, int ndim, float **points, 
                          float *centroid);

void translate(int npoints, int ndim, float **points, 
                      float *translation, float **newpoints);

void transpose(int rows, int cols, float **mat, float **mat_transpose);

void invertmatrix(int n, float **mat, float **mat_invert);

void raw_matrix_multiply(int ldim, int mdim, int ndim, 
                                float **Amat, float **Bmat, float **Cmat);

void matrix_multiply(int ldim, int mdim, int ndim, 
                            float **Amat, float **Bmat, float **Cmat);

float trace(int size, float **the_matrix);

void matrix_scalar_multiply(int rows, int cols, float scalar, 
                            float **the_matrix, float **product);

void nr_identd(double **A, int m1, int m2, int n1, int n2 );
void nr_identf(float **A, int m1, int m2, int n1, int n2 );

void nr_copyd(double **A, int m1, int m2, int n1, int n2, double **B );
void nr_copyf(float  **A, int m1, int m2, int n1, int n2, float **B );

void nr_rotxd(double **M, double a);
void nr_rotxf(float **M, float a);

void nr_rotyd(double **M,double a);
void nr_rotyf(float **M, float a);

void nr_rotzd(double **M,double a);
void nr_rotzf(float **M, float a);

void nr_multd(double **A, int mA1, int mA2, int nA1, int nA2,
                     double **B, int mB1, int mB2, int nB1, int nB2, 
                     double **C);
void nr_multf(float **A, int mA1, int mA2, int nA1, int nA2,
                     float **B, int mB1, int mB2, int nB1, int nB2, 
                     float **C);


void transformations_to_homogeneous(int ndim, 
                  float *translation, float *centre_of_rotation,
                  float **rotation, float scale,
                  float **transformation);

void translation_to_homogeneous(int ndim, float *translation,
                                       float **transformation);

void rotation_to_homogeneous(int ndim, float **rotation,
                                       float **transformation);



/* ----------------------------- MNI Header -----------------------------------
@NAME       : printmatrix
@INPUT      : rows   - number of rows in matrix
              cols   - number of columns in matrix
              the_matrix - matrix to be printed (in zero offset form).
                 The dimensions of this matrix should be defined to be 
                 1 to rows and 1 to cols.
@OUTPUT     : (nothing)
@RETURNS    : (nothing)
@DESCRIPTION: Prints out a matrix on stdout with one row per line.
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : Feb. 26, 1990 (Weiqian Dai)
@MODIFIED   : January 31, 1992 (Peter Neelin)
                 - change to roughly NIL-abiding code
---------------------------------------------------------------------------- */
void printmatrix(int rows, int cols, float **the_matrix)
{
   int i,j;
   float f;

   /* Loop through rows and columns, printing one row per line */
   for (i=1; i <= rows; ++i) {
      for (j=1; j <= cols; ++j) {
         f=the_matrix[i][j];
         (void) print(" %10.6f ",f);
      }
      (void) print("\n");
   }
}

/* ----------------------------- MNI Header -----------------------------------
@NAME       : calc_centroid
@INPUT      : npoints - number of points
              ndim    - number of dimensions
              points  - points matrix (in zero offset form).
                 The dimensions of this matrix should be defined to be 
                 1 to npoints and 1 to ndim.
@OUTPUT     : centroid - vector of centroid of points (in num. rec. form)
                 This vector should run from 1 to ndim.
@RETURNS    : (nothing)
@DESCRIPTION: Calculates the centroid of a number of points in ndim dimensions.
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : Feb. 26, 1990 (Weiqian Dai)
@MODIFIED   : January 31, 1992 (Peter Neelin)
                 - change to roughly NIL-abiding code and modified calling
                 sequence.
---------------------------------------------------------------------------- */
void calc_centroid(int npoints, int ndim, float **points, 
                          float *centroid)
{
   int i,j;

   /* Loop over each dimension */
   for (i=1; i <= ndim; ++i) {
      /* Add up points and divide by number of points */
      centroid[i] = 0;
      for (j=1; j <= npoints; ++j) {
         centroid[i] += points[j][i];
      }
      if (npoints >0) centroid[i] /= (float) npoints;
   }
}

/* ----------------------------- MNI Header -----------------------------------
@NAME       : translate
@INPUT      : npoints - number of points
              ndim    - number of dimensions
              points  - points matrix (in zero offset form).
                 The dimensions of this matrix should be defined to be 
                 1 to npoints and 1 to ndim.
              translation - translation vector (in num. rec. form, running
                 from 1 to ndim).
@OUTPUT     : newpoints - translated points matrix (see points). This matrix
                 can be the original points matrix.
@RETURNS    : (nothing)
@DESCRIPTION: Translates a set of points by a given translation.
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : Feb. 26, 1990 (Weiqian Dai)
@MODIFIED   : January 31, 1992 (Peter Neelin)
                 - change to roughly NIL-abiding code and modified calling
                 sequence.
---------------------------------------------------------------------------- */
void translate(int npoints, int ndim, float **points, 
                      float *translation, float **newpoints)
{
   int i,j;

   for (i=1; i <= npoints; ++i) {
      for (j=1; j <= ndim; ++j) {
         newpoints[i][j] = points[i][j] + translation[j];
      }
   }
}

/* ----------------------------- MNI Header -----------------------------------
@NAME       : transpose
@INPUT      : rows    - number of rows
              cols    - number of columns
              mat     - original matrix (in zero offset form).
                 The dimensions of this matrix should be defined to be 
                 1 to rows and 1 to cols.
@OUTPUT     : mat_transpose  - transposed matrix (in zero offset form,
                 with dimensions 1 to cols and 1 to rows). 
@RETURNS    : (nothing)
@DESCRIPTION: Transposes a matrix.
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : Feb. 26, 1990 (Weiqian Dai)
@MODIFIED   : January 31, 1992 (Peter Neelin)
                 - change to roughly NIL-abiding code and modified calling
                 sequence.
Fri Jun  4 14:10:34 EST 1993 LC
    added the possibility to have input and out matrices the same!
---------------------------------------------------------------------------- */
void transpose(int rows, int cols, float **mat, float **mat_transpose)
{
   int i,j;

   float **Ctemp;

   if (mat==mat_transpose) {              /* if input and output the same, then alloc
                                         temporary space, so as not to overwrite
                                         the input before the compete transpose is 
                                         done. */
     /* Allocate a temporary matrix */
     VIO_ALLOC2D(Ctemp,cols+1,rows+1);
     
     for (i=1; i <= rows; ++i) {
       for (j=1; j <= cols; ++j) {
         Ctemp[j][i]=mat[i][j];
       }
     }
     
     /* Copy the result */
     for (i=1; i <= cols; ++i)
       for (j=1; j <= rows; ++j)
         mat_transpose[i][j] = Ctemp[i][j];
     
     /* Free the matrix */
     VIO_FREE2D(Ctemp);
   }
   else {
     for (i=1; i <= rows; ++i) {
       for (j=1; j <= cols; ++j) {
         mat_transpose[j][i]=mat[i][j];
       }
     }
   }
}

/* ----------------------------- MNI Header -----------------------------------
@NAME       : invertmatrix
@INPUT      : n       - number of rows/or columns (must be square)
              mat     - original matrix (in zero offset form).
                 The dimensions of this matrix should be defined to be 
                 1 to n rows and 1 to n cols.
@OUTPUT     : mat_invert  - the inverted  matrix (in zero offset form,
                 with dimensions 1 to n cols and 1 to n rows). 
@RETURNS    : (nothing)
@DESCRIPTION: Inverts a matrix.
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : Fri Jun  4 14:10:34 EST 1993 Louis Collins
@MODIFIED   : 
---------------------------------------------------------------------------- */
void raw_invertmatrix(int n, float **mat, float **mat_invert)
{

  int 
    i,j;
  VIO_Real 
    **Rmat, **Rinv;

  VIO_ALLOC2D( Rmat, n, n );
  VIO_ALLOC2D( Rinv, n, n );

  for (i=1; i<=n; ++i)                /* copy the input matrix */
    for (j=1; j<=n; ++j) {
      Rmat[i-1][j-1] = mat[i][j];
    }

  (void)invert_square_matrix(n, Rmat, Rinv);

  for (i=1; i<=n; ++i)                /* copy the result */
    for (j=1; j<=n; ++j) {
      mat_invert[i][j] = Rinv[i-1][j-1];
    }

  VIO_FREE2D( Rmat );
  VIO_FREE2D( Rinv );

/*
                               this is the old inversion code
  float 
    d, **u, *col;
  int 
    i,j,*indx;


  u=mat rix(1,n,1,n);
  col=vec tor(1,n);
  indx=ivec tor(1,n);

  for (i=1; i<=n; ++i)                / * copy the input matrix * /
    for (j=1; j<=n; ++j)
      u[i][j] = mat[i][j];

  lud cmp(u,n,indx,&d);
  for(j=1; j<=n; ++j) {
    for(i=1; i<=n; ++i) col[i] = 0.0;
    col[j]=1.0;
    lub ksb(u,n,indx,col);
    for(i=1; i<=n; ++i) mat_invert[i][j]=col[i];
  }

  free_ matrix(u,1,n,1,n);
  free_ vector(col,1,n);
  free_ ivector(indx,1,n);

*/

}

void invertmatrix(int ndim, float **mat, float **mat_invert)
{
  float **Ctemp;
  int i,j;

  if (mat==mat_invert) {              /* if input and output the same, then alloc
                                         temporary space, so as not to overwrite
                                         the input as the inverse is being done. */
    /* Allocate a temporary matrix */
    VIO_ALLOC2D(Ctemp,ndim+1,ndim+1);
    
    /* invert the matrix */
    raw_invertmatrix(ndim, mat, Ctemp);
    
    /* Copy the result */
    for (i=1; i <= ndim; ++i)
      for (j=1; j <= ndim; ++j)
        mat_invert[i][j] = Ctemp[i][j];
    
    /* Free the matrix */
    VIO_FREE2D(Ctemp);
  }
  else {
    raw_invertmatrix(ndim, mat, mat_invert);
  }
}

/* ----------------------------- mni Header -----------------------------------
@NAME       : raw_matrix_multiply
@INPUT      : ldim, mdim, ndim - dimensions of matrices. Matrix Amat has
                 dimensions (ldim x mdim), matrix Bmat has dimension
                 (mdim x ndim) and resultant matrix has dimension
                 (ldim x ndim).
              Amat - First matrix of multiply (in zero offset form).
                 Dimensions are 1 to ldim and 1 to mdim.
              Bmat - Second matrix of multiply (in zero offset form).
                 Dimensions are 1 to mdim and 1 to ndim.
@OUTPUT     : Cmat - Resulting matrix (in zero offset form).
                 Dimensions are 1 to ldim and 1 to ndim. This matrix cannot
                 be either Amat or Bmat.
@RETURNS    : (nothing)
@DESCRIPTION: Multiplies two matrices.
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : Feb. 26, 1990 (Weiqian Dai)
@MODIFIED   : January 31, 1992 (Peter Neelin)
                 - change to roughly NIL-abiding code and modified calling
                 sequence.
---------------------------------------------------------------------------- */
void raw_matrix_multiply(int ldim, int mdim, int ndim, 
                                float **Amat, float **Bmat, float **Cmat)
{
   int i,j,k;

   /* Zero the output matrix */
   for (i=1; i <= ldim; ++i)
      for (j=1; j <= ndim; ++j)
         Cmat[i][j]=0.;

   /* Calculate the product */
   for (i=1; i <= ldim; ++i)
      for (j=1; j <= ndim; ++j)
         for (k=1; k <=mdim; ++k)
            Cmat[i][j] += (Amat[i][k] * Bmat[k][j]);
}

/* ----------------------------- MNI Header -----------------------------------
@NAME       : matrix_multiply
@INPUT      : ldim, mdim, ndim - dimensions of matrices. Matrix Amat has
                 dimensions (ldim x mdim), matrix Bmat has dimension
                 (mdim x ndim) and resultant matrix has dimension
                 (ldim x ndim).
              Amat - First matrix of multiply (in zero offset form).
                 Dimensions are 1 to ldim and 1 to mdim.
              Bmat - Second matrix of multiply (in zero offset form).
                 Dimensions are 1 to mdim and 1 to ndim.
@OUTPUT     : Cmat - Resulting matrix (in zero offset form).
                 Dimensions are 1 to ldim and 1 to ndim. This matrix can
                 be either matrix Amat or Bmat.
@RETURNS    : (nothing)
@DESCRIPTION: Multiplies two matrices.
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : March 2, 1992 (Peter Neelin)
@MODIFIED   : March 2, 1992 (P.N.)
                 - Changed so that calling program can use an input matrix for
                 output.
---------------------------------------------------------------------------- */
void matrix_multiply(int ldim, int mdim, int ndim, 
                            float **Amat, float **Bmat, float **Cmat)
{
   int i,j;
   float **Ctemp;

   /* Allocate a temporary matrix */
   VIO_ALLOC2D(Ctemp, ldim+1, ndim+1);

   /* Do the multiplication */
   raw_matrix_multiply(ldim,mdim,ndim,Amat,Bmat,Ctemp);

   /* Copy the result */
   for (i=1; i <= ldim; ++i)
      for (j=1; j <= ndim; ++j)
         Cmat[i][j] = Ctemp[i][j];

   /* Free the matrix */
   VIO_FREE2D(Ctemp);
}
                  

/* ----------------------------- MNI Header -----------------------------------
@NAME       : trace
@INPUT      : size   - size of the_matrix (the_matrix should be square)
              the_matrix - matrix for which trace should be calculated (in 
                 zero offset form). Dimensions are 1 to size and 
                 1 to size.
@OUTPUT     : (none)
@RETURNS    : trace of matrix
@DESCRIPTION: Calculates the trace of a matrix.
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : Feb. 26, 1990 (Weiqian Dai)
@MODIFIED   : January 31, 1992 (Peter Neelin)
                 - change to roughly NIL-abiding code and modified calling
                 sequence.
---------------------------------------------------------------------------- */
float trace(int size, float **the_matrix)
{
   float sum=0.;
   int i;

   for (i=1; i <= size; ++i) {
      sum += the_matrix[i][i];
   }

   return(sum);
}


/* ----------------------------- MNI Header -----------------------------------
@NAME       : matrix_scalar_multiply
@INPUT      : rows    - number of rows of the_matrix.
              cols    - number of columns of the_matrix
              scalar  - scalar by which the_matrix should be multiplied.
              the_matrix  - matrix to be multiplied (in zero offset 
                 form). Dimensions are 1 to rows and 1 to cols.
@OUTPUT     : product - result of multiply ( in zero offset form).
                 Dimensions are 1 to rows and 1 to cols. This matrix
                 can be the input matrix.
@RETURNS    : (nothing)
@DESCRIPTION: Multiplies a matrix by a scalar.
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : Feb. 26, 1990 (Weiqian Dai)
@MODIFIED   : January 31, 1992 (Peter Neelin)
                 - change to roughly NIL-abiding code and modified calling
                 sequence.
---------------------------------------------------------------------------- */
void matrix_scalar_multiply(int rows, int cols, float scalar, 
                            float **the_matrix, float **product)
{
   int i,j;

   for (i=1; i <= rows; ++i)
      for (j=1; j<=cols; ++j)
         product[i][j]=scalar*the_matrix[i][j];
}




/* ----------------------------- MNI Header -----------------------------------
@NAME       : nr_identd, nr_identf - make identity matrix
@INPUT      : A - pointer to matrix
              m1,m2 - row limits
              n1,n2 - col limits
              (matrix in zero offset form, allocated by calling routine)
@OUTPUT     : identiy matrix in A
@RETURNS    : (nothing)
@DESCRIPTION: 
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : Tue Jun  1 12:49:21 EST 1993 (Louis Collins)
@MODIFIED   : 

---------------------------------------------------------------------------- */
void nr_identd(double **A, int m1, int m2, int n1, int n2 )
{

   int i,j;

   for (i=m1; i<=m2; ++i)
      for (j=n1; j<=n2; ++j) {
         if (i==j) 
            A[i][j] = 1.0;
         else
            A[i][j] = 0.0;
      }
   
}

void nr_identf(float **A, int m1, int m2, int n1, int n2 )
{

   int i,j;

   for (i=m1; i<=m2; ++i)
      for (j=n1; j<=n2; ++j) {
         if (i==j) 
            A[i][j] = 1.0;
         else
            A[i][j] = 0.0;
      }
   
}

/* ----------------------------- MNI Header -----------------------------------
@NAME       : nr_copyd, nr_copyf - copy matrix
@INPUT      : A - source matrix
              m1,m2 - row limits
              n1,n2 - col limits
              (matrix in zero offset form, allocated by calling routine)
@OUTPUT     : B - copy of A
@RETURNS    : (nothing)
@DESCRIPTION: 
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : Tue Jun  1 12:49:21 EST 1993 (Louis Collins)
@MODIFIED   : 

---------------------------------------------------------------------------- */
void nr_copyd(double **A, int m1, int m2, int n1, int n2, double **B )
{
   int i,j;

   for (i=m1; i<=m2; ++i)
      for (j=n1; j<=n2; ++j)
         B[i][j] = A[i][j];
}

void nr_copyf(float  **A, int m1, int m2, int n1, int n2, float **B )
{
   int i,j;

   for (i=m1; i<=m2; ++i)
      for (j=n1; j<=n2; ++j)
         B[i][j] = A[i][j];
}



/* ----------------------------- MNI Header -----------------------------------
@NAME       : nr_rotxd,nr_rotxf - make rot X matrix
@INPUT      : M - 4x4 matrix
              a - rotation angle in radians
              (matrix in zero offset form, allocated by calling routine)
@OUTPUT     : modified matrix M
@RETURNS    : (nothing)
@DESCRIPTION: 
@METHOD     : 
   rx =[1   0      0      0 
        0  cos(a)  -sin(a) 0
        0 sin(a)  cos(a) 0
        0   0      0      1];
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : Tue Jun  1 12:49:21 EST 1993 (Louis Collins)
@MODIFIED   : Tue Jun  8 08:44:59 EST 1993 (LC) changed to mat*vec format
---------------------------------------------------------------------------- */
void nr_rotxd(double **M, double a)
{
   nr_identd(M,1,4,1,4);

   M[2][2] = cos(a);   M[2][3] = -sin(a);
   M[3][2] = sin(a);   M[3][3] = cos(a);
}


void nr_rotxf(float **M, float a)
{
   nr_identf(M,1,4,1,4);

   M[2][2] = cos((double)a);    M[2][3] = -sin((double)a);
   M[3][2] = sin((double)a);   M[3][3] = cos((double)a);
}


/* ----------------------------- MNI Header -----------------------------------
@NAME       : nr_rotyd,nr_rotyf - make rot Y matrix
@INPUT      : M - 4x4 matrix
              a - rotation angle in radians
              (matrix in zero offset form, allocated by calling routine)
@RETURNS    : (nothing)
@DESCRIPTION: 
@METHOD     : 
ry = [  cos(a)   0 sin(a)  0 
        0       1   0       0
        -sin(a)  0  cos(a)   0
        0   0      0      1];
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : Tue Jun  1 12:49:21 EST 1993 (Louis Collins)
@MODIFIED   : Tue Jun  8 08:44:59 EST 1993 (LC) changed to mat*vec format
---------------------------------------------------------------------------- */
void nr_rotyd(double **M,double a)
{

   nr_identd(M,1,4,1,4);

   M[1][1] = cos(a);   M[1][3] = sin(a);
   M[3][1] = -sin(a);   M[3][3] = cos(a);
}

void nr_rotyf(float **M, float a)
{

   nr_identf(M,1,4,1,4);

   M[1][1] = cos((double)a);   M[1][3] = sin((double)a);
   M[3][1] = -sin((double)a);   M[3][3] = cos((double)a);
}


/* ----------------------------- MNI Header -----------------------------------
@NAME       : nr_rotzd, nr_rotzf - make rot Z matrix
@INPUT      : M - 4x4 matrix
              a - rotation angle in radians
              (matrix in zero offset form, allocated by calling routine)
@RETURNS    : (nothing)
@DESCRIPTION: 
@METHOD     : 
rz = [cos(a)  -sin(a) 0  0
      sin(a) cos(a) 0  0
        0     0      1  0
        0     0      0  1];
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : Tue Jun  1 12:49:21 EST 1993 (Louis Collins)
@MODIFIED   : Tue Jun  8 08:44:59 EST 1993 (LC) changed to mat*vec format
---------------------------------------------------------------------------- */
void nr_rotzd(double **M,double a)
{

   nr_identd(M,1,4,1,4);

   M[1][1] = cos(a);   M[1][2] = -sin(a);
   M[2][1] = sin(a);  M[2][2] = cos(a);
}

void nr_rotzf(float **M, float a)
{

   nr_identf(M,1,4,1,4);

   M[1][1] = cos((double)a);   M[1][2] = -sin((double)a);
   M[2][1] = sin((double)a);  M[2][2] = cos((double)a);
}

/* ----------------------------- MNI Header -----------------------------------
@NAME       : nr_multd, nr_multf - mult matrix
@INPUT      : A - source matrix
              mA1,mA2 - row limits of A
              nA1,nA2 - col limits of A
              B - source matrix
              mB1,mB2 - row limits of B
              nB1,nB2 - col limits of B
              (matrix in zero offset form, allocated by calling routine)
@OUTPUT     : C = A * B
@RETURNS    : (nothing)
@DESCRIPTION: 
   Routine multiplies matrices A*B to give C. A is a mA x nA matrix and
   B is a mB x nB matrix. The result is returned in C which is mA x nB.
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : (nothing special)
@CREATED    : Tue Jun  1 12:49:21 EST 1993 (Louis Collins)
@MODIFIED   : 

---------------------------------------------------------------------------- */

void nr_multd(double **A, int mA1, int mA2, int nA1, int nA2, 
         double **B, int mB1, int mB2, int nB1, int nB2, 
         double **C )
{
   int i, j, k;

   for ( k = mA1; k <= mA2; k++ ) {
      for ( i = nB1; i <= nB2; i++ ) {
         C[k][i] = 0.0;
         for ( j = mB1; j <= mB2; j++ ) {
            C[k][i] += A[k][j] * B[j][i];
         }
      }
   }

   return;
}


void nr_multf(float **A, int mA1, int mA2, int nA1, int nA2, 
         float **B, int mB1, int mB2, int nB1, int nB2, 
         float **C)
{
   int i, j, k;

   for ( k = mA1; k <= mA2; k++ ) {
      for ( i = nB1; i <= nB2; i++ ) {
         C[k][i] = 0.0;
         for ( j = mB1; j <= mB2; j++ ) {
            C[k][i] += A[k][j] * B[j][i];
         }
      }
   }

   return;
}


/* ----------------------------- MNI Header -----------------------------------
@NAME       : transformations_to_homogeneous
@INPUT      : ndim    - number of dimensions
              translation - zero offset vector (1 to ndim) that 
                 specifies the translation to be applied first.
              centre_of_rotation - zero offset vector (1 to ndim) that
                 specifies the centre of rotation and scaling.
              rotation - zero offset matrix (1 to ndim by 1 to ndim) 
                 for rotation about centre_of_rotation (applied after 
                 translation). Note that this matrix need not only specify
                 rotation/reflexion - any ndim x ndim matrix will work.
              scale - Scalar value giving global scaling to be applied after
                 translation and rotation.
@OUTPUT     : transformation - zero offset matrix (1 to ndim+1 by
                 1 to ndim+1) specifying the transformation for homogeneous 
                 coordinates. To apply this transformation, a point
                 vector should be pre-multiplied by this matrix, with the
                 last coordinate of the ndim+1 point vector having value
                 one. The calling routine must allocate space for this
                 matrix.
@RETURNS    : (nothing)
@DESCRIPTION: Computes a transformation matrix in homogeneous coordinates
              given a translation, a rotation matrix (or other 
              non-homogeneous matrix) and a global scaling factor.
              Transformations are applied in that order.
@METHOD     : Apply the following operations (multiply from left to right):
                 1) Translate by translation
                 2) Translate by -centre_of_rotation
                 3) Rotate
                 4) Scale
                 5) Translate by centre_of_rotation
@GLOBALS    : (none)
@CALLS      : translation_to_homogeneous
              matrix_multiply
              matrix_scalar_multiply
@CREATED    : February 7, 1992 (Peter Neelin)
@MODIFIED   : 
Fri Jun  4 14:10:34 EST 1993  LC
   changed matrices, so that they must be applied by pre-multiplication:
      ie newvec = matrix * oldvec
---------------------------------------------------------------------------- */
void transformations_to_homogeneous(int ndim, 
                  float *translation, float *centre_of_rotation,
                  float **rotation, float scale,
                  float **transformation)
{
   int i;
   int size;
   float *centre_translate;
   float **trans1, **trans2;
   float **trans_temp, **rotation_and_scale;

   size=ndim+1;

   /* Allocate matrices and vectors */
   ALLOC(centre_translate,ndim+1);
   VIO_ALLOC2D(trans1 ,size+1, size+1);
   VIO_ALLOC2D(trans2 ,size+1, size+1);
   VIO_ALLOC2D(trans_temp,size+1, size+1); 
   VIO_ALLOC2D(rotation_and_scale,ndim+1, ndim+1);


   /* Construct translation matrix */
   translation_to_homogeneous(ndim, translation, trans1);


   /* Construct translation matrix for centre of rotation and
      apply it */
   for (i=1; i<=ndim; i++) centre_translate[i] = -centre_of_rotation[i];
   translation_to_homogeneous(ndim, centre_translate, trans_temp);
   matrix_multiply(size, size, size, trans1, trans_temp, trans2);


   /* Scale rotation matrix, then convert it to homogeneous coordinates and
      apply it */
   matrix_scalar_multiply(ndim, ndim, scale, rotation, rotation_and_scale);
   rotation_to_homogeneous(ndim, rotation_and_scale, trans_temp);
   matrix_multiply(size, size, size, trans2, trans_temp, trans1);


   /* Return to centre of rotation */
   translation_to_homogeneous(ndim, centre_of_rotation, trans_temp);
   matrix_multiply(size, size, size, trans1, trans_temp, transformation);


   /* Free matrices */
   FREE(  centre_translate);
   VIO_FREE2D(trans1);
   VIO_FREE2D(trans2);
   VIO_FREE2D(trans_temp);
   VIO_FREE2D(rotation_and_scale);

}

/* ----------------------------- MNI Header -----------------------------------
@NAME       : translation_to_homogeneous
@INPUT      : ndim    - number of dimensions
              translation - zero offset vector (1 to ndim) that 
                 specifies the translation.
@OUTPUT     : transformation - zero offset matrix (1 to ndim+1 by
                 1 to ndim+1) specifying the transformation for homogeneous 
                 coordinates. To apply this transformation, a point
                 vector should be pre-multiplied by this matrix, with the
                 last coordinate of the ndim+1 point vector having value
                 one. The calling routine must allocate space for this
                 matrix.
@RETURNS    : (nothing)
@DESCRIPTION: Computes a transformation matrix in homogeneous coordinates
              given a translation.
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : 
@CREATED    : February 7, 1992 (Peter Neelin)
@MODIFIED   : 
Fri Jun  4 14:10:34 EST 1993  LC
   changed matrices, so that they must be applied by pre-multiplication:
      ie newvec = matrix * oldvec
---------------------------------------------------------------------------- */
void translation_to_homogeneous(int ndim, float *translation,
                                       float **transformation)
{
   int i,j;
   int size;

   size=ndim+1;

   /* Construct translation matrix */
   for (i=1; i<=ndim; i++) {
      for (j=1; j<=size; j++) {
         if (i == j) {
            transformation[i][j] = 1.0;
         }
         else {
            transformation[i][j] = 0.0;
         }
      }
   }
   for (j=1; j<=ndim; j++) {
      transformation[j][size] = translation[j];
   }

   transformation[size][size] = 1.0;

}

/* ----------------------------- MNI Header -----------------------------------
@NAME       : rotation_to_homogeneous
@INPUT      : ndim    - number of dimensions
              rotation - zero offset matrix (1 to ndim by 1 to ndim) 
                 for rotation about origin. Note that this matrix need not 
                 only specify rotation/reflexion - any ndim x ndim matrix 
                 will work.
@OUTPUT     : transformation - zero offset matrix (1 to ndim+1 by
                 1 to ndim+1) specifying the transformation for homogeneous 
                 coordinates. To apply this transformation, a point
                 vector should be pre-multiplied by this matrix, with the
                 last coordinate of the ndim+1 point vector having value
                 one. The calling routine must allocate space for this
                 matrix.
@RETURNS    : (nothing)
@DESCRIPTION: Computes a transformation matrix in homogeneous coordinates
              given a rotation matrix.
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : 
@CREATED    : February 7, 1992 (Peter Neelin)
@MODIFIED   : 
Fri Jun  4 14:10:34 EST 1993  LC
   changed matrices, so that they must be applied by pre-multiplication:
      ie newvec = matrix * oldvec
---------------------------------------------------------------------------- */
void rotation_to_homogeneous(int ndim, float **rotation,
                                       float **transformation)
{
   int i,j;
   int size;

   size=ndim+1;

   /* Construct  matrix */
   for (i=1; i<=size; i++) {
      for (j=1; j<=size; j++) {
         if ((i==size) || (j==size)) {
            transformation[i][j] = 0.0;
         }
         else {
            transformation[i][j] = rotation[i][j];
         }
      }
   }

   transformation[size][size] = 1.0;

}


/* ----------------------------- MNI Header -----------------------------------
@NAME       : angles_to_homogeneous
@INPUT      : ndim    - number of dimensions
              angles - zero offset array (1 to ndim)
                 for rotation angles (in radians) about origin. 
@OUTPUT     : transformation - zero offset matrix (1 to ndim+1 by
                 1 to ndim+1) specifying the transformation for homogeneous 
                 coordinates. To apply this transformation, a point
                 vector should be pre-multiplied by this matrix, with the
                 last coordinate of the ndim+1 point vector having value
                 one. The calling routine must allocate space for this
                 matrix.
@RETURNS    : (nothing)
@DESCRIPTION: Computes a transformation matrix in homogeneous coordinates
              given a rotation matrix.
@METHOD     : 
@GLOBALS    : (none)
@CALLS      : 
@CREATED    : Fri Jun  4 14:10:34 EST 1993  LC
@MODIFIED   : 
---------------------------------------------------------------------------- */
void angles_to_homogeneous(int ndim, float *angles,
                                  float **transformation)
{
   int i,j;
   int size;
   float **rot_matrix;


   size=ndim+1;

   VIO_ALLOC2D(rot_matrix,5,5);


   if (ndim==2 || ndim==3) {

     if (ndim==2)
       nr_rotzf(rot_matrix,*angles );
     else
       make_rots(rot_matrix, 
                 (float)(angles[0]),
                 (float)(angles[1]),
                 (float)(angles[2]));

     /* Construct  matrix */
     for (i=1; i<=size; i++) {
       for (j=1; j<=size; j++) {
         if ((i==size) || (j==size)) {
           transformation[i][j] = 0.0;
         }
         else {
           transformation[i][j] = rot_matrix[i][j];
         }
       }
     }
     transformation[size][size] = 1.0;

   }
   else {
     (void)fprintf (stderr,"Can't handle %d dimensions in angles_to_homogeneous()\n",ndim);
     (void)fprintf (stderr,"Error in %s, line %d\n",__FILE__,__LINE__);
     exit(-1);
   }


   VIO_FREE2D(rot_matrix);
}



/* ----------------------------- MNI Header -----------------------------------
@NAME       : make_rots
@INPUT      : rot_x, rot_y, rot_z - three rotation angles, in radians.
@OUTPUT     : xmat, a zero offset matrix for homogeous transformations
@RETURNS    : nothing
@DESCRIPTION: to be applied by premultiplication, ie rot*vec = newvec
@METHOD     : 
@GLOBALS    : 
@CALLS      : 
@CREATED    : Tue Jun  8 08:44:59 EST 1993 LC
@MODIFIED   : 
---------------------------------------------------------------------------- */

void   make_rots(float **xmat, float rot_x, float rot_y, float rot_z)
{
   float
      **TRX,
      **TRY,
      **TRZ,
      **T1;
   
   VIO_ALLOC2D(TRX  ,5,5);
   VIO_ALLOC2D(TRY  ,5,5);
   VIO_ALLOC2D(TRZ  ,5,5);
   VIO_ALLOC2D(T1   ,5,5);

   nr_rotxf(TRX, rot_x);             /* create the rotate X matrix */
   nr_rotyf(TRY, rot_y);             /* create the rotate Y matrix */
   nr_rotzf(TRZ, rot_z);             /* create the rotate Z matrix */
   
   nr_multf(TRY,1,4,1,4,  TRX,1,4,1,4,  T1); /* apply rx and ry */
   nr_multf(TRZ,1,4,1,4,  T1,1,4,1,4,   xmat); /* apply rz */


   VIO_FREE2D(TRX);
   VIO_FREE2D(TRY);
   VIO_FREE2D(TRZ);
   VIO_FREE2D(T1 );

}


/* ----------------------------- MNI Header -----------------------------------
@NAME       : make_shears
@INPUT      : shear - an array of six shear parameters.
@OUTPUT     : xmat, a zero offset matrix for homogeous transformations
@RETURNS    : nothing
@DESCRIPTION: to be applied by premultiplication, ie rot*vec = newvec
@METHOD     : 
                xmat = [ 1 a b 0
                         c 1 d 0
                         e f 1 0
                         0 0 0 1 ];

                where shear[] = [a b c d e f]     

@CREATED    : Sat Apr 16 10:44:26 EST 1994
@MODIFIED   : 
---------------------------------------------------------------------------- */

void   make_shears(float **xmat,                                         
                 double *shears)
{

    nr_identf(xmat,1,4,1,4);        /* start with identity */
    xmat[2][1] = shears[0];
    xmat[3][1] = shears[1];
    xmat[3][2] = shears[2];
  
}




/* ----------------------------- MNI Header -----------------------------------
@NAME       : build_transformation_matrix
@INPUT      : center, translations, scales, rotations
@OUTPUT     : *lt->mat - a linear transformation matrix
@RETURNS    : nothing
@DESCRIPTION: mat = (T)(C)(S)(SH)(R)(-C)
               the matrix is to be  PREmultiplied with a column vector (mat*colvec)
               when used in the application
@METHOD     : 
@GLOBALS    : 
@CALLS      : 
@CREATED    : Thu Jun  3 09:37:56 EST 1993 lc
@MODIFIED   : 
---------------------------------------------------------------------------- */

void build_transformation_matrix(VIO_Transform *trans,
                                        double *center,
                                        double *translations,
                                        double *scales,
                                        double *shears,
                                        double *rotations)
{
  
  float
    **T,
    **SH,
    **S,
    **R,
    **C,
    **T1,
    **T2,
    **T3,
    **T4;
  int
    i,j;
  
  VIO_ALLOC2D(T  ,5,5);
  VIO_ALLOC2D(SH ,5,5);
  VIO_ALLOC2D(S  ,5,5);
  VIO_ALLOC2D(R  ,5,5);
  VIO_ALLOC2D(C  ,5,5);
  VIO_ALLOC2D(T1 ,5,5);
  VIO_ALLOC2D(T2 ,5,5);
  VIO_ALLOC2D(T3 ,5,5);
  VIO_ALLOC2D(T4 ,5,5);
  
                                             /* mat = (T)(C)(SH)(S)(R)(-C) */

  nr_identf(T,1,4,1,4);                     /* make (T)(C) */
  for(i=0; i<3; i++) {
    T[1+i][4] = translations[i] + center[i];                
  }
                                /* make rotation matix */
  make_rots(R,
            (float)(rotations[0]),
            (float)(rotations[1]),
            (float)(rotations[2])); 

                                /* make shear rotation matrix */
  make_shears(SH, shears);

                                /* make scaling matrix */
  nr_identf(S,1,4,1,4);                   
  for(i=0; i<3; i++) {
    S[1+i][1+i] = scales[i];
  }

  nr_identf(C,1,4,1,4);      /* make center          */
  for(i=0; i<3; i++) {
    C[1+i][4] = -center[i];
  }

  nr_multf(T, 1,4,1,4, S  ,1,4,1,4, T1 );  
  nr_multf(T1,1,4,1,4, SH ,1,4,1,4, T2 );  
  nr_multf(T2,1,4,1,4, R  ,1,4,1,4, T3 );  
  nr_multf(T3,1,4,1,4, C  ,1,4,1,4, T4 );  

  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      Transform_elem(*trans, i, j ) = T4[i+1][j+1];

  VIO_FREE2D(T    );
  VIO_FREE2D(SH   );
  VIO_FREE2D(S    );
  VIO_FREE2D(R    );
  VIO_FREE2D(C    );
  VIO_FREE2D(T1   );
  VIO_FREE2D(T2   );
  VIO_FREE2D(T3   );
  VIO_FREE2D(T4   );
}

/* ----------------------------- MNI Header -----------------------------------
@NAME       : build_transformation_matrix_quater
@INPUT      : center, translations, scales, quaternions
@OUTPUT     : *lt->mat - a linear transformation matrix
@RETURNS    : nothing
@DESCRIPTION: mat = (T)(C)(S)(SH)(R)(-C)
               the matrix is to be  PREmultiplied with a column vector (mat*colvec)
               when used in the application
               same as build_transformation_matrix but with quaternions
@METHOD     : 
@GLOBALS    : 
@CALLS      : 
@CREATED    : Thr Apr 18 10:45:56 EST 2002 pln
@MODIFIED   : 
---------------------------------------------------------------------------- */

void build_transformation_matrix_quater(VIO_Transform *trans,
                                               double *center,
                                               double *translations,
                                               double *scales,
                                               double *shears,
                                               double *quaternions)
{
  
  float
    **T,
    **SH,
    **S,
    **R,
    **C,
    **T1,
    **T2,
    **T3,
    **T4;

  double normal;


  int
    i,j;
  
  VIO_ALLOC2D(T  ,5,5);
  VIO_ALLOC2D(SH ,5,5);
  VIO_ALLOC2D(S  ,5,5);
  VIO_ALLOC2D(R  ,5,5);
  VIO_ALLOC2D(C  ,5,5);
  VIO_ALLOC2D(T1 ,5,5);
  VIO_ALLOC2D(T2 ,5,5);
  VIO_ALLOC2D(T3 ,5,5);
  VIO_ALLOC2D(T4 ,5,5);
  
 
  


  
  normal=(quaternions[0]*quaternions[0] + quaternions[1]*quaternions[1] + quaternions[2]*quaternions[2] + quaternions[3]*quaternions[3]);
  if (normal>1){
   for(i = 0; i < 4; i++){
      quaternions[i] /= normal;
   }} 
   
   

                        /* mat = (T)(C)(SH)(S)(R)(-C) */

  nr_identf(T,1,4,1,4);



                                             /* make (T)(C) */
  for(i=0; i<3; i++) {
    T[1+i][4] = translations[i] + center[i];                
  }
   
  

  build_rotmatrix(R,quaternions); /* make rotation matrix from quaternions */
  

                                /* make shear rotation matrix */
  make_shears(SH, shears);

                                /* make scaling matrix */
  nr_identf(S,1,4,1,4);                   
  for(i=0; i<3; i++) {
    S[1+i][1+i] = scales[i];
  }


  nr_identf(C,1,4,1,4);      /* make center          */
  for(i=0; i<3; i++) {
    C[1+i][4] = -center[i];                
  }

  nr_multf(T, 1,4,1,4, S  ,1,4,1,4, T1 );  
  nr_multf(T1,1,4,1,4, SH ,1,4,1,4, T2 );  
  nr_multf(T2,1,4,1,4, R  ,1,4,1,4, T3 );  
  nr_multf(T3,1,4,1,4, C  ,1,4,1,4, T4 );  

  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      Transform_elem(*trans, i, j ) = T4[i+1][j+1];

  VIO_FREE2D(T    );
  VIO_FREE2D(SH   );
  VIO_FREE2D(S    );
  VIO_FREE2D(R    );
  VIO_FREE2D(C );
  VIO_FREE2D(T1   );
  VIO_FREE2D(T2   );
  VIO_FREE2D(T3   );
  VIO_FREE2D(T4   );
}

/* ----------------------------- MNI Header -----------------------------------
@NAME       : build_inverse_transformation_matrix
@INPUT      : center, translations, scales, rotations
@OUTPUT     : the inverse linear transformation matrix of mat:
                since mat = (T)(C)(SH)(S)(R)(-C), then

                invmat = (C)(inv(r))(inv(S))(inv(SH))(-C)(-T)

@RETURNS    : nothing
@DESCRIPTION: 
               the matrix is to be  PREmultiplied with a vector (mat*vec)
               when used in the application
@METHOD     : 
@GLOBALS    : 
@CALLS      : 
@CREATED    : Tue Jun 15 16:45:35 EST 1993 LC
@MODIFIED   : 
---------------------------------------------------------------------------- */
void build_inverse_transformation_matrix(VIO_Transform *trans,
                                                double *center,
                                                double *translations,
                                                double *scales,
                                                double *shears,
                                                double *rotations)
{
  float
    **T,
    **SH,
    **S,
    **R,
    **C,
    **T1,
    **T2,
    **T3,
    **T4;
  int
    i,j;
  
  VIO_ALLOC2D(T   ,5,5);
  VIO_ALLOC2D(SH  ,5,5);
  VIO_ALLOC2D(S   ,5,5);
  VIO_ALLOC2D(R   ,5,5);
  VIO_ALLOC2D(C   ,5,5);
  VIO_ALLOC2D(T1  ,5,5);
  VIO_ALLOC2D(T2  ,5,5);
  VIO_ALLOC2D(T3  ,5,5);
  VIO_ALLOC2D(T4  ,5,5);
  
                                /* invmat = (C)(inv(r))(inv(S))(inv(SH))(-C)(-T)
                                   mat = (T)(C)(SH)(S)(R)(-C) */

  nr_identf(T,1,4,1,4);                     /* make (-T)(-C) */
  for(i=0; i<3; i++) {
    T[1+i][4] = -translations[i] - center[i];                
  }

                                /* make rotation matix */
  make_rots(T1,
            (float)(rotations[0]),
            (float)(rotations[1]),
            (float)(rotations[2])); 


  transpose(4,4,T1,R);
  

  make_shears(T1,shears);        /* make shear rotation matrix */
  invertmatrix(4, T1, SH);        /* get inverse of the matrix */

                                /* make scaling matrix */
  nr_identf(S,1,4,1,4);                   
  for(i=0; i<3; i++) {
    if (scales[i] != 0.0)
      S[1+i][1+i] = 1/scales[i];
    else
      S[1+i][1+i] = 1.0;
  }

  nr_identf(C,1,4,1,4);      /* make center          */
  for(i=0; i<3; i++) {
    C[1+i][4] = center[i];                
  }

  nr_multf(C,1,4,1,4,  R ,1,4,1,4, T1 );  
  nr_multf(T1,1,4,1,4, SH,1,4,1,4, T2 );  
  nr_multf(T2,1,4,1,4, S ,1,4,1,4, T3 );  
  nr_multf(T3,1,4,1,4, T ,1,4,1,4, T4 );  

  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      Transform_elem(*trans, i, j ) = T4[i+1][j+1];

  VIO_FREE2D(T    );
  VIO_FREE2D(SH   );
  VIO_FREE2D(S    );
  VIO_FREE2D(R    );
  VIO_FREE2D(C );
  VIO_FREE2D(T1   );
  VIO_FREE2D(T2   );
  VIO_FREE2D(T3   );
  VIO_FREE2D(T4   );
}


/* ----------------------------- MNI Header -----------------------------------
@NAME       : build_inverse_transformation_matrix_quater
@INPUT      : center, translations, scales, quaternions
@OUTPUT     : the inverse linear transformation matrix of mat:
                since mat = (T)(C)(SH)(S)(R)(-C), then

                invmat = (C)(inv(r))(inv(S))(inv(SH))(-C)(-T)

@RETURNS    : nothing
@DESCRIPTION: 
               the matrix is to be  PREmultiplied with a vector (mat*vec)
               when used in the application
               same as build_inverse_transformation_matrix but with quaternions
@METHOD     : 
@GLOBALS    : 
@CALLS      : 
@CREATED    : Thr Apr 18 10:45:56 EST 2002 pln
@MODIFIED   : 
---------------------------------------------------------------------------- */
void build_inverse_transformation_matrix_quater(VIO_Transform *trans,
                                                       double *center,
                                                       double *translations,
                                                       double *scales,
                                                       double *shears,
                                                       double *quaternions)
{
  float
    **T,
    **SH,
    **S,
    **R,
    **C,
    **T1,
    **T2,
    **T3,
    **T4;

  int
    i,j;
  
  VIO_ALLOC2D(T   ,5,5);
  VIO_ALLOC2D(SH  ,5,5);
  VIO_ALLOC2D(S   ,5,5);
  VIO_ALLOC2D(R   ,5,5);
  VIO_ALLOC2D(C   ,5,5);
  VIO_ALLOC2D(T1  ,5,5);
  VIO_ALLOC2D(T2  ,5,5);
  VIO_ALLOC2D(T3  ,5,5);
  VIO_ALLOC2D(T4  ,5,5);
  
                                /* invmat = (C)(inv(r))(inv(S))(inv(SH))(-C)(-T)
                                   mat = (T)(C)(SH)(S)(R)(-C) */

  nr_identf(T,1,4,1,4);                     /* make (-T)(-C) */
  for(i=0; i<3; i++) {
    T[1+i][4] = -translations[i] - center[i];                
  }

  

  build_rotmatrix(T1,quaternions); /* make rotation matrix from quaternions */
  transpose(4,4,T1,R);
  

  make_shears(T1,shears);        /* make shear rotation matrix */
  invertmatrix(4, T1, SH);        /* get inverse of the matrix */

                                /* make scaling matrix */
  nr_identf(S,1,4,1,4);                   
  for(i=0; i<3; i++) {
    if (scales[i] != 0.0)
      S[1+i][1+i] = 1/scales[i];
    else
      S[1+i][1+i] = 1.0;
  }

  nr_identf(C,1,4,1,4);      /* make center          */
  for(i=0; i<3; i++) {
    C[1+i][4] = center[i];                
  }

  nr_multf(C,1,4,1,4,  R ,1,4,1,4, T1 );  
  nr_multf(T1,1,4,1,4, SH,1,4,1,4, T2 );  
  nr_multf(T2,1,4,1,4, S ,1,4,1,4, T3 );  
  nr_multf(T3,1,4,1,4, T ,1,4,1,4, T4 );  

  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      Transform_elem(*trans, i, j ) = T4[i+1][j+1];

  VIO_FREE2D(T    );
  VIO_FREE2D(SH   );
  VIO_FREE2D(S    );
  VIO_FREE2D(R    );
  VIO_FREE2D(C );
  VIO_FREE2D(T1   );
  VIO_FREE2D(T2   );
  VIO_FREE2D(T3   );
  VIO_FREE2D(T4   );
}

/* ----------------------------- MNI Header -----------------------------------
@NAME       : extract_parameters_from_matrix
@INPUT      : trans    - a linear transformation matrix structure
              center   - an array of the desired center of rotation and scaling.
@OUTPUT     : translations, scales, rotations
@RETURNS    : nothing
@DESCRIPTION: mat = (C)(SH)(S)(R)(-C)(T)
@METHOD     : 
@GLOBALS    : 
@CALLS      : 
@CREATED    : Thu Jun  3 09:37:56 EST 1993 lc
@MODIFIED   : Sun Apr 17 09:54:14 EST 1994 - tried to extract shear parameters

      if det(ROT) != 1, then the ROT matrix is not a pure rotation matrix.
      I will find the shear matrix required, by building a rotation matrix R1
      from the extracted rotation parameters (rx,ry and rz), and multiply ROT by 
      inv(R1).
---------------------------------------------------------------------------- */

VIO_BOOL extract_parameters_from_matrix(VIO_Transform *trans,
                                              double *center,
                                              double *translations,
                                              double *scales,
                                              double *shears,
                                              double *rotations)
{
  int 
    i,j;

  float 
    magnitude,
    **center_of_rotation,
    **result,
    **unit_vec,
    *ang,*tmp,
    **xmat,**T,**Tinv,**C,**Sinv,**R,**SR,**SRinv,**Cinv,**TMP1,**TMP2;

  VIO_ALLOC2D(xmat  ,5,5); nr_identf(xmat ,1,4,1,4);
  VIO_ALLOC2D(TMP1  ,5,5); nr_identf(TMP1 ,1,4,1,4);
  VIO_ALLOC2D(TMP2  ,5,5); nr_identf(TMP2 ,1,4,1,4);
  VIO_ALLOC2D(Cinv  ,5,5); nr_identf(Cinv ,1,4,1,4);
  VIO_ALLOC2D(SR    ,5,5); nr_identf(SR   ,1,4,1,4);
  VIO_ALLOC2D(SRinv ,5,5); nr_identf(SRinv,1,4,1,4);
  VIO_ALLOC2D(Sinv  ,5,5); nr_identf(Sinv ,1,4,1,4); 
  VIO_ALLOC2D(T     ,5,5); nr_identf(T    ,1,4,1,4);
  VIO_ALLOC2D(Tinv  ,5,5); nr_identf(Tinv ,1,4,1,4);
  VIO_ALLOC2D(C     ,5,5); nr_identf(C    ,1,4,1,4);
  VIO_ALLOC2D(R     ,5,5); nr_identf(R    ,1,4,1,4);

  VIO_ALLOC2D(center_of_rotation ,5,5);        /* make column vectors */
  VIO_ALLOC2D(result             ,5,5);
  VIO_ALLOC2D(unit_vec           ,5,5);

  ALLOC(tmp ,4);
  ALLOC(ang ,4);


  for(i=0; i<=3; i++)        /* copy the input matrix */
    for(j=0; j<=3; j++)
      xmat[i+1][j+1] = (float)Transform_elem(*trans,i,j);

  

  /* -------------DETERMINE THE TRANSLATION FIRST! ---------  */

                                /* see where the center of rotation is displaced... */

  FILL_NR_COLVEC( center_of_rotation, center[0], center[1], center[2] );


  invertmatrix(4, xmat, TMP1);        /* get inverse of the matrix */

  matrix_multiply( 4, 4, 1, xmat, center_of_rotation, result); /* was TMP! in place of xmat */

  SUB_NR_COLVEC( result, result, center_of_rotation );

  for(i=0; i<=2; i++) 
    translations[i] = result[i+1][1];

  /* -------------NOW GET THE SCALING VALUES! ----------------- */

  for(i=0; i<=2; i++) 
    tmp[i+1] = -translations[i];
  translation_to_homogeneous(3, tmp, Tinv); 

  for(i=0; i<=2; i++) 
    tmp[i+1] = center[i];
  translation_to_homogeneous(3, tmp, C); 
  for(i=0; i<=2; i++) 
    tmp[i+1] = -center[i];
  translation_to_homogeneous(3, tmp, Cinv); 


  matrix_multiply(4,4,4, xmat, C, TMP1);    /* get scaling*rotation matrix */


  matrix_multiply(4,4,4, Tinv, TMP1, TMP1);


  matrix_multiply(4,4,4, Cinv, TMP1,    SR);

  invertmatrix(4, SR, SRinv);        /* get inverse of scaling*rotation */

                                /* find each scale by mapping a unit vector backwards,
                                   and finding the magnitude of the result. */
  FILL_NR_COLVEC( unit_vec, 1.0, 0.0, 0.0 );
  matrix_multiply( 4, 4, 1, SRinv, unit_vec, result);
  magnitude = MAG_NR_COLVEC( result );
  if (magnitude != 0.0) {
    scales[0] = 1/magnitude;
    Sinv[1][1] = magnitude;
  }
  else {
    scales[0] = 1.0;
    Sinv[1][1] = 1.0;
  }

  FILL_NR_COLVEC( unit_vec, 0.0, 1.0, 0.0 );
  matrix_multiply( 4, 4, 1, SRinv, unit_vec, result);
  magnitude = MAG_NR_COLVEC( result );
  if (magnitude != 0.0) {
    scales[1] = 1/magnitude;
    Sinv[2][2] = magnitude;
  }
  else {
    scales[1]  = 1.0;
    Sinv[2][2] = 1.0;
  }

  FILL_NR_COLVEC( unit_vec, 0.0, 0.0, 1.0 );
  matrix_multiply( 4, 4, 1, SRinv, unit_vec, result);
  magnitude = MAG_NR_COLVEC( result );
  if (magnitude != 0.0) {
    scales[2] = 1/magnitude;
    Sinv[3][3] = magnitude;
  }
  else {
    scales[2] = 1.0;
    Sinv[3][3] = 1.0;
  }

  /* ------------NOW GET THE ROTATION ANGLES!----- */

                                /* extract rotation matrix */
  matrix_multiply(4,4, 4, Sinv, SR,   R);

                                /* get rotation angles */
  if (!rotmat_to_ang(R, ang)) {
    (void)fprintf(stderr,"Cannot convert R to radians!");
    printmatrix(3,3,R);
    return(FALSE);
  }

  for(i=0; i<=2; i++)
    rotations[i] = ang[i+1];


  VIO_FREE2D(xmat);
  VIO_FREE2D(TMP1);
  VIO_FREE2D(TMP2);
  VIO_FREE2D(Cinv);
  VIO_FREE2D(SR  );
  VIO_FREE2D(SRinv);
  VIO_FREE2D(Sinv);
  VIO_FREE2D(T   );
  VIO_FREE2D(Tinv);
  VIO_FREE2D(C   );
  VIO_FREE2D(R   );
  
  VIO_FREE2D(center_of_rotation);
  VIO_FREE2D(result            );
  VIO_FREE2D(unit_vec          );

  FREE(ang);
  FREE(tmp);

  return(TRUE);
}




/*
   function getparams will get the trnsform  parameters from a 
   transformation matrix 'tran' (that has already had the translation 
   componants removed).
   in description below, I assume that tran is a forward xform, from 
   native space to talairach space.
   I assume that trans is a 4x4 homogeneous matrix, with the 
   principal axis stored in the upper left 3x3 matrix.  The first col
   of tran represents is the coordinate of (1,0,0)' mapped through the
   transformation.  (this means  vec2 = tran * vec1).
  
   trans = [scale][shear][rot]
         = [scale][shear][rz][ry][rx];

   the shear matrix is constrained to be of the form:
     shear = [1 0 0 0
              f 1 0 0
              g h 1 0
              0 0 0 1];
     where f,g,h can take on any value.

   the scale matrix is constrained to be of the form:
     scale = [sx 0  0  0
              0  sy 0  0
              0  0  sz 0
              0  0  0  1];
     where sx,sy,sz must be positive.
   
  all rotations are assumed to be in the range -pi/2..pi/2

  the rotation angles are returned as radians and are applied 
  counter clockwise, when looking down the axis (from the positive end
  towards the origin).

  trans is assumed to be invertible.

  i assume a coordinate system:
             ^ y
             |
             |
             |
             |_______> x
            /
           /
          /z  (towards the viewer).

  
  
  procedure: 
          start with t = inv(tran) (after removing translation from tran )
  
          t = inv(r)*inv(sh)*inv(sc)
                  t maps the talairach space unit vectors into native
                space, which means that the columns of T are the
                direction cosines of these vectors x,y,z

   (1)  the length of the vectors x,y,z give the inverse scaling
        parameters:
             sx = 1/norm(x); sy = 1/norm(y); sz = 1/norm(z);

   (2)  with the constraints on the form of sh above, inv(sh) has the
        same form.  let inv(sh) be as above in terms of a,b and c.
          inv(sh) = [1 0 0 0; a 1 0 0; b c 1 0; 0 0 0 1];

        for c: project y onto z and normalize:

                  /     |y.z|^2     \(1/2)
             c = <  ---------------  >
                  \ |y|^2 - |y.z|^2 /

        for b: project x onto z and normalize

                  /        |x.z|^2        \(1/2)
             b = <  ---------------------  >
                  \ |x|^2 - |x.z|^2 - a^2 /
     
          where a is the projection of x onto the coordinate sys Y axis.

        for a: project x onto z and normalize
           a is taken from (b) above, and normalized... see below

  (3) rots are returned by getrots by giving the input transformation:
        rot_mat = [inv(sh)][inv(sc)][trans]

  (4) once completed, the parameters of sx,sy,sz and a,b,c are
      adjusted so that they maintain the matrix contraints above.


*/
VIO_BOOL extract2_parameters_from_matrix(VIO_Transform *trans,
                                               double *center,
                                               double *translations,
                                               double *scales,
                                               double *shears,
                                               double *rotations)
{
  int 
    i,j;

  float 
    n1,n2,
    magnitude, magz, magx, magy, ai,bi,ci,scalar,a1,
    **center_of_rotation,
    **result,
    **unit_vec,
    *ang,*tmp,**x,**y,**z, **nz, **y_on_z, **ortho_y,
    **xmat,**T,**Tinv,**C,**Sinv,
    **R,**SR,**SRinv,**Cinv,**TMP1,**TMP2;

  VIO_ALLOC2D(xmat  ,5,5); nr_identf(xmat ,1,4,1,4);
  VIO_ALLOC2D(TMP1  ,5,5); nr_identf(TMP1 ,1,4,1,4);
  VIO_ALLOC2D(TMP2  ,5,5); nr_identf(TMP2 ,1,4,1,4);
  VIO_ALLOC2D(Cinv  ,5,5); nr_identf(Cinv ,1,4,1,4);
  VIO_ALLOC2D(SR    ,5,5); nr_identf(SR   ,1,4,1,4);
  VIO_ALLOC2D(SRinv ,5,5); nr_identf(SRinv,1,4,1,4);
  VIO_ALLOC2D(Sinv  ,5,5); nr_identf(Sinv ,1,4,1,4); 
  VIO_ALLOC2D(T     ,5,5); nr_identf(T    ,1,4,1,4);
  VIO_ALLOC2D(Tinv  ,5,5); nr_identf(Tinv ,1,4,1,4);
  VIO_ALLOC2D(C     ,5,5); nr_identf(C    ,1,4,1,4);
  VIO_ALLOC2D(R     ,5,5); nr_identf(R    ,1,4,1,4);

  VIO_ALLOC2D(center_of_rotation ,5,5);        /* make column vectors */
  VIO_ALLOC2D(result             ,5,5);
  VIO_ALLOC2D(unit_vec           ,5,5);
  VIO_ALLOC2D(x                  ,5,5);
  VIO_ALLOC2D(y                  ,5,5);
  VIO_ALLOC2D(z                  ,5,5);
  VIO_ALLOC2D(nz                 ,5,5);
  VIO_ALLOC2D(y_on_z             ,5,5);
  VIO_ALLOC2D(ortho_y            ,5,5);

  ALLOC(tmp ,4);
  ALLOC(ang ,4);

  for(i=0; i<=3; i++)        /* copy the input matrix */
    for(j=0; j<=3; j++)
      xmat[i+1][j+1] = (float)Transform_elem(*trans,i,j);

  /* -------------DETERMINE THE TRANSLATION FIRST! ---------  */

                                /* see where the center of rotation is displaced... */

  FILL_NR_COLVEC( center_of_rotation, center[0], center[1], center[2] );


  invertmatrix(4, xmat, TMP1);        /* get inverse of the matrix */

  matrix_multiply( 4, 4, 1, xmat, center_of_rotation, result); /* was TMP! in place of xmat */

  SUB_NR_COLVEC( result, result, center_of_rotation );

  for(i=0; i<=2; i++) 
    translations[i] = result[i+1][1];

  /* -------------NOW GET THE SCALING VALUES! ----------------- */

  for(i=0; i<=2; i++) 
    tmp[i+1] = -translations[i];
  translation_to_homogeneous(3, tmp, Tinv); 

  for(i=0; i<=2; i++) 
    tmp[i+1] = center[i];
  translation_to_homogeneous(3, tmp, C); 
  for(i=0; i<=2; i++) 
    tmp[i+1] = -center[i];
  translation_to_homogeneous(3, tmp, Cinv); 


  matrix_multiply(4,4,4, xmat, C, TMP1);    /* get scaling*shear*rotation matrix */


  matrix_multiply(4,4,4, Tinv, TMP1, TMP1);


  matrix_multiply(4,4,4, Cinv, TMP1,    SR);

  invertmatrix(4, SR, SRinv);        /* get inverse of scaling*shear*rotation */

                                /* find each scale by mapping a unit vector backwards,
                                   and finding the magnitude of the result. */
  FILL_NR_COLVEC( unit_vec, 1.0, 0.0, 0.0 );
  matrix_multiply( 4, 4, 1, SRinv, unit_vec, result);
  magnitude = MAG_NR_COLVEC( result );
  if (magnitude != 0.0) {
    scales[0] = 1/magnitude;
    Sinv[1][1] = magnitude;
  }
  else {
    scales[0] = 1.0;
    Sinv[1][1] = 1.0;
  }

  FILL_NR_COLVEC( unit_vec, 0.0, 1.0, 0.0 );
  matrix_multiply( 4, 4, 1, SRinv, unit_vec, result);
  magnitude = MAG_NR_COLVEC( result );
  if (magnitude != 0.0) {
    scales[1] = 1/magnitude;
    Sinv[2][2] = magnitude;
  }
  else {
    scales[1]  = 1.0;
    Sinv[2][2] = 1.0;
  }

  FILL_NR_COLVEC( unit_vec, 0.0, 0.0, 1.0 );
  matrix_multiply( 4, 4, 1, SRinv, unit_vec, result);
  magnitude = MAG_NR_COLVEC( result );
  if (magnitude != 0.0) {
    scales[2] = 1/magnitude;
    Sinv[3][3] = magnitude;
  }
  else {
    scales[2] = 1.0;
    Sinv[3][3] = 1.0;
  }

  /* ------------NOW GET THE SHEARS, using the info from above ----- */

  /* SR contains the [scale][shear][rot], must multiply [inv(scale)]*SR
     to get shear*rot. */

                                /* make [scale][rot] */
  matrix_multiply(4,4, 4, Sinv, SR,  TMP1);

                                /* get inverse of [scale][rot] */
  invertmatrix(4, TMP1, SRinv);        


  FILL_NR_COLVEC(x, SRinv[1][1], SRinv[2][1], SRinv[3][1]);
  FILL_NR_COLVEC(y, SRinv[1][2], SRinv[2][2], SRinv[3][2]);
  FILL_NR_COLVEC(z, SRinv[1][3], SRinv[2][3], SRinv[3][3]);



                                /* get normalized z direction  */
  magz = MAG_NR_COLVEC(z);
  SCALAR_MULT_NR_COLVEC( nz, z, 1/magz );

                                /* get a direction perpendicular 
                                   to z, in the yz plane.  */
  scalar = DOTSUM_NR_COLVEC(  y, nz );
  SCALAR_MULT_NR_COLVEC( y_on_z, nz, scalar );

  SUB_NR_COLVEC( result, y, y_on_z ); /* result = y - y_on_z */
  scalar = MAG_NR_COLVEC( result);     /* ortho_y = result ./ norm(result)  */
  SCALAR_MULT_NR_COLVEC( ortho_y, result, 1/scalar);


                                /* GET C for the skew matrix */

  scalar = DOTSUM_NR_COLVEC( y, nz ); /* project y onto z */
  magy   = MAG_NR_COLVEC(y);
  ci = scalar / sqrt((double)( magy*magy - scalar*scalar)) ;
                                /* GET B for the skew matrix */

                                /*    first need a1 */

  a1     = DOTSUM_NR_COLVEC( x, ortho_y ); /* project x onto ortho_y */
  magx   = MAG_NR_COLVEC(x);

                                /*    now get B  */

  scalar = DOTSUM_NR_COLVEC( x, nz );
  bi = scalar / sqrt((double)( magx*magx - scalar*scalar - a1*a1)) ;

                                /* GET A for skew matrix  */

  ai = a1 / sqrt((double)( magx*magx - scalar*scalar - a1*a1));

                                /* normalize the inverse shear parameters.
                                   so that there is no scaling in the matrix 
                                   (all scaling is already accounted for 
                                   in sx,sy and sz. */

  n1 = sqrt((double)(1 + ai*ai + bi*bi));
  n2 = sqrt((double)(1 + ci*ci));

  ai = ai / n1;
  bi = bi / n1;
  ci = ci / n2;

                                /* ai,bi,c1 now contain the parameters for 
                                   the inverse NORMALIZED shear matrix 
                                   (i.e., norm(col_i) = 1.0). */

  
  /* ------------NOW GET THE ROTATION ANGLES!----- */

                                  /*  since SR = [scale][shear][rot], then
                                    rot = [inv(shear)][inv(scale)][SR] */

  nr_identf(TMP1 ,1,4,1,4);        /* make inverse scale matrix */
  TMP1[1][1] = 1/scales[0];
  TMP1[2][2] = 1/scales[1];
  TMP1[3][3] = 1/scales[2];

  nr_identf(TMP2 ,1,4,1,4);        /* make_inverse normalized shear matrix */
  TMP2[1][1] = sqrt((double)(1 - ai*ai - bi*bi));
  TMP2[2][2] = sqrt((double)(1 - ci*ci));
  TMP2[2][1] = ai;
  TMP2[3][1] = bi;
  TMP2[3][2] = ci;


                                /* extract rotation matrix */
  matrix_multiply(4,4, 4, TMP2, TMP1, T);
  matrix_multiply(4,4, 4, T,    SR,   R);

                                /* get rotation angles */
  if (!rotmat_to_ang(R, ang)) {
    (void)fprintf(stderr,"Cannot convert R to radians!");
    printmatrix(3,3,R);
    return(FALSE);
  }

  for(i=0; i<=2; i++)
    rotations[i] = ang[i+1];

  /* ------------NOW ADJUST THE SCALE AND SKEW PARAMETERS ------------ */

  invertmatrix(4, T, Tinv);        /* get inverse of the matrix */
  

  scales[0] = Tinv[1][1];
  scales[1] = Tinv[2][2];
  scales[2] = Tinv[3][3];
  shears[0] = Tinv[2][1]/scales[1] ;
  shears[1] = Tinv[3][1]/scales[2] ;
  shears[2] = Tinv[3][2]/scales[2] ;
  

  VIO_FREE2D(xmat);
  VIO_FREE2D(TMP1);
  VIO_FREE2D(TMP2);
  VIO_FREE2D(Cinv);
  VIO_FREE2D(SR  );
  VIO_FREE2D(SRinv);
  VIO_FREE2D(Sinv);
  VIO_FREE2D(T   );
  VIO_FREE2D(Tinv);
  VIO_FREE2D(C   );
  VIO_FREE2D(R   );
  
  VIO_FREE2D(center_of_rotation);
  VIO_FREE2D(result            );
  VIO_FREE2D(unit_vec          );
  VIO_FREE2D(x                 );
  VIO_FREE2D(y                 );
  VIO_FREE2D(z                 );
  VIO_FREE2D(nz                );
  VIO_FREE2D(y_on_z            );
  VIO_FREE2D(ortho_y           );

  FREE(ang);
  FREE(tmp);

  return(TRUE);
}


/*
   function getparams will get the trnsform  parameters from a 
   transformation matrix 'tran' (that has already had the translation 
   componants removed).
   in description below, I assume that tran is a forward xform, from 
   native space to talairach space.
   I assume that trans is a 4x4 homogeneous matrix, with the 
   principal axis stored in the upper left 3x3 matrix.  The first col
   of tran represents is the coordinate of (1,0,0)' mapped through the
   transformation.  (this means  vec2 = tran * vec1).
  
   trans = [scale][shear][rot]
         = [scale][shear][rz][ry][rx];

   the shear matrix is constrained to be of the form:
     shear = [1 0 0 0
              f 1 0 0
              g h 1 0
              0 0 0 1];
     where f,g,h can take on any value.

   the scale matrix is constrained to be of the form:
     scale = [sx 0  0  0
              0  sy 0  0
              0  0  sz 0
              0  0  0  1];
     where sx,sy,sz must be positive.
   
  all rotations are assumed to be in the range -pi/2..pi/2

  the rotation angles are returned as radians and are applied 
  counter clockwise, when looking down the axis (from the positive end
  towards the origin).

  trans is assumed to be invertible.

  i assume a coordinate system:
             ^ y
             |
             |
             |
             |_______> x
            /
           /
          /z  (towards the viewer).

  
  
  procedure: 
          start with t = inv(tran) (after removing translation from tran )
  
          t = inv(r)*inv(sh)*inv(sc)
                  t maps the talairach space unit vectors into native
                space, which means that the columns of T are the
                direction cosines of these vectors x,y,z

   (1)  the length of the vectors x,y,z give the inverse scaling
        parameters:
             sx = 1/norm(x); sy = 1/norm(y); sz = 1/norm(z);

   (2)  with the constraints on the form of sh above, inv(sh) has the
        same form.  let inv(sh) be as above in terms of a,b and c.
          inv(sh) = [1 0 0 0; a 1 0 0; b c 1 0; 0 0 0 1];

        for c: project y onto z and normalize:

                  /     |y.z|^2     \(1/2)
             c = <  ---------------  >
                  \ |y|^2 - |y.z|^2 /

        for b: project x onto z and normalize

                  /        |x.z|^2        \(1/2)
             b = <  ---------------------  >
                  \ |x|^2 - |x.z|^2 - a^2 /
     
          where a is the projection of x onto the coordinate sys Y axis.

        for a: project x onto z and normalize
           a is taken from (b) above, and normalized... see below

  (3) rots are returned by getrots by giving the input transformation:
        rot_mat = [inv(sh)][inv(sc)][trans]

  (4) once completed, the parameters of sx,sy,sz and a,b,c are
      adjusted so that they maintain the matrix contraints above.


*/



VIO_BOOL extract2_parameters_from_matrix_quater(VIO_Transform *trans,
                                                      double *center,
                                                      double *translations,
                                                      double *scales,
                                                      double *shears,
                                                      double *quaternions)
{
  int 
    i,j;

  float 
    n1,n2,
    magnitude, magz, magx, magy, ai,bi,ci,scalar,a1,
    **center_of_rotation,
    **result,
    **unit_vec,
    *ang,*tmp,**x,**y,**z, **nz, **y_on_z, **ortho_y,
    **xmat,**T,**Tinv,**C,**Sinv,
    **R,**SR,**SRinv,**Cinv,**TMP1,**TMP2;

  VIO_ALLOC2D(xmat  ,5,5); nr_identf(xmat ,1,4,1,4);
  VIO_ALLOC2D(TMP1  ,5,5); nr_identf(TMP1 ,1,4,1,4);
  VIO_ALLOC2D(TMP2  ,5,5); nr_identf(TMP2 ,1,4,1,4);
  VIO_ALLOC2D(Cinv  ,5,5); nr_identf(Cinv ,1,4,1,4);
  VIO_ALLOC2D(SR    ,5,5); nr_identf(SR   ,1,4,1,4);
  VIO_ALLOC2D(SRinv ,5,5); nr_identf(SRinv,1,4,1,4);
  VIO_ALLOC2D(Sinv  ,5,5); nr_identf(Sinv ,1,4,1,4); 
  VIO_ALLOC2D(T     ,5,5); nr_identf(T    ,1,4,1,4);
  VIO_ALLOC2D(Tinv  ,5,5); nr_identf(Tinv ,1,4,1,4);
  VIO_ALLOC2D(C     ,5,5); nr_identf(C    ,1,4,1,4);
  VIO_ALLOC2D(R     ,5,5); nr_identf(R    ,1,4,1,4);

  VIO_ALLOC2D(center_of_rotation ,5,5);        /* make column vectors */
  VIO_ALLOC2D(result             ,5,5);
  VIO_ALLOC2D(unit_vec           ,5,5);
  VIO_ALLOC2D(x                  ,5,5);
  VIO_ALLOC2D(y                  ,5,5);
  VIO_ALLOC2D(z                  ,5,5);
  VIO_ALLOC2D(nz                 ,5,5);
  VIO_ALLOC2D(y_on_z             ,5,5);
  VIO_ALLOC2D(ortho_y            ,5,5);

  ALLOC(tmp ,4);
  ALLOC(ang ,4);

  for(i=0; i<=3; i++)        /* copy the input matrix */
    for(j=0; j<=3; j++)
      xmat[i+1][j+1] = (float)Transform_elem(*trans,i,j);
  
  /* -------------DETERMINE THE TRANSLATION FIRST! ---------  */

                                /* see where the center of rotation is displaced... */

  FILL_NR_COLVEC( center_of_rotation, center[0], center[1], center[2] );


  invertmatrix(4, xmat, TMP1);        /* get inverse of the matrix */

  matrix_multiply( 4, 4, 1, xmat, center_of_rotation, result); /* was TMP1 in place of xmat */

  SUB_NR_COLVEC( result, result, center_of_rotation );

  for(i=0; i<=2; i++) 
    translations[i] = result[i+1][1];

  /* -------------NOW GET THE SCALING VALUES! ----------------- */

  for(i=0; i<=2; i++) 
    tmp[i+1] = -translations[i];
  translation_to_homogeneous(3, tmp, Tinv); 

  for(i=0; i<=2; i++) 
    tmp[i+1] = center[i];
  translation_to_homogeneous(3, tmp, C); 
  for(i=0; i<=2; i++) 
    tmp[i+1] = -center[i];
  translation_to_homogeneous(3, tmp, Cinv); 


  matrix_multiply(4,4,4, xmat, C, TMP1);    /* get scaling*shear*rotation matrix */

 
  matrix_multiply(4,4,4, Tinv, TMP1, TMP1);


  matrix_multiply(4,4,4, Cinv, TMP1,    SR);

  invertmatrix(4, SR, SRinv);        /* get inverse of scaling*shear*rotation */
  
 
                                /* find each scale by mapping a unit vector backwards,
                                   and finding the magnitude of the result. */
  FILL_NR_COLVEC( unit_vec, 1.0, 0.0, 0.0 );
  matrix_multiply( 4, 4, 1, SRinv, unit_vec, result);
  magnitude = MAG_NR_COLVEC( result );
  if (magnitude != 0.0) {
    scales[0] = 1/magnitude;
    Sinv[1][1] = magnitude;
  }
  else {
    scales[0] = 1.0;
    Sinv[1][1] = 1.0;
  }

  FILL_NR_COLVEC( unit_vec, 0.0, 1.0, 0.0 );
  matrix_multiply( 4, 4, 1, SRinv, unit_vec, result);
  magnitude = MAG_NR_COLVEC( result );
  if (magnitude != 0.0) {
    scales[1] = 1/magnitude;
    Sinv[2][2] = magnitude;
  }
  else {
    scales[1]  = 1.0;
    Sinv[2][2] = 1.0;
  }

  FILL_NR_COLVEC( unit_vec, 0.0, 0.0, 1.0 );
  matrix_multiply( 4, 4, 1, SRinv, unit_vec, result);
  magnitude = MAG_NR_COLVEC( result );
  
  if (magnitude != 0.0) {
    scales[2] = 1/magnitude;
    Sinv[3][3] = magnitude;
  }
  else {
    scales[2] = 1.0;
    Sinv[3][3] = 1.0;
  }

  /* ------------NOW GET THE SHEARS, using the info from above ----- */

  /* SR contains the [scale][shear][rot], must multiply [inv(scale)]*SR
     to get shear*rot. */

                                /* make [scale][rot] */
  matrix_multiply(4,4, 4, Sinv, SR,  TMP1);

                                /* get inverse of [scale][rot] */
  invertmatrix(4, TMP1, SRinv);        


  FILL_NR_COLVEC(x, SRinv[1][1], SRinv[2][1], SRinv[3][1]);
  FILL_NR_COLVEC(y, SRinv[1][2], SRinv[2][2], SRinv[3][2]);
  FILL_NR_COLVEC(z, SRinv[1][3], SRinv[2][3], SRinv[3][3]);



                                /* get normalized z direction  */
  magz = MAG_NR_COLVEC(z);
  SCALAR_MULT_NR_COLVEC( nz, z, 1/magz );

                                /* get a direction perpendicular 
                                   to z, in the yz plane.  */
  scalar = DOTSUM_NR_COLVEC(  y, nz );
  SCALAR_MULT_NR_COLVEC( y_on_z, nz, scalar );

  SUB_NR_COLVEC( result, y, y_on_z ); /* result = y - y_on_z */
  scalar = MAG_NR_COLVEC( result);     /* ortho_y = result ./ norm(result)  */
  SCALAR_MULT_NR_COLVEC( ortho_y, result, 1/scalar);


                                /* GET C for the skew matrix */

  scalar = DOTSUM_NR_COLVEC( y, nz ); /* project y onto z */
  magy   = MAG_NR_COLVEC(y);
  ci = scalar / sqrt((double)( magy*magy - scalar*scalar)) ;
                                /* GET B for the skew matrix */

                                /*    first need a1 */

  a1     = DOTSUM_NR_COLVEC( x, ortho_y ); /* project x onto ortho_y */
  magx   = MAG_NR_COLVEC(x);

                                /*    now get B  */

  scalar = DOTSUM_NR_COLVEC( x, nz );
  bi = scalar / sqrt((double)( magx*magx - scalar*scalar - a1*a1)) ;

                                /* GET A for skew matrix  */

  ai = a1 / sqrt((double)( magx*magx - scalar*scalar - a1*a1));

                                /* normalize the inverse shear parameters.
                                   so that there is no scaling in the matrix 
                                   (all scaling is already accounted for 
                                   in sx,sy and sz. */

  n1 = sqrt((double)(1 + ai*ai + bi*bi));
  n2 = sqrt((double)(1 + ci*ci));

  ai = ai / n1;
  bi = bi / n1;
  ci = ci / n2;

                                /* ai,bi,c1 now contain the parameters for 
                                   the inverse NORMALIZED shear matrix 
                                   (i.e., norm(col_i) = 1.0). */

  
  /* ------------NOW GET THE ROTATION ANGLES!----- */

                                  /*  since SR = [scale][shear][rot], then
                                    rot = [inv(shear)][inv(scale)][SR] */

  nr_identf(TMP1 ,1,4,1,4);        /* make inverse scale matrix */
  TMP1[1][1] = 1/scales[0];
  TMP1[2][2] = 1/scales[1];
  TMP1[3][3] = 1/scales[2];

  nr_identf(TMP2 ,1,4,1,4);        /* make_inverse normalized shear matrix */
  TMP2[1][1] = sqrt((double)(1 - ai*ai - bi*bi));
  TMP2[2][2] = sqrt((double)(1 - ci*ci));
  TMP2[2][1] = ai;
  TMP2[3][1] = bi;
  TMP2[3][2] = ci;


                                /* extract rotation matrix */
  matrix_multiply(4,4, 4, TMP2, TMP1, T);
  matrix_multiply(4,4, 4, T,    SR,   R);

 
                                /* get rotation angles */
  extract_quaternions(R,quaternions);

  /* ------------NOW ADJUST THE SCALE AND SKEW PARAMETERS ------------ */

  invertmatrix(4, T, Tinv);        /* get inverse of the matrix */
  

  scales[0] = Tinv[1][1];
  scales[1] = Tinv[2][2];
  scales[2] = Tinv[3][3];
  shears[0] = Tinv[2][1]/scales[1] ;
  shears[1] = Tinv[3][1]/scales[2] ;
  shears[2] = Tinv[3][2]/scales[2] ;
  

  VIO_FREE2D(xmat);
  VIO_FREE2D(TMP1);
  VIO_FREE2D(TMP2);
  VIO_FREE2D(Cinv);
  VIO_FREE2D(SR  );
  VIO_FREE2D(SRinv);
  VIO_FREE2D(Sinv);
  VIO_FREE2D(T   );
  VIO_FREE2D(Tinv);
  VIO_FREE2D(C   );
  VIO_FREE2D(R   );
  
  VIO_FREE2D(center_of_rotation);
  VIO_FREE2D(result            );
  VIO_FREE2D(unit_vec          );
  VIO_FREE2D(x                 );
  VIO_FREE2D(y                 );
  VIO_FREE2D(z                 );
  VIO_FREE2D(nz                );
  VIO_FREE2D(y_on_z            );
  VIO_FREE2D(ortho_y           );

  FREE(ang);
  FREE(tmp);

  return(TRUE);
}




#define EPS  0.00000000001        /* epsilon, should be calculated */

/*
    this routine extracts the rotation angles from the rotation
    matrix.  The rotation matrix is assumed to be a 3x3 matrix in
    zero offset form [1..3][1..3].  It is locally copied into a 
    4x4 homogeneous matrix for manipulation.

    we assume that the matrix rotation center is (0,0,0).
    we assume that the application of this matrix to a vector
        is done with rot_mat*vec = premultiplication by matrix

    the resulting angles rx=ang[1],ry=ang[2],rz=ang[3], follow
    the following assumptions:

    -all rotations are assumed to be in the range -pi/2..pi/2
    routine returns FALSE is this is found not to hold
    -rotations are applied 1 - rx, 2 - ry and 3 - rz
    -applying these rotations to an identity matrix will
    result in a matrix equal to `rot' (the input matrix)
    -positive rotations are counter-clockwise when looking
    down the axis, from the positive end towards the origin.
    -I assume a coordinate system:
                ^ Y
                |
                |
                |
                |
                +---------> X
                /
            /
            / Z  (towards the viewer).

    -The problem is posed as:  
        given a rotation matrix ROT, determine the rotations
        rx,ry,rz applied in order to give ROT
    solution:
        assume the rot matrix is equivalent to a normalized
        orthogonal local coord sys.  i.e.  row 1 of ROT is
        the local x direction, row 2 is the local y and row 3
        is the local z.
    
        (note local is lower case, world is UPPER)

        1- find RZ that brings local x into the XZ plane
        2- find RY that brings local x*RZ onto X
        3- find RX that brings local y*RZ*RY onto Y

        the required rotations are -RX,-RY and -RZ!
 */

VIO_BOOL rotmat_to_ang(float **rot, float *ang)
{

   float 
      rx,ry,rz,
      **t,**s,
      **R,
      **Rx,
      **Ry,
      **Rz,
      len,
      i,j,k;

   int
      m,n;

   VIO_ALLOC2D(t  ,5,5);        /* make two column vectors */
   VIO_ALLOC2D(s  ,5,5);

   VIO_ALLOC2D(R  ,5,5);        /* working space matrices */
   VIO_ALLOC2D(Rx ,5,5);
   VIO_ALLOC2D(Ry ,5,5);
   VIO_ALLOC2D(Rz ,5,5);

   nr_identf(R,1,4,1,4);        /* init R homogeneous matrix */

   for (m=1; m<=3; ++m)                /* copy rot matrix into R */
      for (n=1; n<=3; ++n)
         R[m][n] = rot[m][n];
   
/* ---------------------------------------------------------------
   step one,  find the RZ rotation reqd to bring 
              the local x into the world XZ plane
*/

   for (m=1; m<=3; ++m)                /* extract local x vector, ie the first column */
      t[m][1] = R[m][1];
   t[4][1] = 1.0;

   i = t[1][1];                        /* make local vector componants */
   j = t[2][1]; 
   k = t[3][1];

   if (i<EPS) {                        /* if i is not already in the positive X range, */
      print("WARNING: (%s:%d) %s\n",__FILE__, __LINE__,"step one: rz not in the range -pi/2..pi/2");
      return(FALSE);
   }

   len = sqrt(i*i + j*j);        /* length of vect x on XY plane */
   if (fabs(len)<EPS) {
      print("WARNING: (%s:%d) %s\n",__FILE__, __LINE__,"step one: length of vect x null.");
      return(FALSE);
   }

   if (fabs(i)>fabs(j)) {
      rz = fabs(asin((double)(j/len)));
   }
   else {
      rz = fabs(acos((double)(i/len)));
   }

   if (j>0)                        /* what is the counter clockwise angle */
      rz = -rz;                 /* necessary to bring vect x ont XY plane? */
      
  
/*---------------------------------------------------------------
   step two:  find the RY rotation reqd to align 
              the local x on the world X axis 

  (since i was positive above, RY should already by in range -pi/2..pi/2 
  but we'll check it  anyway)                                             */

   for (m=1; m<=3; ++m)                /* extract local x vector */
      t[m][1] = R[m][1];
   t[4][1] = 1.0;

   nr_rotzf(Rz,rz);             /* create the rotate Z matrix */
 
   nr_multf(Rz,1,4,1,4,  t,1,4,1,1,   s);   /* apply RZ, to get x in XZ plane */

   i = s[1][1];                        /* make local vector componants */
   j = s[2][1]; 
   k = s[3][1];

   if (i<EPS) {
      print("WARNING: (%s:%d) %s\n",__FILE__, __LINE__,"step two: ry not in the range -pi/2..pi/2");
      return(FALSE);
   }

   len = sqrt(i*i + k*k);                /* length of vect x in XZ plane, after RZ */

   if (fabs(len)<EPS) {
      print("WARNING: (%s:%d) %s\n",__FILE__, __LINE__,"step two: length of vect z null.");
      return(FALSE);
   }

   if (fabs(i)>fabs(k)) {
      ry = fabs(asin((double)(k/len)));
   }
   else {
      ry = fabs(acos((double)(i/len)));
   }

   /*    what is the counter clockwise angle necessary to bring  */
   /*    vect x onto X? */
   if (k < 0) { 
      ry = -ry;
   }

   /*--------------------------------------------------------------- */
   /*   step three,rotate around RX to */
   /*              align the local y with Y and z with Z */

   for (m=1; m<=3; ++m)                /* extract local z vector */
      t[m][1] = R[m][3];
   t[4][1] = 1.0;

   nr_rotyf(Ry,ry);             /* create the rotate Y matrix */

                                /* t =  roty(ry*180/pi) *(rotz(rz*180/pi) *r(3,:)); */
   nr_multf(Rz,1,4,1,4,  t,1,4,1,1,  s); /* apply RZ, to get x in XZ plane */
   nr_multf(Ry,1,4,1,4,  s,1,4,1,1,  t); /* apply RY, to get x onto X      */

   i = t[1][1];                        /* make local vector componants */
   j = t[2][1]; 
   k = t[3][1];

   len = sqrt(j*j + k*k);        /* length of vect x in Y,Z plane */

   if (fabs(len)<EPS) {
      print("WARNING: (%s:%d) %s\n",__FILE__, __LINE__,"step three: length of vect z null.");
      return(FALSE);
   }

   if (fabs(k)>fabs(j)) {
      rx = fabs(asin((double)(j/len)));
   }
   else {
      rx = fabs(acos((double)(k/len)));
   }

   if (j< 0) { 
      rx = -rx;
   }
        
   rx = -rx;  /* these are the required rotations */
   ry = -ry;
   rz = -rz;

   ang[1] = rx;
   ang[2] = ry;
   ang[3] = rz;

   VIO_FREE2D(t);
   VIO_FREE2D(s);
   VIO_FREE2D(R);
   VIO_FREE2D(Rx);
   VIO_FREE2D(Ry);
   VIO_FREE2D(Rz);

   return(TRUE);
}



/* (c) Copyright 1993, 1994, Silicon Graphics, Inc.
 * ALL RIGHTS RESERVED
 * Permission to use, copy, modify, and distribute this software for
 * any purpose and without fee is hereby granted, provided that the above
 * copyright notice appear in all copies and that both the copyright notice
 * and this permission notice appear in supporting documentation, and that
 * the name of Silicon Graphics, Inc. not be used in advertising
 * or publicity pertaining to distribution of the software without specific,
 * written prior permission.
 *
 * THE MATERIAL EMBODIED TRUE THIS SOFTWARE IS PROVIDED TO YOU "AS-IS"
 * AND WITHOUT WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR OTHERWISE,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTY OF MERCHANTABILITY OR
 * FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL SILICON
 * GRAPHICS, INC.  BE LIABLE TO YOU OR ANYONE ELSE FOR ANY DIRECT,
 * SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY
 * KIND, OR ANY DAMAGES WHATSOEVER, INCLUDING WITHOUT LIMITATION,
 * LOSS OF PROFIT, LOSS OF USE, SAVINGS OR REVENUE, OR THE CLAIMS OF
 * THIRD PARTIES, WHETHER OR NOT SILICON GRAPHICS, INC.  HAS BEEN
 * ADVISED OF THE POSSIBILITY OF SUCH LOSS, HOWEVER CAUSED AND TRUE
 * ANY THEORY OF LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE
 * POSSESSION, USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * US Government Users Restricted Rights
 * Use, duplication, or disclosure by the Government is subject to
 * restrictions set forth in FAR 52.227.19(c)(2) or subparagraph
 * (c)(1)(ii) of the Rights in Technical Data and Computer Software
 * clause at DFARS 252.227-7013 and/or in similar or successor
 * clauses in the FAR or the DOD or NASA FAR Supplement.
 * Unpublished-- rights reserved under the copyright laws of the
 * United States.  Contractor/manufacturer is Silicon Graphics,
 * Inc., 2011 N.  Shoreline Blvd., Mountain View, CA 94039-7311.
 *
 * OpenGL(TM) is a trademark of Silicon Graphics, Inc.
 * Original code from:
 * David M. Ciemiewicz, Mark Grossman, Henry Moreton, and Paul Haeberli
 *
 * Much mucking with by:
 * Gavin Bell
 *
 * And more mucking with by:
 * Andrew Janke, Patricia Le Nezet
 *
 *
 *
 *You can find some documentions about quaternions in  /data/web/users/lenezet/QUATERNIONS
 *
 *
 *
 *
 */

#define SQR(a) (a)*(a)
#define cube(a) (a)*(a)*(a)

/* copy a vector */
void vcopy(double *copy, double *v){
   copy[0] = v[0];
   copy[1] = v[1];
   copy[2] = v[2];
   }

/* compute the addition of two vectors */
void vadd(double *add, double *v1, double *v2){
   add[0] = v1[0] + v2[0];
   add[1] = v1[1] + v2[1];
   add[2] = v1[2] + v2[2];
   }

/* multiply a vector by a constant */
void vscale(double *v, double scale){
   v[0] *= scale;
   v[1] *= scale;
   v[2] *= scale;
   }

/*computes the vector cross product of two vectors */
void vcross(double *cross, double *v1, double *v2){
   cross[0] = (v1[1] * v2[2]) - (v1[2] * v2[1]);
   cross[1] = (v1[2] * v2[0]) - (v1[0] * v2[2]);
   cross[2] = (v1[0] * v2[1]) - (v1[1] * v2[0]);
   }
   
/* returns the vector dot product of two vectors */
double vdot(double *v1, double *v2){
   return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
   }

/* returns the euc length of a vector */
double vlength(double *v){
   return sqrt(SQR(v[0]) + SQR(v[1]) + SQR(v[2]));
   }

/* returns the euc distance between two points */
double veuc(double *p1, double *p2){
   return sqrt(SQR(p2[0]-p1[0]) +
               SQR(p2[1]-p1[1]) +
               SQR(p2[2]-p1[2]));
   }


/* normalise a vector */
void vnormal(double *v){
   vscale(v, 1.0 / vlength(v));
   }


/* Given an axis and angle, compute quaternion */
void axis_to_quat(double vec[3], double phi, double quat[4]){
   vnormal(vec);
   vcopy(quat, vec);
   vscale(quat, sin(phi/2.0));
   quat[3] = cos(phi/2.0);
   }


/* Given an quaternion compute an axis and angle */
void quat_to_axis(double vec[3], double *phi, double quat[4]){
   double scale;
   double eps=0.00001;

   scale = quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2];
   
   if(scale < eps){  /* no rotation, we're stuffed */
      vec[0] = 1;
      vec[1] = 0;
      vec[2] = 0;
      *phi = 0;
      }
   else{
      vcopy(vec, quat);
      vscale(vec, 1.0/scale);
      vnormal(vec);
      
      *phi  = 2.0*acos(quat[3]);
      }
   }


/* Quaternions always obey:  a^2 + b^2 + c^2 + d^2 = 1.0
 * If they don't add up to 1.0, dividing by their magnitued will
 * renormalize them.
 *
 * Note: See the following for more information on quaternions:
 *
 * - Shoemake, K., Animating rotation with quaternion curves, Computer
 *   Graphics 19, No 3 (Proc. SIGGRAPH'85), 245-254, 1985.
 * - Pletinckx, D., Quaternion calculus as a basic tool in computer
 *   graphics, The Visual Computer 5, 2-13, 1989.
 */
static void normalize_quat(double q[4]){
   int i;
   double mag;

   mag = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
   for(i = 0; i < 4; i++){
      q[i] /= mag;
      }
   }




/* Given two rotations, e1 and e2, expressed as quaternion rotations,
 * figure out the equivalent single rotation and stuff it into dest.
 *
 * This routine also normalizes the result every RENORMCOUNT times it is
 * called, to keep error from creeping in.
 *
 * NOTE: This routine is written so that q1 or q2 may be the same
 * as dest (or each other).
 */


#define RENORMCOUNT 97
void add_quats(double q1[4], double q2[4], double dest[4]){
   static int count=0;
   double t1[4], t2[4], t3[4];
   double tf[4];

   vcopy(t1, q1);
   vscale(t1,q2[3]);

   vcopy(t2, q2);
   vscale(t2,q1[3]);

   vcross(t3,q2,q1);
   vadd(tf,t1,t2);
   vadd(tf,t3,tf);
   tf[3] = q1[3] * q2[3] - vdot(q1,q2);

   dest[0] = tf[0];
   dest[1] = tf[1];
   dest[2] = tf[2];
   dest[3] = tf[3];

   if (++count > RENORMCOUNT) {
      count = 0;
      normalize_quat(dest);
      }
   }
/* Build a rotation matrix, given a quaternion rotation. */
/* this is a more general form to the original           */

void build_rotmatrix(float **m, double *quat){
 
 
  normalize_quat(quat);

  m[1][1] = SQR(quat[3]) + SQR(quat[0]) - SQR(quat[1]) - SQR(quat[2]);
  m[1][2] = 2.0 * (quat[0]*quat[1] - quat[2]*quat[3]);
  m[1][3] = 2.0 * (quat[2]*quat[0] + quat[1]*quat[3]);
  m[1][4] = 0.0;
  
  m[2][1] = 2.0 * (quat[0]*quat[1] + quat[2]*quat[3]);
  m[2][2] = SQR(quat[3]) - SQR(quat[0]) + SQR(quat[1]) - SQR(quat[2]);
  m[2][3] = 2.0*(quat[1]*quat[2] - quat[0]*quat[3]);
  m[2][4] = 0.0;
  
  m[3][1] = 2.0 * (quat[2]*quat[0] - quat[1]*quat[3]);
  m[3][2] = 2.0 * (quat[1]*quat[2] + quat[0]*quat[3]);
  m[3][3] = SQR(quat[3]) - SQR(quat[0]) - SQR(quat[1]) + SQR(quat[2]);
  m[3][4] = 0.0;
  
  m[4][1] = 0.0;
  m[4][2] = 0.0;
  m[4][3] = 0.0;
  m[4][4] = 1.0;

 }


/* from a rotation matrix this program give a quaternion associate to the rotation */

void extract_quaternions(float **m, double *quat){
 double max,indice;
 double a[4];
 int i;

 a[0] = 1 + m[1][1] - m[2][2] - m[3][3]; 
 a[1] = 1 - m[1][1] - m[2][2] + m[3][3];
 a[2] = 1 - m[1][1] + m[2][2] - m[3][3];
 a[3] = 1 + m[1][1] + m[2][2] + m[3][3];


 max =a[0];
 indice = 0;
 for(i=1; i<4; i++)
   if(a[i]>max){max=a[i]; indice=i;}
   

 if(indice==0)
   {
   quat[0] = (double) sqrt(fabs(a[0]))/2;
   max = 4*quat[0];
   quat[1] =(double) (m[1][2] + m[2][1])/max;
   quat[2] =(double) (m[3][1] + m[1][3])/max;
   quat[3] =(double) (m[3][2] - m[2][3])/max;
   }
   
 if(indice==1)
   {
   quat[1] = (double) sqrt(fabs(a[1]))/2;
   max = 4*quat[1];
   quat[0] = (double)(m[2][1] + m[1][2])/max;
   quat[3] = (double)(m[1][3] - m[3][1])/max;
   quat[2] = (double)(m[2][3] + m[3][2])/max;
   }
   
   
 if(indice==2)
   {
   quat[2] = (double) sqrt(fabs(a[2]))/2;
   max = 4*quat[2];
   quat[3] = (double) (m[2][1] - m[1][2])/max;
   quat[0] = (double) (m[3][1] + m[1][3])/max;
   quat[1] = (double) (m[2][3] + m[3][2])/max;
   }
   
 if(indice==3)
   {
   quat[3] = (double) sqrt(fabs(a[3]))/2;
   max = 4*quat[3];
   quat[2] = (double) (m[2][1] - m[1][2])/max;
   quat[1] = (double) (m[1][3] - m[3][1])/max;
   quat[2] = (double) (m[3][2] - m[2][3])/max;
   }

}
