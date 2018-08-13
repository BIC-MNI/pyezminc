#  Copyright 2013, Haz-Edine Assemlal

#  This file is part of PYEZMINC.
# 
#  PYEZMINC is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, version 2.
# 
#  PYEZMINC is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with PYEZMINC.  If not, see <http://www.gnu.org/licenses/>.

# cython: c_string_type=unicode, c_string_encoding=utf8

from cython.operator cimport dereference as deref, preincrement as inc #dereference and increment operators
from libcpp cimport bool, string, vector


cdef extern from "string" namespace "std":
    cdef cppclass string:
        char* c_str()

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        vector()
        void push_back(T&)
        T& operator[](int)
        T& at(int)
        iterator begin()
        iterator end()

cdef extern from "netcdf.h" :
    ctypedef int nc_type

    
cdef extern from "minc_1_rw.h" namespace "minc":

    # MINC dimension space definition
    cdef enum dimensions: 
        DIM_UNKNOWN=0,DIM_X,DIM_Y,DIM_Z,DIM_TIME,DIM_VEC

    # MINC dimension info
    cdef cppclass dim_info:
        dim_info(int l, double sta,double spa,dimensions d,bool have_dir_cos)
        dim_info(int l, double sta,double spa,dimensions d)
        size_t length
        double step,start
        bool have_dir_cos
        double dir_cos[3]
        string name
        dimensions  dim

    # MINC volume dimensions
    ctypedef vector[dim_info] minc_info

    # MINC base class for reading/writing
    cdef cppclass minc_1_base:
        minc_1_base()
        void close() except +
        nc_type datatype()
        string history()
        bool is_minc2()
        bool is_signed()
        int dim_no()
        int ndim(int i)
        double nstart(int i)
        double nspacing(int i)
        double ndir_cos(int i, int j)
        bool have_dir_cos(int i)

        int var_id(const char *var_name) 
        long var_length(const char *var_name) 
        long var_length(int var_id) 
        int var_number() 
        string var_name(int no) 
        int att_number(const char *var_name) 
        int att_number(int var_no) 
        string att_name(const char *var_name,int no) 
        string att_name(int varid,int no) 
        
        string att_value_string(const char *var_name,const char *att_name) 
        string att_value_string(int varid,const char *att_name) 
        
        vector[double] att_value_double(const char *var_name,const char *att_name) 
        vector[int] att_value_int(const char *var_name,const char *att_name) 
        vector[short] att_value_short(const char *var_name,const char *att_name) 
        vector[unsigned char] att_value_byte(const char *var_name,const char *att_name) 
        
        vector[double] att_value_double(int varid,const char *att_name) 
        vector[int] att_value_int(int varid,const char *att_name) 
        vector[short] att_value_short(int varid,const char *att_name) 
        vector[unsigned char] att_value_byte(int varid,const char *att_name) 

        nc_type att_type(char *var_name,char *att_name) 
        nc_type att_type(int varid,char *att_name) 
        int att_length(char *var_name,char *att_name) 
        int att_length(int varid,char *att_name) 

        int create_var_id(const char *varname)
        void insert(const char *varname,const char *attname,double val)
        void insert(const char *varname,const char *attname,const char* val)
        void insert(const char *varname,const char *attname,const vector[double] &val)
        void insert(const char *varname,const char *attname,const vector[int] &val)
        void insert(const char *varname,const char *attname,const vector[short] &val)
        void insert(const char *varname,const char *attname,const vector[unsigned char] &val)
        
        minc_info info()

    # MINC reader
    cdef cppclass minc_1_reader:
        minc_1_reader() except +
        minc_1_reader(minc_1_reader) except +
        void setup_read_double() except +
        void setup_read_float() except +
        void setup_read_int() except +
        void setup_read_uint() except +
        void setup_read_short() except +
        void setup_read_ushort() except +
        void setup_read_byte() except +
        void open(const char *path, bool positive_directions, bool metadate_only, bool rw) except +
        void open(const char *path, bool positive_directions, bool metadate_only) except +
        void open(const char *path, bool positive_directions) except +
        void open(const char *path) except +
        minc_info info()

    # MINC writer
    cdef cppclass minc_1_writer:
        minc_1_writer() except +
        minc_1_writer(minc_1_writer) except +
        void setup_write_double() except +
        void setup_write_float() except +
        void setup_write_int() except +
        void setup_write_short() except +
        void setup_write_uint() except +
        void setup_write_ushort() except +
        void setup_write_byte() except +
        void open(const char *path, minc_1_reader imitate) except +
        void open(const char *path, minc_1_base imitate) except +
        void open(const char *path, char *imitate_file) except +
        void open(const char *path, const minc_info &info,int slice_dimensions,nc_type datatype,bool signed) except +
        void open(const char *path) except +
        void copy_headers(const minc_1_base &src) except +
        void copy_headers(const minc_1_reader &src) except +
        void append_history(const char *append_history) except +
        minc_info info()

cdef extern from "minc_1_iterators.h" namespace "minc":

    # MINC input iterator
    cdef cppclass minc_input_iterator[T]:
        vector[long] cur() const
        minc_input_iterator()
        minc_input_iterator(minc_1_reader&) except +
        minc_input_iterator(minc_input_iterator&) except +
        attach(minc_1_reader&) except +
        bool next() except +
        bool last() except +
        void begin() except +
        const T& value()
        double progress()

    # MINC output iterator
    cdef cppclass minc_output_iterator[T]:
        vector[long] cur() const
        minc_output_iterator()
        minc_output_iterator(minc_1_writer&) except +
        minc_output_iterator(minc_output_iterator&) except +
        attach(minc_1_writer&) except +
        bool next() except +
        bool last() except +
        void begin() except +
        void value(const T&)
        double progress()

    
    cdef cppclass minc_parallel_input_iterator:
        bool next()
        bool last()
        void begin()
        bool have_mask()
        bool mask_value()
        void value(vector[double]& v)
        void value(double* v)
        void open(const vector[string] &in_files,const string & mask_file)  except +
        void open(const vector[string] &in_files)  except +
        size_t dim()
        double progress()

    cdef cppclass minc_parallel_output_iterator:
        bool next()
        bool last()
        void begin()
        void value(const vector[double]& v)
        void value(const double* v)
        void open(const vector[string] &out_files,const minc_info & output_info,const char* history)  except +
        void open(const vector[string] &out_files,const minc_info & output_info) except +
        size_t dim()
        double progress()

    cdef void load_standard_volume(minc_1_reader& rw, double *vol) except +
    cdef void load_standard_volume(minc_1_reader& rw, float *vol) except +
    cdef void load_standard_volume(minc_1_reader& rw, int *vol) except +
    cdef void load_standard_volume(minc_1_reader& rw, short *vol) except +
    cdef void load_standard_volume(minc_1_reader& rw, unsigned int *vol) except +
    cdef void load_standard_volume(minc_1_reader& rw, unsigned short *vol) except +
    cdef void load_standard_volume(minc_1_reader& rw, unsigned char *vol) except +

    cdef void save_standard_volume(minc_1_writer& rw, double *vol) except +
    cdef void save_standard_volume(minc_1_writer& rw, float *vol) except +
    cdef void save_standard_volume(minc_1_writer& rw, int *vol) except +
    cdef void save_standard_volume(minc_1_writer& rw, unsigned int *vol) except +
    cdef void save_standard_volume(minc_1_writer& rw, short *vol) except +
    cdef void save_standard_volume(minc_1_writer& rw, unsigned short *vol) except +
    cdef void save_standard_volume(minc_1_writer& rw, unsigned char *vol) except +

cdef extern from "minc.h":
    pass
    
cdef extern from "minc2.h":
    pass

cdef extern from "volume_io.h" :
    ctypedef int VIO_BOOL 
    ctypedef const char * VIO_STR
    ctypedef double VIO_Transform_elem_type
    ctypedef void * VIO_Volume
    
    ctypedef enum VIO_Status:
        VIO_OK=0, VIO_ERROR, VIO_INTERNAL_ERROR, VIO_END_OF_FILE,VIO_QUIT
    
    ctypedef enum VIO_Transform_types:
        LINEAR, THIN_PLATE_SPLINE, USER_TRANSFORM, CONCATENATED_TRANSFORM, GRID_TRANSFORM

    ctypedef struct VIO_General_transform:
        VIO_Transform_types  type
        VIO_BOOL             inverse_flag
        VIO_STR              displacement_volume_file 
    
    ctypedef struct VIO_Transform:
        VIO_Transform_elem_type    m[4][4]

    VIO_Status input_transform_file(VIO_STR filename,VIO_General_transform *tfm)
    VIO_Transform_types get_transform_type(VIO_General_transform   *transform )
    VIO_Transform  * get_linear_transform_ptr(VIO_General_transform   *transform)
    int get_n_concated_transforms(VIO_General_transform   *transform)
    VIO_General_transform  *get_nth_general_transform( VIO_General_transform   *transform, int n )
    void  delete_general_transform(VIO_General_transform   *transform )
    VIO_Status  output_transform_file(VIO_STR filename, VIO_STR comments, VIO_General_transform   *transform )
    void  concat_general_transforms(VIO_General_transform   *first, VIO_General_transform   *second, VIO_General_transform   *result )
    void  create_grid_transform_no_copy(
        VIO_General_transform    *transform,
        VIO_Volume               displacement_volume,
        VIO_STR                  displacement_volume_file )
    void  create_linear_transform(
       VIO_General_transform   *transform,
       VIO_Transform           *linear_transform )


cdef extern from "xfm_param.h" :
    int matrix_extract_linear_param(const double *in_matrix,
                                 double *center,
                                 double *translations,
                                 double *scales,
                                 double *shears,
                                 double *rotations)

    int linear_param_to_matrix(double *out_matrix,
                               const double *center,
                               const double *translations,
                               const double *scales,
                               const double *shears,
                               const double *rotations)

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
