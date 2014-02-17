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

cdef extern from "netcdf.h":
    ctypedef int nc_type

cdef extern from "minc_1_rw.h" namespace "minc":

    cdef enum dimensions: 
        DIM_UNKNOWN=0,DIM_X,DIM_Y,DIM_Z,DIM_TIME,DIM_VEC

    cdef cppclass dim_info:
        dim_info(int l, double sta,double spa,dimensions d,bool have_dir_cos)
        dim_info(int l, double sta,double spa,dimensions d)
        size_t length
        double step,start
        bool have_dir_cos
        double dir_cos[3]
        string name
        dimensions  dim
        
    ctypedef vector[dim_info] minc_info
    
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
      
    cdef cppclass minc_1_reader:
        minc_1_reader() except +
        minc_1_reader(minc_1_reader) except +
        void setup_read_double() except +
        void setup_read_float() except +
        void setup_read_int() except +
        void open(const char *path, bool positive_directions, bool metadate_only, bool rw) except +
        void open(const char *path, bool positive_directions, bool metadate_only) except +
        void open(const char *path, bool positive_directions) except +
        void open(const char *path) except +
        minc_info info()

    cdef cppclass minc_1_writer:
        minc_1_writer() except +
        minc_1_writer(minc_1_writer) except +
        void setup_write_double() except +
        void setup_write_float() except +
        void setup_write_int() except +
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
        
    cdef cppclass minc_parallel_input_iterator:
        bool next() except +
        bool last() except +
        void begin() except +
        bool have_mask()
        bool mask_value()
        void value(vector[double]& v)
        void value(double* v)
        void open(const vector[string] &in_files,const string & mask_file)
        void open(const vector[string] &in_files)
        size_t dim()
        
    cdef cppclass minc_parallel_output_iterator:
        bool next() except +
        bool last() except +
        void begin() except +
        void value(const vector[double]& v)
        void value(const double* v)
        void open(const vector[string] &out_files,const minc_info & output_info,const char* history)
        void open(const vector[string] &out_files,const minc_info & output_info)
        size_t dim()

    cdef void load_standard_volume(minc_1_reader& rw, double *vol) except +
    cdef void load_standard_volume(minc_1_reader& rw, float *vol) except +
    cdef void load_standard_volume(minc_1_reader& rw, int *vol) except +

    cdef void save_standard_volume(minc_1_writer& rw, double *vol) except +
    cdef void save_standard_volume(minc_1_writer& rw, float *vol) except +
    cdef void save_standard_volume(minc_1_writer& rw, int *vol) except +

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
