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

        int var_id(char *var_name) 
        long var_length(char *var_name) 
        long var_length(int var_id) 
        int var_number() 
        string var_name(int no) 
        int att_number(char *var_name) 
        int att_number(int var_no) 
        string att_name(char *var_name,int no) 
        string att_name(int varid,int no) 
        
        string att_value_string(char *var_name,char *att_name) 
        string att_value_string(int varid,char *att_name) 
        
        vector[double] att_value_double(char *var_name,char *att_name) 
        vector[int] att_value_int(char *var_name,char *att_name) 
        vector[short] att_value_short(char *var_name,char *att_name) 
        vector[unsigned char] att_value_byte(char *var_name,char *att_name) 
        
        vector[double] att_value_double(int varid,char *att_name) 
        vector[int] att_value_int(int varid,char *att_name) 
        vector[short] att_value_short(int varid,char *att_name) 
        vector[unsigned char] att_value_byte(int varid,char *att_name) 

        nc_type att_type(char *var_name,char *att_name) 
        nc_type att_type(int varid,char *att_name) 
        int att_length(char *var_name,char *att_name) 
        int att_length(int varid,char *att_name) 

        int create_var_id(char *varname)
        void insert(char *varname,char *attname,double val)
        void insert(char *varname,char *attname,char* val)
        void insert(char *varname,char *attname,vector[double] &val)
        void insert(char *varname,char *attname,vector[int] &val)
        void insert(char *varname,char *attname,vector[short] &val)
        void insert(char *varname,char *attname,vector[unsigned char] &val)
      
    cdef cppclass minc_1_reader:
        minc_1_reader() except +
        minc_1_reader(minc_1_reader) except +
        void setup_read_double() except +
        void setup_read_float() except +
        void setup_read_int() except +
        void open(char *path, bool positive_directions, bool metadate_only, bool rw) except +
        void open(char *path, bool positive_directions, bool metadate_only) except +
        void open(char *path, bool positive_directions) except +
        void open(char *path) except +

    cdef cppclass minc_1_writer:
        minc_1_writer() except +
        minc_1_writer(minc_1_writer) except +
        void setup_write_double() except +
        void setup_write_float() except +
        void setup_write_int() except +
        void open(char *path, minc_1_reader imitate) except +
        void open(char *path, minc_1_base imitate) except +
        void open(char *path, char *imitate_file) except +
        void open(char *path) except +
        void copy_headers(minc_1_base src) except +
        void copy_headers(minc_1_reader src) except +
        void append_history(char *append_history) except +

cdef extern from "minc_1_iterators.h" namespace "minc":

    cdef cppclass minc_input_iterator[T]:
        vector[long] cur() const
        minc_input_iterator() except +
        minc_input_iterator(minc_1_reader&) except +
        minc_input_iterator(minc_input_iterator&) except +
        attach(minc_1_reader&) except +
        bool next() except +
        bool last() except +
        void begin() except +
        const T& value()
        
    cdef cppclass minc_output_iterator[T]:
        vector[long] cur() const
        minc_output_iterator() except +
        minc_output_iterator(minc_1_writer&) except +
        minc_output_iterator(minc_output_iterator&) except +
        attach(minc_1_writer&) except +
        bool next() except +
        bool last() except +
        void begin() except +
        void value(const T&)

    cdef void load_standard_volume(minc_1_reader& rw, double *vol) except +
    cdef void load_standard_volume(minc_1_reader& rw, float *vol) except +
    cdef void load_standard_volume(minc_1_reader& rw, int *vol) except +

    cdef void save_standard_volume(minc_1_writer& rw, double *vol) except +
    cdef void save_standard_volume(minc_1_writer& rw, float *vol) except +
    cdef void save_standard_volume(minc_1_writer& rw, int *vol) except +

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
