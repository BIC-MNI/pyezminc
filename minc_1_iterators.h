/* ----------------------------- MNI Header -----------------------------------
@NAME       : 
@DESCRIPTION: Simplified interator-based access to minc files, using minc_1_rw interface
@COPYRIGHT  :
              Copyright 2007 Vladimir Fonov, McConnell Brain Imaging Centre, 
              Montreal Neurological Institute, McGill University.
              Permission to use, copy, modify, and distribute this
              software and its documentation for any purpose and without
              fee is hereby granted, provided that the above copyright
              notice appear in all copies.  The author and McGill University
              make no representations about the suitability of this
              software for any purpose.  It is provided "as is" without
              express or implied warranty.
---------------------------------------------------------------------------- */
#ifndef __MINC_1_ITERATORS_H__
#define __MINC_1_ITERATORS_H__

#include "minc_1_rw.h"

namespace minc
{
  class minc_input_iterator_base
  {
  protected:
    mutable minc_1_reader* _rw;
    std::vector<long> _cur;
    bool _last;
    size_t _count;
    
  public:
    
    const std::vector<long>& cur(void) const
    {
      return _cur;
    }
    
    virtual void attach(minc_1_reader& rw)
    {
      _rw=&rw;
      _last=false;
      _count=0;
    }
    
    virtual bool next(void)=0;
    virtual void begin(void)=0;
    
    bool last(void)
    {
      return _last;
    }
    
    minc_input_iterator_base(): 
      _rw(NULL),_last(false),_count(0)
    {}
      
    minc_input_iterator_base(const minc_input_iterator_base& a): 
      _rw(a._rw),_cur(a._cur),_last(a._last),_count(a._count)
    {}
      
    minc_input_iterator_base(minc_1_reader& rw)
    {
      attach(rw);
    }
      
    virtual ~minc_input_iterator_base()
    {}
    
  };

  class minc_output_iterator_base
  {
  protected:
    mutable minc_1_writer* _rw;
    std::vector<long> _cur;
    bool _last;
    size_t _count;
  public:
    minc_output_iterator_base():
      _rw(NULL),_last(false),_count(0)
    {}
      
    minc_output_iterator_base(const minc_output_iterator_base& a):
    _rw(a._rw),_cur(a._cur),_last(a._last),_count(a._count)
    {}
      
    minc_output_iterator_base(minc_1_writer& rw)
    {
      attach(rw);
    }
    
    const std::vector<long>& cur(void) const
    {
      return _cur;
    }
      
    virtual void attach(minc_1_writer& rw)
    {
      _rw=&rw;
      _last=false;
      _count=0;
    }
    virtual bool next(void)=0;
    virtual void begin(void)=0;
    
    bool last(void)
    {
      return _last;
    }
    
    virtual ~minc_output_iterator_base()
    {}
      
  };
  
  template <class T> class minc_input_iterator:public minc_input_iterator_base
  {
      std::vector<T> _buf;
    public:
    
    minc_input_iterator(const minc_input_iterator<T>& a):
      minc_input_iterator_base(a)
    {
      //TODO: init buffer
    }
    
    minc_input_iterator(minc_1_reader& rw):minc_input_iterator_base(rw)
    {
    }
    
    minc_input_iterator()
    {
    }
    
    void attach(minc_1_reader& rw)
    {
      minc_input_iterator_base::attach(rw);
      _buf.resize(rw.slice_len());
    }
    
    
    bool next(void)
    {
      if(_last) return false;
      _count++;
      for(size_t i=static_cast<size_t>(_rw->dim_no()-1);
          i>static_cast<size_t>(_rw->dim_no()-_rw->slice_dimensions()-1);i--)
      {
        _cur[i]++;
        if(_cur[i]<static_cast<long>(_rw->dim(i).length))
          break;
        if(i>static_cast<size_t>(_rw->dim_no()-_rw->slice_dimensions())) 
          _cur[i]=0;
        else
        {
          //move to next slice 
          if(i==0) // the case when slice_dimensions==dim_no
          {
            _last=true;
            _count=0;
            break;
          }
          if(!_rw->next_slice())
          {
            _last=true;
            break;
          }
          _rw->read(&_buf[0]);
          _cur=_rw->current_slice();
          _count=0;
          break;
        }
      }
      return !_last;
    }
    
    
    void begin(void)
    {
      _cur.resize(MAX_VAR_DIMS,0);
      _buf.resize(_rw->slice_len());
      _count=0;
      _rw->begin();
      _rw->read(&_buf[0]);
      _cur=_rw->current_slice();
    }
    
    const T& value(void) const
    {
      return _buf[_count];
    }
  };
  
  template <class T> class minc_output_iterator: public minc_output_iterator_base
  {
    protected:
      std::vector<T> _buf;
    public:
    
    minc_output_iterator(const minc_output_iterator<T>& a):minc_output_iterator_base(a)
    {
    }
    
    minc_output_iterator(minc_1_writer& rw):minc_output_iterator_base(rw)
    {
    }
    
    minc_output_iterator()
    {
    }
    
    void attach(minc_1_writer& rw)
    {
      minc_output_iterator_base::attach(rw);
      _buf.resize(rw.slice_len());
    }  
    
    ~minc_output_iterator()
    {
      if(_count && !_last && _rw)
        _rw->write(&_buf[0]);
    }
    
    bool next(void)
    {
      if(_last) return false;
      _count++;
      for(int i=_rw->dim_no()-1;i>(_rw->dim_no()-_rw->slice_dimensions()-1);i--)
      {
        _cur[i]++;
        if(_cur[i]<static_cast<long>(_rw->dim(i).length))
          break;
        if(i>(_rw->dim_no()-_rw->slice_dimensions())) 
          _cur[i]=0;
        else
        {
          //write slice into minc file
          _rw->write(&_buf[0]);
          _count=0;
          //move to next slice 
          if(i==0) // the case when slice_dimensions==dim_no
          {
            _last=true;
            return false;
          }
          if(!_rw->next_slice())
          {
            _last=true;
            break;
          }
          _cur=_rw->current_slice();
          break;
        }
      }
      return !_last;
    }
        
    void begin(void)
    {
      _buf.resize(_rw->slice_len());
      _cur.resize(MAX_VAR_DIMS,0);
      _count=0;
      _rw->begin();
      _cur=_rw->current_slice();
    }
    
    void value(const T& v) 
    {
      _buf[_count]=v;
    }
  }; 

  class minc_parallel_input_iterator
  {
  protected:
    std::vector<minc_1_reader> files;
    std::vector<minc_input_iterator<double> > in;
    minc_1_reader mask;
    minc_input_iterator<unsigned char> mask_it;
    bool _have_mask;
  public:
    minc_parallel_input_iterator():_have_mask(false)
    {}
      
    void begin(void);
    bool last(void);
    bool next(void);
    
    bool have_mask(void) const
    {
      return _have_mask;
    }
    
    bool mask_value() const
    {
      if(_have_mask)
        return mask_it.value();
      return true;
    }
  
    void value(std::vector<double>&v) const;
    void value(double *v);
    void open(const std::vector<std::string> &in_files,const std::string &mask_file="");
  
    size_t dim(void) const
    {
      return in.size();
    }
  };

  class minc_parallel_output_iterator
  {
  protected:
    std::vector<minc_1_writer> wrt;
    std::vector<minc_output_iterator<double> > out;
  public:
    minc_parallel_output_iterator()
    {}
      
    void begin(void);
    bool last(void);
    bool next(void);
    void value(const std::vector<double>&v);
    void value(const double *v);
    void open(const std::vector<std::string> &out_files,const minc_info & output_info,const char* history=NULL);
    
    size_t dim(void) const
    {
      return out.size();
    }
    
  };


  //! will attempt to load the whole volume in T Z Y X V order into buffer, file should be prepared (setup_read_XXXX)
  template<class T> void load_standard_volume(minc_1_reader& rw, T* volume)
  {
    std::vector<size_t> strides(MAX_VAR_DIMS,0);
    size_t str=1;
    for(size_t i=0;i<5;i++)
    {      
      if(rw.map_space(i)<0) continue;
      strides[rw.map_space(i)]=str;
      str*=rw.ndim(i);
    }

    minc_input_iterator<T> in(rw);
    for(in.begin();!in.last();in.next())
    {
      size_t address=0;
      for(size_t i=0;i<static_cast<size_t>(rw.dim_no());i++)
        address+=in.cur()[i]*strides[i];
        
      volume[address]=in.value();
    }
  }
  
  //! will attempt to save the whole volume in T Z Y X V order from buffer, file should be prepared (setup_read_XXXX)
  template<class T> void save_standard_volume(minc_1_writer& rw, const T* volume)
  {
    std::vector<size_t> strides(MAX_VAR_DIMS,0);
    size_t str=1;
    for(size_t i=0;i<5;i++)
    {      
      if(rw.map_space(i)<0) continue;
      strides[rw.map_space(i)]=str;
      str*=rw.ndim(i);
    }
    
    minc_output_iterator<T> out(rw);
    for(out.begin();!out.last();out.next())
    {
      size_t address=0;
      for(size_t i=0;i<static_cast<size_t>(rw.dim_no());i++)
        address+=out.cur()[i]*strides[i];
        
      out.value(volume[address]);
    }
  }
  
  
};//minc


#endif //__MINC_1_ITERATORS_H__
