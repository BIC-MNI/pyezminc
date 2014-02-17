#include <iostream>
#include <fstream>
#include "minc_1_iterators.h"
#include <assert.h>
#include <string.h>

namespace minc
{

  void minc_parallel_input_iterator::open(const std::vector<std::string> &in_files,const std::string &mask_file)
  {
    if(!mask_file.empty())
    {
      mask.open(mask_file.c_str());
      mask.setup_read_byte();
      mask_it.attach(mask);
      //mask_it.begin();
      _have_mask=true;
    }

    files.resize(in_files.size());
    in.resize(in_files.size());
  
    for(int i=0;i<in_files.size();i++)
    {
      files[i].open(in_files[i].c_str());
      if(i==0 && _have_mask)
      {
        if(files[0].dim_no()!=mask.dim_no())
        {
          std::cerr<<"Input file "<< in_files[i].c_str() <<" should have same number of dimensions as mask!"<<std::endl;
          REPORT_ERROR("Input file with inconsistent dimensions");
        }
        bool good=true;
        for(int j=0;j<files[0].dim_no();j++)
          if(mask.dim(j).length!=files[0].dim(j).length)
            good=false;
        if(!good)
        {
          std::cerr<<"Input file "<< in_files[i].c_str() <<" should have same dimensions as mask!"<<std::endl;
          REPORT_ERROR("Input file with inconsistent dimensions");
        }
      }
      //check to make sure that all files are proper
      if(i>0)
      {
        if(files[0].dim_no()!=files[i].dim_no())
        {
          std::cerr<<"Input file "<< in_files[i].c_str() <<" should have same number of dimensions as first file!"<<std::endl;
          REPORT_ERROR("Input file with inconsistent dimensions");
        }
        bool good=true;
        for(int j=0;j<files[0].dim_no();j++)
          if(files[i].dim(j).length!=files[0].dim(j).length)
            good=false;
        if(!good)
        {
          std::cerr<<"Input file "<< in_files[i].c_str() <<" should have same dimensions as first file!"<<std::endl;
          REPORT_ERROR("Input file with inconsistent dimensions");
        }
      }
      files[i].setup_read_double();
      in[i].attach(files[i]);
      //in[i].begin();
    }
  }

  void minc_parallel_input_iterator::begin(void)
  {
    if(_have_mask)
      mask_it.begin();
    for(size_t i=0;i<in.size();i++)
      in[i].begin();
  }
  
  bool minc_parallel_input_iterator::last(void)
  {
    if(_have_mask)
      if(mask_it.last()) return true;
    
    return in[0].last();
  }
  
  bool minc_parallel_input_iterator::next(void)
  {
    bool _good=true;
    
    if(_have_mask)
      _good = _good && mask_it.next();

    for(size_t i=0;i<in.size();i++)
      _good = _good && in[i].next();
    return _good;
  }
  
  void minc_parallel_input_iterator::value(std::vector<double>&v) const
  {
    assert(v.size()==in.size());
    
    for(size_t i=0;i<in.size();i++)
      v[i]=in[i].value();
  }
  
  
  void minc_parallel_output_iterator::open(const std::vector<std::string> &out_files,const minc_info & output_info,const char* history)
  {
    wrt.resize(out_files.size());
    out.resize(out_files.size());
            
    for(int k=0;k<out_files.size();k++)
    {      
      wrt[k].open(out_files[k].c_str(),output_info,2,NC_FLOAT);
      if(history)
        wrt[k].append_history(history);
          
      wrt[k].setup_write_double();
      
      out[k].attach(wrt[k]);
      //out[k].begin();
    }
  }
  
  void minc_parallel_output_iterator::begin(void)
  {
    for(int k=0;k<out.size();k++)
      out.begin();
  }
  
  bool minc_parallel_output_iterator::last(void)
  {
    return out[0].last();
  }
  
  bool minc_parallel_output_iterator::next(void)
  {
    bool good=true;
    for(int k=0;k<out.size();k++)
      good = good && out[k].next();
    return good;
  }
  
  void minc_parallel_output_iterator::value(const std::vector<double>&v)
  {
    assert(v.size()==out.size());
    for(int k=0;k<out.size();k++)
      out[k].value(v[k]);
  }
}; //minc
