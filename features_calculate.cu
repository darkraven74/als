#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <ctime>
#include <thrust/device_vector.h>
#include "features_calculate.hpp"
#include "matrix_mul.cuh"


const int _c_fea = 2;

features_calculate::features_calculate(std::string& u_file, 
                                       int u_size,
                                       std::string& i_file, 
                                       int i_size,
                                       std::string& in_f_list,
                                       int f_size) 
                                       : features_size(f_size),
                                         items_size(i_size)
{

   std::cerr<< "Read user's matrix: " << std::endl;
   
   users.assign(u_size * features_size, 0);
   read_matrix(u_file, users, u_id2pos, u_pos2id);
   
   std::cerr<< "Read items's matrix: " << std::endl;
   
   items.assign(i_size * features_size, 0);
   read_matrix(i_file, items, i_id2pos, i_pos2id);
   
   std::cerr<< "Read cluster matrix: " << std::endl;
   read_data_template(in_f_list);

   std::cerr<< "Normalize items: " << std::endl;
   clock_t time = clock();
   normalize_items();
   cudaDeviceSynchronize();
   time = clock() - time;
   std::cerr<< "Normalize time: " << (float)time / CLOCKS_PER_SEC << std::endl;

}


void features_calculate::read_matrix(const std::string& file_name, 
                                     std::vector<float>& matrix, 
                                     std::vector<int>& ids,
                                     std::vector<int>& pos)
{
    std::ifstream m_stream(file_name.c_str());
    std::string line;
    char const tab_delim = '\t';
    size_t i=0; 
    
    while ( getline(m_stream, line) && ((i + 1) * features_size - 1 < matrix.size()))
    {
        
        std::istringstream line_stream(line);
        std::string value;
        getline(line_stream, value, tab_delim);        
        ids.push_back(atoi(value.c_str()));
        
        for (int j = 0; j < features_size; j++)
        {
          getline(line_stream, value, tab_delim);
          matrix[i * features_size + j] = (float)atof(value.c_str());
        }
        
        if( i % 10000 == 0) std::cerr << i << "\r";
        
        i++;
    }
       
   m_stream.close();
   int max_id = *std::max_element(ids.begin(), ids.end());
   //pos.assign(ids.size(), 0);
   pos.assign(max_id, 0);
   for( i=0; i < ids.size(); i++)
   {
      pos[ids[i]] = i; 
   }
}

void features_calculate::read_data_template(const std::string& file_name )
{
    std::ifstream m_stream(file_name.c_str());
    std::string line;
    char const tab_delim = '\t';
    size_t i=0; 
    while ( getline(m_stream, line) && (i < items_size))
    {        
        std::istringstream line_stream(line);
        std::string value;
        getline(line_stream, value, tab_delim);        
        int user = atoi(value.c_str());
        
        getline(line_stream, value, tab_delim);        
        int rgroup = atoi(value.c_str());
        
        getline(line_stream, value, tab_delim);        
        int group = atoi(value.c_str());
        
        __key k = std::make_pair(user, rgroup);
        
         __value::iterator it = data[k].find(group);
         
         if( it == data[k].end() )
         {
              data[k][group].assign(_c_fea, 0);
         }
                
        if( i % 10000 == 0) std::cerr << i << "\r";
        i++;
    }
       
}

void features_calculate::compute_features( )
{
     std::map<__key, __value >::iterator it = data.begin();
     
     for( ; it !=  data.end() ;  ++it)   
     {
         int user = (*it).first.first;
         int rgroup = (*it).first.second;
         __value& m = (*it).second;
         __value::iterator git = m.begin();
         
         for( ; git != m.end() ; ++git)
         {
            int group = (*git).first;
            (*git).second[0] = calculate_p(user, group);
            (*git).second[1] = calculate_sim(rgroup, group);
         }
     }
}

float features_calculate::calculate_p( int u_idx, int group_idx)
{
    float sum = 0;
    for (int j = 0; j < features_size; j++)
    {
    	if ((u_idx * features_size + j < users.size()) && (group_idx * features_size + j < items.size()))
    		sum += users[ u_idx * features_size + j] * items[group_idx * features_size + j];
    }
  return sum;    
}

float features_calculate::calculate_sim( int rgroup_idx, int group_idx)
{
    float sum = 0;
    for (int j = 0; j < features_size; j++)
    {
    	if ((rgroup_idx * features_size + j < norm_items.size()) && (group_idx * features_size + j < norm_items.size()))
    		sum += norm_items[ rgroup_idx * features_size + j] * norm_items[group_idx * features_size + j];
    }
  return sum;    
}



///

void features_calculate::normalize_items( )
{
	cudaSetDevice(0);
	int part1_size = items_size / 2;
	int part2_size = items_size - part1_size;
	norm_items = items;

	thrust::device_vector<float> c_device_part1(norm_items.begin(), norm_items.begin() + part1_size * features_size);
	dim3 block1(NORM_BLOCK_SIZE, 1);
	dim3 grid1(1 + part1_size / NORM_BLOCK_SIZE, 1);
	euclidian_normalize<<<grid1, block1>>>(thrust::raw_pointer_cast(&c_device_part1[0]), part1_size, features_size);

	cudaSetDevice(1);
	thrust::device_vector<float> c_device_part2(norm_items.begin() + part1_size * features_size, norm_items.end());
	dim3 block2(NORM_BLOCK_SIZE, 1);
	dim3 grid2(1 + part2_size / NORM_BLOCK_SIZE, 1);
	euclidian_normalize<<<grid2, block2>>>(thrust::raw_pointer_cast(&c_device_part2[0]), part2_size, features_size);
    cudaDeviceSynchronize();
    if ( cudaSuccess != cudaGetLastError() )
         std::cerr <<  "features_calculate::normalize_items:: !WARN - Cuda error : "  << cudaGetLastError() << std::endl;

    thrust::copy( c_device_part2.begin(), c_device_part2.end(), norm_items.begin() + part1_size * features_size );
    cudaSetDevice(0);
    cudaDeviceSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
		 std::cerr <<  "features_calculate::normalize_items:: !WARN - Cuda error : "  << cudaGetLastError() << std::endl;

	thrust::copy( c_device_part1.begin(), c_device_part1.end(), norm_items.begin() );


	/*thrust::device_vector<float> c_device(norm_items);

    dim3 block(NORM_BLOCK_SIZE, 1);
    dim3 grid(1 + i_id2pos.size() / NORM_BLOCK_SIZE, 1);
    
    euclidian_normalize<<<grid, block>>>(thrust::raw_pointer_cast(&c_device[0]), i_id2pos.size(), features_size);


    cudaDeviceSynchronize();
    if ( cudaSuccess != cudaGetLastError() )
         std::cerr <<  "features_calculate::normalize_items:: !WARN - Cuda error : "  << cudaGetLastError() << std::endl;
    

    thrust::copy( c_device.begin(),c_device.end(), norm_items.begin()  );


    for (int i = 0; i < 5; i++)
    {
        std::cerr << norm_items[i] << " ";
    }
    for (int i = items_size - 1; i > items_size - 6; i--)
	{
		std::cerr << norm_items[i] << " ";
	}
    std::cerr << std::endl;
    */
}

void features_calculate::serialize( std::ostream& out)
{
     std::map<__key, __value >::iterator it = data.begin();
     
     for( ; it !=  data.end() ;  ++it)   
     {
         int user = (*it).first.first;
         int rgroup = (*it).first.second;
         __value& m = (*it).second;
         __value::iterator git = m.begin();
         
         for( ; git != m.end() ; ++git)
         {
            int group = (*git).first;
            out << user << "\t" << rgroup << "\t" << group 
                << "\t" << (*git).second[0] 
                << "\t" << (*git).second[1] << std::endl; 
         }
     }
     
}



