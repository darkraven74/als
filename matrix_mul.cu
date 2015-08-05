#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <limits>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cublas_v2.h>
#include <ctime>
#include <iostream>
#include "matrix_mul.cuh"

matrix_mul::matrix_mul(const std::string& a_file_name,
                       const std::string& b_file_name, 
                       int a_size,
		               int b_size, 
                       int features_size)
		: a_file_name(a_file_name),
		  b_file_name(b_file_name),
		  a_size(a_size),
		  b_size(b_size),
		  features_size(features_size),
          b(b_size * features_size),
          a(a_size * features_size),
          status(cublasCreate(&handle)),
          alpha(1.0f),
	      beta(0.0f)
{
  int cout_dev=0;  
  cudaGetDeviceCount(&cout_dev);
  for( int i=0; i < cout_dev; i++ )
  {	  
     std::cerr << "=== CUDA Device: " <<  i << std::endl;
     cudaGetDeviceProperties(&prop, i);
     std::cerr << "Cuda Device: " << prop.name << std::endl;
     std::cerr << "Total gloabl mem: " << prop.totalGlobalMem << std::endl;
     std::cerr << "Total shared mem per block: " << prop.sharedMemPerBlock << std::endl; 
     std::cerr << "Multi processors: " << prop.multiProcessorCount << std::endl;
     std::cerr << "Warp size: " <<  prop.warpSize << std::endl;
     std::cerr << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
     std::cerr << "maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;

 } 
 
  cudaSetDevice(0);
  cudaDeviceEnablePeerAccess(1, 0);
  cudaSetDevice(1);
  cudaDeviceEnablePeerAccess(0, 0);
  cudaSetDevice(0);
  
  std::cerr << "\nReading matrix b: " << b_size << std::endl;
  read_matrix(b_file_name, b_size, b, b_ids);
  
  std::cerr << "\nReading matrix a: " << a_size << std::endl;
  read_matrix(a_file_name, a_size, a, a_ids);    
}

matrix_mul::~matrix_mul()
{
  status = cublasDestroy(handle);
}

void matrix_mul::read_matrix(const std::string& file_name, 
                             int m_size, 
                             std::vector<float>& matrix, 
                             std::vector<int>& ids)
{
    std::ifstream m_stream(file_name.c_str());
    std::string line;
    char const tab_delim = '\t';
    
    for ( int i =0; i < m_size; i++ )
    {
        getline(m_stream, line);
        
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
    }
       
   m_stream.close();
}
          
part_matrix_mul::part_matrix_mul(const std::string& a_file_name,
                                 const std::string& b_file_name, 
                                 const std::string& part_file,
                                 int a_size,
		                         int b_size, 
                                 int features_size) 
                                 : matrix_mul(a_file_name, b_file_name, a_size, b_size, features_size), 
                                   _parts_descr_file(part_file),
                                   last_part(-1),
                                   m_part_stream(_parts_descr_file.c_str()),
                                   _skip_likes_filter(false)
{
   /// save mapping
   map_ids(a_ids, a_map);
   map_ids(b_ids, b_map);
}

const int bytes_segm_length = 8;

void set_bit_index(unsigned char* bits, int index)
{
    int word =  index / bytes_segm_length;
    int bit  =  index % bytes_segm_length;
    bits[word] |= 0x1 << bit;
}

bool part_matrix_mul::read_next_part_file(std::vector<int>& items, std::vector<int>& users, std::vector<unsigned char >& user_likes)
{    
    std::string line;
    char const tab_delim = '\t';
    users = last_users_set;
    items = last_items;
    int part = last_part;
    bool no_eof;
    int i=0;    
    std::map<int, int> umap;
    std::vector<std::vector<int> > tmp_user_likes;
    
    for(size_t u=0; u < last_users_set.size(); u++){
           tmp_user_likes.push_back(std::vector<int>());
           for( int k=0; k < items.size(); k++ ) tmp_user_likes[u].push_back(k);
    }
    
    std::cerr << "Read part: " << part << std::endl;
    
    while ( last_part  == part && (no_eof = getline(m_part_stream, line))  )
    {            
        std::istringstream line_stream(line);
        std::string value;
        
        // first is part index
        getline(line_stream, value, tab_delim);
        part=atoi(value.c_str());        
        if(last_part == -1 ) last_part = part;
        // second is item
        getline(line_stream, value, tab_delim);
        int item = atoi(value.c_str());
        
        std::vector<int>& r_users = ( part != last_part )? last_users_set : users;
        std::vector<int>& r_items = ( part != last_part )? last_items : items;
        
        if( part != last_part ) { umap.clear(); }
                
        while(getline(line_stream, value, tab_delim) )
        {
          int uid = atoi(value.c_str());
          std::map<int, int>::iterator it = umap.find(uid);
          int idx = r_users.size(); 
          if(it == umap.end() ){
            umap[uid] = idx;
            r_users.push_back(uid);
            tmp_user_likes.push_back(std::vector<int>());
            
          }else
          {
              idx = it->second;
          }
          
          if( part != last_part ) tmp_user_likes[idx].push_back( r_items.size() );
        }
        
        r_items.push_back(item);
        i++;
        
    }
    
   last_part = part;   
   ///
   /// Allign arrays
   ///
   int words = items.size() / bytes_segm_length +1;
   user_likes.assign( words * tmp_user_likes.size(), 0);
   if( !_skip_likes_filter )
   {
       for( size_t ul=0; ul < tmp_user_likes.size(); ul++) 
       {       
          for ( size_t k = 0;  k < tmp_user_likes[ul].size(); k++)
          {
            set_bit_index(&user_likes[ul * words], tmp_user_likes[ul][k]);
          }
       }
   }
  // std::cerr << "\nLast part is: " << part << " users: " << users.size() << " items: " << items.size() << " words: " << words  << std::endl;  
  
  std::sort(items.begin(), items.end() );
  std::sort(users.begin(), users.end() );
  
 return no_eof;
}


struct comparator
{
	float* c_device;
	int i;
	int b_size;
	__host__ __device__ comparator(float* c_device, int& i, int& b_size) : c_device(c_device), i(i), b_size(b_size) {}
    bool __host__ __device__ operator()(const int& id1, const int& id2) const
    {        
    	return c_device[i * b_size + id1] > c_device[i * b_size + id2];
    }
};

#define BLOCK_SIZE 128
#define BLOCK_X_SIZE 8
#define BLOCK_SIZE_2 512

///
///  a, b matrixies input
///  c - output mitrix 
///  a_size- a matrix size
///  features_size - features count
///  s3 - b matrix size
///
/*__global__ void matMulGpuShared(float* a, float* b, float* c, int a_size, int features_size, int b_size)
{
	__shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int parts = features_size / BLOCK_SIZE + 1;
    float ans = 0;

	for (int id = 0; id < parts; id++)
	{
        const int x_idx = id * BLOCK_SIZE + threadIdx.x;
        const int y_idx = id * BLOCK_SIZE + threadIdx.y;
        
        if (i < a_size && x_idx < features_size )
        {
            a_shared[threadIdx.y][threadIdx.x] =  a[i * features_size + x_idx];
        }
        else
        {
            a_shared[threadIdx.y][threadIdx.x] = 0;
        }

        if (j < b_size && y_idx <  features_size )
        {
            b_shared[threadIdx.y][threadIdx.x] = b[j * features_size + y_idx];
        }
        else
        {   
            b_shared[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            ans += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        }
        __syncthreads();

    }

//    if (i < a_size && j < b_size)
//    {
//        c[i * b_size + j] = ans;
//    }

}
*/


void matrix_mul::calculate(std::ostream& output_stream, int n, int block_size)
{
    
    out_ids.assign(a_ids.size() * n, 0);
    out_ranks.assign(a_ids.size() * n, 0);
    

    mul_by_block(a, b, b_ids, a_size, b_size, n, block_size, out_ranks, out_ids);
    
    ///
    std::cerr << "Writing data ... " << std::endl;
    for( size_t i=0; i < a_ids.size(); i++ )
    {
      output_stream << a_ids[i];
      for( size_t j = 0; j < n; j++)
      {
         output_stream << "\t" << out_ranks[i * n + j] << "/" << out_ids[i * n + j];
      }
      output_stream << std::endl;   
    }

    
}

void matrix_mul::mul_by_block(std::vector<float>& a,
                              std::vector<float>& b,
                              std::vector<int>& b_ids,
                              int a_size,
                              int b_size,
                              int n, 
                              int block_size,
                              std::vector<float>& items_ranks,
                              std::vector<int>& items_ids                                                            
                             )
{
    thrust::device_vector<float> b_device(b);
    thrust::device_vector<float> a_device(block_size * features_size);
    thrust::device_vector<float> c_device(block_size * b_size);
    thrust::device_vector<int>   b_id_device(b_ids);
    
    cudaSetDevice(1);
    thrust::device_vector<int>   b_id_device2(b_ids);
    thrust::device_vector<float> c_device2(block_size * b_size);
    cudaSetDevice(0);

    //thrust::device_vector<int>   b_ids_device(block_size * b_size);

	int parts = a_size / block_size + (((a_size % block_size) == 0)? 0 : 1);

	clock_t total = 0;

	for (int id = 0; id < parts; id++) //multiplication matrix "a"  by blocks
	{
        std::cerr << "part " << id << " of " << parts << "\r";

		int a_actual_size = (id == parts - 1 && (a_size % block_size) != 0) ? (a_size % block_size) : block_size;
        int a_offset = id * block_size * features_size;

        thrust::copy(a.begin()+a_offset, a.begin()+a_offset+( a_actual_size * features_size), a_device.begin() );
        
		// matrix multiplication
		status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, b_size, a_actual_size, features_size, &alpha,
				 thrust::raw_pointer_cast(&b_device[0]), features_size, thrust::raw_pointer_cast(&a_device[0]),
				 features_size, &beta, thrust::raw_pointer_cast(&c_device[0]), b_size);

		int actual_part1 = a_actual_size / 2;
		int actual_part2 = a_actual_size - actual_part1;
		cudaMemcpyPeer(thrust::raw_pointer_cast(&c_device2[0]), 1, thrust::raw_pointer_cast(&c_device[0]) + actual_part1 * b_size, 0, actual_part2 * b_size * sizeof(float));

		cudaDeviceSynchronize();
		clock_t time = clock();
        process_block(c_device, c_device2, b_id_device, b_id_device2, id,  n, block_size, a_actual_size, b_size, items_ranks, items_ids) ;
        cudaDeviceSynchronize();
        time = clock() - time;
        total += time;
        
	}

	std::cerr<< "process time: " << (float)total / CLOCKS_PER_SEC << std::endl;

     
     
}

void part_matrix_mul::map_ids(const std::vector<int>& ids,                              
                               std::map<int, int>& i_map
                              )
{
   std::vector<int>::const_iterator it = ids.begin();
   int i=0; 
   for ( ; it != ids.end() ; ++it, i++)
   {
       i_map[*it] = i;
   }
}

///
/// Copy vectors
///
void copy_data(std::vector<float>& b, 
               std::vector<int>& items,
               std::map<int, int>& b_map,
               int features_size,
               std::vector<float>& out_items 
              )
{
      static std::vector<float> fea_null(features_size, 0);
      std::vector<int>::iterator iit = items.begin();
      int miss=0;
            
      for( int i=0; iit != items.end(); ++iit) 
      {               
         if(b_map.find(*iit) != b_map.end()) 
         {
             int id = b_map[*iit];
             int offset = id * features_size;
             std::copy(b.begin() + offset, b.begin() + offset + features_size, out_items.begin() + i * features_size);
         }else{
            std::copy(fea_null.begin(), fea_null.end(), out_items.begin() + i * features_size); 
            miss++;
         }
         i++;
      }   
      std::cerr << "miss: " << miss << std::endl;
}

const float block_size_coeff = 1;

void part_matrix_mul::calculate(std::ostream& output_stream, int n, int block_size)
{
    std::vector<int> items;
    std::vector<int> users;
    
    time_t mul = 0;
    time_t merge = 0;
    time_t tot = clock();

    size_t fea_b_size =  sizeof(float) * features_size;
    int count_parts = 0;
    
    ///
    /// output data
    ///
    out_ids.assign(a_ids.size() * n, 0);
    
    ///
    /// user top n ranks, stire its in this array
    ///
    user_top_ranks.assign(a_ids.size() * n, 0);
    
    while(read_next_part_file(items, users, user_likes))
    {
      size_t free_mem_0 =0;
      size_t mem_tot_0 =0;

    
      cudaMemGetInfo  (&free_mem_0, & mem_tot_0);
      long result_mem = free_mem_0 / 2 ;
      
      
      const long i_size = items.size() * fea_b_size;
      long result_size = users.size() * items.size() * sizeof(float);
            
      result_size = (result_size > result_mem)? result_mem : result_size;
      
      std::vector<float> fea_users(users.size() * features_size);
      std::vector<float> fea_items(items.size() * features_size);
      
      copy_data(a, users, a_map, features_size, fea_users);
      copy_data(b, items, b_map, features_size, fea_items);
      
      long block_size =   (result_size / (items.size() * sizeof(float)) +1);
      long free_mem = free_mem_0 - ( result_size  + i_size );
      block_size = ( ( (block_size * fea_b_size) > free_mem ) ? free_mem / fea_b_size : block_size ) * block_size_coeff;
      
//      std::cerr << "block size: " << block_size << " free mem " << free_mem << " result size: " << result_size << std::endl;
      std::cerr << "Num: " << count_parts << " users: " << users.size() << " items: " << items.size() 
                << " mem: " << free_mem_0 
                << " block size: "<< block_size 
                << std::endl;
                
      /// output for matrix multiplication
      std::vector<float> user_items_ranks(n * users.size());
      std::vector<int> items_ids(n * users.size() );
      

      cudaDeviceSynchronize();
      time_t time = clock();
      mul_by_block(fea_users, 
                   fea_items,
                   items, users.size(), items.size(), n, (size_t)block_size,
                   user_items_ranks, items_ids);
      cudaDeviceSynchronize();
      time = clock() - time;
      mul += time;

      time = clock();
      merge_recommends(user_items_ranks, items_ids, users, n, users.size());
      cudaDeviceSynchronize();
      time = clock() - time;
      merge += time;


      count_parts++;
      
      if( count_parts == 30) break;

      user_likes.clear();
   }
    
    tot = clock() - tot;

    std::cerr<< "mul time: " << (float)mul / CLOCKS_PER_SEC << std::endl;
    std::cerr<< "merge time: " << (float)merge / CLOCKS_PER_SEC << std::endl;
    std::cerr<< "total time: " << (float)tot / CLOCKS_PER_SEC << std::endl;

    ///
    std::cerr << "Writing data ... " << std::endl;
    for( size_t i=0; i < a_ids.size(); i++ )
    {
      output_stream << a_ids[i];
      for( size_t j = 0; j < n; j++)
      {
         output_stream << "\t" << user_top_ranks[i * n + j] << "/" << out_ids[i * n + j];
      }
      output_stream << std::endl;   
    }
    
}

#define queue_size 100
/*__shared__ float s[BLOCK_SIZE][queue_size];
__shared__ int s_id[BLOCK_SIZE][queue_size]; 

__device__ void sort_shared_row(int row, int start, int end)
{
    
    int i_pivot = (start+end)/2;
    float pivot = s[row][i_pivot];
    int i =start, j=end;
    while( i < j )
    {
        while( s[row][i] > pivot ) i++;
        while( s[row][j] < pivot ) j--;
        if ( i <= j ){
          float wsp  = s[row][i];
           s[row][i] = s[row][j]; 
           s[row][j] =wsp;
          ++i; 
          --j;
        }
    }
    if( start < j) sort_shared_row(row, start, j);
    if( i < end) sort_shared_row(row, i, end);
}
*/
__device__ void qsort_local_pointer(float* row, int start, int end)
{
    
    int i_pivot = (start+end)/2;
    float pivot = row[i_pivot];
    int i =start, j=end;
    while( i < j )
    {
        while( row[i] > pivot ) i++;
        while( row[j] < pivot ) j--;
        if ( i <= j ){
          float wsp  = row[i];
           row[i] = row[j]; 
           row[j] =wsp;
          ++i; 
          --j;
        }
    }
    if( start < j) qsort_local_pointer(row, start, j);
    if( i < end) qsort_local_pointer(row, i, end);
}

///
/// Retruns insert index
///
__device__ int binarySearch(float a[], size_t n, int x)
{
    size_t first = 0;
    size_t last = n;
     
    if (n == 0 || x > a[0] ) return 0;
    else if ( x < a[n - 1] ) return n;
 
    /* Если просматриваемый участок непустой, first < last */
    while (first < last) {
        size_t mid = first + (last - first) / 2;
 
        if (x >= a[mid])
            last = mid;
        else
            first = mid + 1;
    }
 
    if ( a[last] == x) {        
        return last;
    } else {
        return last;
    }
}
///
/// Combosort on local memory
///
__device__ void combsort_local_pointer(float array[ ], int ids[], size_t size ) 
{
    if (array && size) {
        std::size_t jump = size;
        bool swapped = true;
        while (jump > 1 || swapped) {
            if (jump > 1)
                jump /= 1.24733;
            swapped = false;
            for (std::size_t i = 0; i + jump < size; ++i)
                if ( array[i + jump] > array[i] ) {
                    float wsp  = array[i];
                    int   id   = ids[i];
                    
                    array[i] = array[i + jump];
                    ids[i] = ids[i + jump];
                    
                    array[i + jump] = wsp;
                    ids[i + jump] = id;
                    swapped = true;
                }
        }
    }
}

///
/// c_device - matrix with all user recommendations
/// ids - all items ids
/// filter_mask - filter for users
/// 
///
__global__ void select(float* c_device, 
                        int* ids, 
                        float* out_ranks,
                        int * out_ids, unsigned char* filter_mask, int n, int b_size, int block_size)
{

   
   float s[queue_size];
   int s_id[queue_size];
   int total_inserted = 0;
   int min_index=0;
   int max_index =0;
   float min_val=0;
   float max_val=0;
   int words = b_size / bytes_segm_length +1;
   __shared__ float cache_data[BLOCK_SIZE][BLOCK_X_SIZE];
   __shared__ int cache_ids[BLOCK_SIZE][BLOCK_X_SIZE];
   __shared__ int cache_filter[BLOCK_SIZE][BLOCK_X_SIZE / bytes_segm_length ];
   
   const int row = blockIdx.y * blockDim.y + threadIdx.y;   
   if( row >= block_size ) return;
   
   unsigned char* filter_row = filter_mask + words * row;
   
   for( int i=0; i< queue_size; i++ ) {s[i] = 0; s_id[i]=0; }
   
   const int t_row  = threadIdx.y;
   const int parts  = (b_size/BLOCK_X_SIZE) + 1;
   const int t_col = blockIdx.x * blockDim.x + threadIdx.x;
   const int tx    = threadIdx.x;
   
   // for( int i=0; i< b_size ; i++)
   for( int p=0; p< parts ; p++)
   {
       // copy data to shared memory
       const int x = p * BLOCK_X_SIZE + t_col;
       if( x < b_size)
       {
         cache_data[t_row][tx] = c_device[ row * b_size + x];
         cache_ids[t_row][tx]  = ids[x];
         if( tx  % bytes_segm_length == 0)
         {
           const int word =  x / bytes_segm_length;
           cache_filter[t_row][ tx / bytes_segm_length ] = filter_row[word];
         } 
       }
       
       __syncthreads();
       // only first thread calculate
       if(threadIdx.x == 0 )
       {
           for( int li = 0; li < BLOCK_X_SIZE; li++)
           {
               int i =  p * BLOCK_X_SIZE + li;
               
               if( i >= b_size ) continue;
               
               // const float c_val = c_device[ row * b_size + i];
               const float c_val = cache_data[t_row][li];
               
               // const int word =  i / bytes_segm_length;
               const int word = li / bytes_segm_length;
               const int bit  =  i % bytes_segm_length;
               const int filtered = (cache_filter[t_row][word] >> bit) & 1;
               
               if( filtered ) continue;
               
               if( total_inserted < queue_size )
               {
                   
                  s[total_inserted] = c_val;
                  //s_id[total_inserted]= ids[i];
                  s_id[total_inserted] =  cache_ids[t_row][li];
                  
                  if( c_val < min_val && total_inserted > 0) 
                  { min_val = c_val; min_index = total_inserted;}
                  else if(total_inserted == 0){ min_val = c_val; }
                  
                  if( c_val >= max_val && total_inserted > 0) 
                  { max_val = c_val; max_index = total_inserted;}
                  else if (total_inserted == 0) { max_val = c_val; }
                  
                 total_inserted ++;
               }else{
                   if( c_val > max_val )
                   {
                    s[min_index] = c_val;
                    //s_id[min_index] = ids[i];
                    s_id[min_index] = cache_ids[t_row][li];
                    max_val = c_val;
                    max_index = min_index; 
                    min_val =  c_val;
                    // select new min value
                    for( int j=0; j < total_inserted; j++)
                    {
                        if( s[j] < min_val  )
                        {
                            min_val = s[j];
                            min_index = j;
                        } 
                    }
                   }        
              }
            }
        }
        __syncthreads();
       
   }
   
   combsort_local_pointer(s, s_id, total_inserted);
   
   for(int i=0; i < total_inserted && i < n; i++)
   {
       out_ranks[ row * n + i] = s[i];
       out_ids[  row *  n  + i] =  s_id[i];
   }
}

///
/// c_device - matrix with all user recommendations
/// ids - all items ids
/// filter_mask - filter for users
/// 
///
__global__ void select_no_filter(float* c_device, 
                                 int* ids, 
                                 float* out_ranks,
                                 int * out_ids, int n, int b_size, int block_size)
{

   
   float s[queue_size];
   int s_id[queue_size];
   int total_inserted = 0;
   int min_index=0;
   int max_index =0;
   float min_val=0;
   float max_val=0;
   __shared__ float cache_data[BLOCK_SIZE][BLOCK_X_SIZE];
   __shared__ int cache_ids[BLOCK_SIZE][BLOCK_X_SIZE];
   
   const int row = blockIdx.y * blockDim.y + threadIdx.y;   
   if( row >= block_size ) return;
   
   for( int i=0; i< queue_size; i++ ) {s[i] = 0; s_id[i]=0; }
   
   const int t_row  = threadIdx.y;
   const int parts  = (b_size/BLOCK_X_SIZE) + 1;
   const int t_col = blockIdx.x * blockDim.x + threadIdx.x;
   const int tx    = threadIdx.x;
   
   // for( int i=0; i< b_size ; i++)
   for( int p=0; p< parts ; p++)
   {
       // copy data to shared memory
       const int x = p * BLOCK_X_SIZE + t_col;
       if( x < b_size)
       {
         cache_data[t_row][tx] = c_device[ row * b_size + x];
         cache_ids[t_row][tx]  = ids[x];
       }
       
       __syncthreads();
       // only first thread calculate
       if(threadIdx.x == 0 )
       {
           for( int li = 0; li < BLOCK_X_SIZE; li++)
           {
               int i =  p * BLOCK_X_SIZE + li;
               
               if( i >= b_size ) continue;
               
               // const float c_val = c_device[ row * b_size + i];
               const float c_val = cache_data[t_row][li];
               
               // const int word =  i / bytes_segm_length;
                              
               if( total_inserted < queue_size )
               {
                   
                  s[total_inserted] = c_val;
                  //s_id[total_inserted]= ids[i];
                  s_id[total_inserted] =  cache_ids[t_row][li];
                  
                  if( c_val < min_val && total_inserted > 0) 
                  { min_val = c_val; min_index = total_inserted;}
                  else if(total_inserted == 0){ min_val = c_val; }
                  
                  if( c_val >= max_val && total_inserted > 0) 
                  { max_val = c_val; max_index = total_inserted;}
                  else if (total_inserted == 0) { max_val = c_val; }
                  
                 total_inserted ++;
               }else{
                   if( c_val > max_val )
                   {
                    s[min_index] = c_val;
                    //s_id[min_index] = ids[i];
                    s_id[min_index] = cache_ids[t_row][li];
                    max_val = c_val;
                    max_index = min_index; 
                    min_val =  c_val;
                    // select new min value
                    for( int j=0; j < total_inserted; j++)
                    {
                        if( s[j] < min_val  )
                        {
                            min_val = s[j];
                            min_index = j;
                        } 
                    }
                   }        
              }
            }
        }
        __syncthreads();
       
   }
   
   combsort_local_pointer(s, s_id, total_inserted);
   
   for(int i=0; i < total_inserted && i < n; i++)
   {
       out_ranks[ row * n + i] = s[i];
       out_ids[  row *  n  + i] =  s_id[i];
   }
}

///
/// Normalize vector on euclidian norm
///
__global__ void euclidian_normalize(float* c_device,                           
                                     int m_size, int vec_length)
{
    
    const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( vec_idx >= m_size) return;
    float sum = 0;
    for( int i=0; i < vec_length; i++)
    {
        const float a = c_device[vec_idx * vec_length + i];
        sum += a * a;
    }   
    sum = sqrt (sum);
    
    for( int i=0; i < vec_length; i++)
    {
        float a = c_device[vec_idx * vec_length + i];
        a /= sum ;
        c_device[vec_idx * vec_length + i] = a;
    }   
}


///
/// Without shared memory
///
__global__ void select_no_shared_mem(float* c_device, int* ids, unsigned char* filter_mask, int n, int b_size, int block_size)
{

   
   float s[queue_size];
   int s_id[queue_size];
   int total_inserted = 0;
   int min_index=0;
   int max_index =0;
   float min_val=0;
   float max_val=0;
   int words = b_size / bytes_segm_length +1;
   
   const int row = blockIdx.y * blockDim.y + threadIdx.y;   
   if( row >= block_size ) return;
   
   unsigned char* filter_row = filter_mask + words * row;
   
   for( int i=0; i< queue_size; i++ ) {s[i] = 0; s_id[i]=0; }
      
   for( int i=0; i< b_size ; i++)
   {
       const float c_val = c_device[ row * b_size + i];
       
       const int word =  i / bytes_segm_length;
       const int bit  =  i % bytes_segm_length;
       const int filtered = (filter_row[word] >> bit) & 1;
       
      // if( filtered ) continue;
       
       if( total_inserted < queue_size )
       {
           
          s[total_inserted] = c_val;
          s_id[total_inserted]= ids[i];
          
          if( c_val < min_val && total_inserted > 0) 
          { min_val = c_val; min_index = total_inserted;}
          else if(total_inserted == 0){ min_val = c_val; }
          
          if( c_val >= max_val && total_inserted > 0) 
          { max_val = c_val; max_index = total_inserted;}
          else if (total_inserted == 0) { max_val = c_val; }
          
         total_inserted ++;
       }else{
           if( c_val > max_val )
           {
            s[min_index] = c_val;
            s_id[min_index] = ids[i];
            max_val = c_val;
            max_index = min_index; 
            min_val =  c_val;
            // select new min value
            for( int j=0; j < total_inserted; j++)
            {
                if( s[j] < min_val  )
                {
                    min_val = s[j];
                    min_index = j;
                } 
            }
           }        
      }       
   }
   
   combsort_local_pointer(s, ids, total_inserted);
   
   for(int i=0; i < total_inserted; i++)
   {
       c_device[ row * b_size + i] = s[i];
   }
}


void part_matrix_mul::process_block(thrust::device_vector<float>& c_device,
							   thrust::device_vector<float>& c_device2,
                               thrust::device_vector<int>& b_id_device,
                               thrust::device_vector<int>& b_id_device2,
                               int block_id,
                               int n,
                               int block_size,
                               int actual_block_size,
                               int b_size,
                               std::vector<float>& items_ranks,
                               std::vector<int>& items_ids
                               )
{

    unsigned long words = b_size / bytes_segm_length +1;

    int actual_part1 = actual_block_size / 2;
    int actual_part2 = actual_block_size - actual_part1;

    unsigned long offset =   (unsigned long)(block_id * block_size)* words;
    int length = actual_block_size * words;


    thrust::device_vector<int> a_out_ids_1( actual_part1 * n );
    thrust::device_vector<float> a_out_ranks_1( actual_part1 * n );
    thrust::device_vector<unsigned char> a_filter_device_1( words * block_size );
    thrust::copy(user_likes.begin()+offset,  user_likes.begin()+offset + length, a_filter_device_1.begin());

    dim3 block1(BLOCK_X_SIZE, BLOCK_SIZE);
    dim3 grid1(1, 1 + actual_part1 / BLOCK_SIZE);

    select<<<grid1, block1>>>(thrust::raw_pointer_cast(&c_device[0]),
                            thrust::raw_pointer_cast(&b_id_device[0]),
                            thrust::raw_pointer_cast(&a_out_ranks_1[0]),
                            thrust::raw_pointer_cast(&a_out_ids_1[0]),
                            thrust::raw_pointer_cast(&a_filter_device_1[0]),
                            n, b_size, actual_part1);



    if ( cudaSuccess != cudaGetLastError() )
        std::cerr <<  "!WARN - Cuda error : "  << cudaGetLastError() << std::endl;

    cudaSetDevice(1);

    thrust::device_vector<int> a_out_ids_2( actual_part2 * n );
    thrust::device_vector<float> a_out_ranks_2( actual_part2 * n );
    thrust::device_vector<unsigned char> a_filter_device_2( words * block_size );
    thrust::copy(user_likes.begin()+offset,  user_likes.begin()+offset + length, a_filter_device_2.begin());

    dim3 block2(BLOCK_X_SIZE, BLOCK_SIZE);
    dim3 grid2(1, 1 + actual_part2 / BLOCK_SIZE);

    select<<<grid2, block2>>>(thrust::raw_pointer_cast(&c_device2[0]),
                            thrust::raw_pointer_cast(&b_id_device2[0]),
                            thrust::raw_pointer_cast(&a_out_ranks_2[0]),
                            thrust::raw_pointer_cast(&a_out_ids_2[0]),
                            thrust::raw_pointer_cast(&a_filter_device_2[0]),
                            n, b_size, actual_part2);



    if ( cudaSuccess != cudaGetLastError() )
        std::cerr <<  "!WARN - Cuda error : "  << cudaGetLastError() << std::endl;



    thrust::copy(a_out_ids_2.begin(), a_out_ids_2.end(), items_ids.begin()+ (block_id * block_size + actual_part1) * n );
    thrust::copy(a_out_ranks_2.begin() , a_out_ranks_2.end(),  items_ranks.begin() + (block_id * block_size + actual_part1) * n);


    cudaSetDevice(0);

    thrust::copy(a_out_ids_1.begin(), a_out_ids_1.end(), items_ids.begin()+ (block_id * block_size) * n );
    thrust::copy(a_out_ranks_1.begin() , a_out_ranks_1.end(),  items_ranks.begin() + (block_id * block_size) * n);

 /*
      unsigned long words = b_size / bytes_segm_length +1;
      thrust::device_vector<int> a_out_ids( actual_block_size * n );
      thrust::device_vector<float> a_out_ranks( actual_block_size * n );                 
      thrust::device_vector<unsigned char> a_filter_device( words * block_size );
      
      unsigned long offset =   (unsigned long)(block_id * block_size)* words;
      int length = actual_block_size * words;
      
      thrust::copy(user_likes.begin()+offset,  user_likes.begin()+offset + length, a_filter_device.begin());
      
      dim3 block(BLOCK_X_SIZE, BLOCK_SIZE);
      dim3 grid(1, 1 + actual_block_size / BLOCK_SIZE); 
           
      select<<<grid, block>>>(thrust::raw_pointer_cast(&c_device[0]),
                              thrust::raw_pointer_cast(&b_id_device[0]),
                              thrust::raw_pointer_cast(&a_out_ranks[0]),
                              thrust::raw_pointer_cast(&a_out_ids[0]),
                              thrust::raw_pointer_cast(&a_filter_device[0]),
                              n, b_size, actual_block_size);
      

      
      if ( cudaSuccess != cudaGetLastError() )
          std::cerr <<  "!WARN - Cuda error : "  << cudaGetLastError() << std::endl;
      
      thrust::copy(a_out_ids.begin(), a_out_ids.end(), items_ids.begin()+ (block_id * block_size) * n );
      thrust::copy(a_out_ranks.begin() , a_out_ranks.end(),  items_ranks.begin() + (block_id * block_size) * n);

*/


}

void matrix_mul::process_block(thrust::device_vector<float>& c_device,
							   thrust::device_vector<float>& c_device2,
                               thrust::device_vector<int>& b_id_device,
                               thrust::device_vector<int>& b_id_device2,
                               int block_id,     
                               int n,
                               int block_size,
                               int actual_block_size,
                               int b_size, 
                               std::vector<float>& items_ranks,
                               std::vector<int>& items_ids
                               )
{
      unsigned long words = b_size / bytes_segm_length +1;      

      int actual_part1 = actual_block_size / 2;
      int actual_part2 = actual_block_size - actual_part1;
      
      thrust::device_vector<int> a_out_ids_1( actual_part1 * n );
      thrust::device_vector<float> a_out_ranks_1( actual_part1 * n );

      dim3 block1(BLOCK_X_SIZE, BLOCK_SIZE);
      dim3 grid1(1, 1 + actual_part1 / BLOCK_SIZE);
           
      select_no_filter<<<grid1, block1>>>(thrust::raw_pointer_cast(&c_device[0]),
                                        thrust::raw_pointer_cast(&b_id_device[0]),
                                        thrust::raw_pointer_cast(&a_out_ranks_1[0]),
                                        thrust::raw_pointer_cast(&a_out_ids_1[0]),
                                        n, b_size, actual_part1);
      
      
      if ( cudaSuccess != cudaGetLastError() )
          std::cerr <<  "!WARN - Cuda error : "  << cudaGetLastError() << std::endl;
      
      //thrust::host_vector<float> c_host(c_device);
      //thrust::host_vector<int> b_id_host(b_id_device);


      cudaSetDevice(1);

      //thrust::device_vector<float> c_device2(c_host);
      //thrust::device_vector<int> b_id_device2(b_id_host);


      thrust::device_vector<int> a_out_ids_2( actual_part2 * n );
      thrust::device_vector<float> a_out_ranks_2( actual_part2 * n );

      dim3 block2(BLOCK_X_SIZE, BLOCK_SIZE);
      dim3 grid2(1, 1 + actual_part2 / BLOCK_SIZE);

      select_no_filter<<<grid2, block2>>>(thrust::raw_pointer_cast(&c_device2[0]),
                                        thrust::raw_pointer_cast(&b_id_device2[0]),
                                        thrust::raw_pointer_cast(&a_out_ranks_2[0]),
                                        thrust::raw_pointer_cast(&a_out_ids_2[0]),
                                        n, b_size, actual_part2);


      if ( cudaSuccess != cudaGetLastError() )
          std::cerr <<  "!WARN - Cuda error : "  << cudaGetLastError() << std::endl;


      thrust::copy(a_out_ids_2.begin(), a_out_ids_2.end(), items_ids.begin()+ (block_id * block_size + actual_part1) * n );
      thrust::copy(a_out_ranks_2.begin() , a_out_ranks_2.end(),  items_ranks.begin() + (block_id * block_size + actual_part1) * n);

      cudaSetDevice(0);

      thrust::copy(a_out_ids_1.begin(), a_out_ids_1.end(), items_ids.begin()+ (block_id * block_size) * n );
      thrust::copy(a_out_ranks_1.begin() , a_out_ranks_1.end(),  items_ranks.begin() + (block_id * block_size) * n);


/*
      thrust::device_vector<int> a_out_ids( actual_block_size * n );
      thrust::device_vector<float> a_out_ranks( actual_block_size * n );

      dim3 block(BLOCK_X_SIZE, BLOCK_SIZE);
      dim3 grid(1, 1 + actual_block_size / BLOCK_SIZE);

      select_no_filter<<<grid, block>>>(thrust::raw_pointer_cast(&c_device[0]),
                                        thrust::raw_pointer_cast(&b_id_device[0]),
                                        thrust::raw_pointer_cast(&a_out_ranks[0]),
                                        thrust::raw_pointer_cast(&a_out_ids[0]),
                                        n, b_size, actual_block_size);


      if ( cudaSuccess != cudaGetLastError() )
          std::cerr <<  "!WARN - Cuda error : "  << cudaGetLastError() << std::endl;

      thrust::copy(a_out_ids.begin(), a_out_ids.end(), items_ids.begin()+ (block_id * block_size) * n );
      thrust::copy(a_out_ranks.begin() , a_out_ranks.end(),  items_ranks.begin() + (block_id * block_size) * n);
*/
}

///
/// in dest_ranks - row size = n
/// in_ranks - row size = b_size
///
__global__ void merge_lists(float* dest_ranks, int* dest_list, float* in_ranks, int* in_list, int n, int cusers, int offset)
{
    float s[queue_size];
    int s_id[queue_size];
    const int row = blockIdx.y * blockDim.y + threadIdx.y;   
    
    if( row >= cusers ) return;
    
    int row2 = (row + offset) * n;
    for( int i=0, l=0, r=0; i <  n && l < n && r < n;  i++)
    {
        const float lr = dest_ranks[row2 + l];
        const float rr = in_ranks[row2 + r];
        
        if( lr >=  rr )
        {
            s[i] = lr;
            s_id[i] = dest_list[row2 + l ];
            l++;
        }else
        {
            s[i]    = rr;
            s_id[i] = in_list[row2 + r ];
            r++;            
        }
    }

   for(int i=0; i <  n;  i++)
   {
       dest_ranks[row2 + i] = s[i];
       dest_list[row2 + i] = s_id[i];
   }
  
}
 
void part_matrix_mul::merge_recommends(
                                        std::vector<float>& items_ranks,
                                        std::vector<int>& items_ids,
                                        std::vector<int>& user_block_ids,
                                        int n,
                                        int cusers
                                       )
{

      int max_records = prop.totalGlobalMem / ( n * 16) * 0.9;

      max_records = std::min(max_records, cusers ); 
      
      int parts =  cusers / max_records;
      
      parts += (  (cusers % max_records) == 0)? 0 : 1;
      
      std::cerr << "Recommends merge parts: " << parts << " from records: " 
      << max_records << std::endl;
      
      thrust::device_vector<float> cu_merged_ranks1(n * max_records);
      thrust::device_vector<int> cu_merged_ids1(n * max_records);

      thrust::device_vector<float> cu_in_ranks1(n * max_records);
      thrust::device_vector<int> cu_in_ids1(n * max_records);



      
      std::vector<float> merged_ranks(n * cusers, 0);
      std::vector<int> merged_ids(n * cusers, 0);
      
      
      for(int i=0; i < cusers; i++)
      {
          int row = a_map[ user_block_ids[i] ];
          
          std::copy( user_top_ranks.begin() + row * n,  user_top_ranks.begin() + row * n + n, merged_ranks.begin()+i*n );        
          std::copy( out_ids.begin() + row * n,  out_ids.begin() + row * n + n, merged_ids.begin()+i * n );
          
      }
      
      for(int p=0; p < parts; p++)
      {
              
          
          int start = p * max_records;
          int p_size = ( p == parts -1 &&  (cusers % max_records) != 0)?   cusers % max_records : max_records;
          
          int s_offset = start * n;
          int e_offset = (start + p_size) * n; 
          int cp_length =  p_size * n;
          
          int p1 = p_size / 2;
	      int p2 = p_size - p1;


          std::cerr << "Merge part: " << p << " size:  " << p_size << std::endl;
          
          thrust::copy( items_ranks.begin() + s_offset, items_ranks.begin() + (start + p1) * n, cu_in_ranks1.begin());
          thrust::copy( items_ids.begin() + s_offset, items_ids.begin() + (start + p1) * n, cu_in_ids1.begin());

          thrust::copy( merged_ranks.begin() + s_offset,  merged_ranks.begin() + (start + p1) * n , cu_merged_ranks1.begin() );
          thrust::copy( merged_ids.begin()+ s_offset,  merged_ids.begin() + (start + p1) * n, cu_merged_ids1.begin() );

          cudaSetDevice(1);


          thrust::device_vector<float> cu_merged_ranks2(n * max_records);
		  thrust::device_vector<int> cu_merged_ids2(n * max_records);

		  thrust::device_vector<float> cu_in_ranks2(n * max_records);
  		  thrust::device_vector<int> cu_in_ids2(n * max_records);

          thrust::copy( items_ranks.begin() + (start + p1) * n, items_ranks.begin() + (start + p1 + p2) * n, cu_in_ranks2.begin() + p1 * n);
          thrust::copy( items_ids.begin() + (start + p1) * n, items_ids.begin() + (start + p1 + p2) * n, cu_in_ids2.begin() + p1 * n);

          thrust::copy( merged_ranks.begin() + (start + p1) * n,  merged_ranks.begin() + (start + p1 + p2) * n, cu_merged_ranks2.begin() + p1 * n );
          thrust::copy( merged_ids.begin() + (start + p1) * n,  merged_ids.begin() + (start + p1 + p2) * n, cu_merged_ids2.begin()  + p1 * n);


          
          cudaSetDevice(0);
          

          dim3 block(1, BLOCK_SIZE_2);
          dim3 grid(1, 1 + p1 / BLOCK_SIZE_2);
          


          merge_lists<<<grid, block>>>(thrust::raw_pointer_cast(&cu_merged_ranks1[0]),
                                       thrust::raw_pointer_cast(&cu_merged_ids1[0]),
                                       thrust::raw_pointer_cast(&cu_in_ranks1[0]),
                                       thrust::raw_pointer_cast(&cu_in_ids1[0]), n, p1, 0);

          cudaSetDevice(1);

          	dim3 block2(1, BLOCK_SIZE_2);
			dim3 grid2(1, 1 + p2 / BLOCK_SIZE_2);



			merge_lists<<<grid2, block2>>>(thrust::raw_pointer_cast(&cu_merged_ranks2[0]),
										 thrust::raw_pointer_cast(&cu_merged_ids2[0]),
										 thrust::raw_pointer_cast(&cu_in_ranks2[0]),
										 thrust::raw_pointer_cast(&cu_in_ids2[0]), n, p2, p1);




                                       
          if ( cudaSuccess != cudaGetLastError() )
            std::cerr <<  "!WARN - Cuda error : "  << cudaGetLastError() << std::endl;


          thrust::copy( cu_merged_ranks2.begin() + p1 * n, cu_merged_ranks2.begin() + (p1 + p2) * n, merged_ranks.begin() + s_offset + p1 * n  );
          thrust::copy( cu_merged_ids2.begin() + p1 * n, cu_merged_ids2.begin() + (p1 + p2) * n, merged_ids.begin()+ s_offset + p1 * n );

      	cudaSetDevice(0);

          thrust::copy( cu_merged_ranks1.begin(),cu_merged_ranks1.begin() + p1 * n, merged_ranks.begin() + s_offset  );
          thrust::copy( cu_merged_ids1.begin(), cu_merged_ids1.begin() + p1 * n, merged_ids.begin()+ s_offset );
  
     } 
                                
    for(int i=0; i < cusers; i++)
    {
      int row = a_map[ user_block_ids[i] ];
      std::copy( merged_ranks.begin()+i*n, merged_ranks.begin()+i*n + n, user_top_ranks.begin() + row * n );
      std::copy( merged_ids.begin()+i*n, merged_ids.begin()+i*n + n, out_ids.begin() + row * n );
    } 

/*
	      int max_records = prop.totalGlobalMem / ( n * 16) * 0.9;

	      max_records = std::min(max_records, cusers );

	      int parts =  cusers / max_records;

	      parts += (  (cusers % max_records) == 0)? 0 : 1;

	      std::cerr << "Recommends merge parts: " << parts << " from records: "
	      << max_records << std::endl;

	      thrust::device_vector<float> cu_merged_ranks(n * max_records);
	      thrust::device_vector<int> cu_merged_ids(n * max_records);

	      thrust::device_vector<float> cu_in_ranks(n * max_records);
	      thrust::device_vector<int> cu_in_ids(n * max_records);

	      std::vector<float> merged_ranks(n * cusers, 0);
	      std::vector<int> merged_ids(n * cusers, 0);


	      for(int i=0; i < cusers; i++)
	      {
	          int row = a_map[ user_block_ids[i] ];

	          std::copy( user_top_ranks.begin() + row * n,  user_top_ranks.begin() + row * n + n, merged_ranks.begin()+i*n );
	          std::copy( out_ids.begin() + row * n,  out_ids.begin() + row * n + n, merged_ids.begin()+i * n );

	      }

	      for(int p=0; p < parts; p++)
	      {


	          int start = p * max_records;
	          int p_size = ( p == parts -1 &&  (cusers % max_records) != 0)?   cusers % max_records : max_records;

	          int s_offset = start * n;
	          int e_offset = (start + p_size) * n;
	          int cp_length =  p_size * n;

	          std::cerr << "Merge part: " << p << " size:  " << p_size << std::endl;

	          thrust::copy( items_ranks.begin() + s_offset, items_ranks.begin() + e_offset, cu_in_ranks.begin());
	          thrust::copy( items_ids.begin() + s_offset, items_ids.begin() + e_offset, cu_in_ids.begin());

	          thrust::copy( merged_ranks.begin() + s_offset,  merged_ranks.begin() + e_offset , cu_merged_ranks.begin() );
	          thrust::copy( merged_ids.begin()+ s_offset,  merged_ids.begin() + e_offset, cu_merged_ids.begin() );

	          dim3 block(1, BLOCK_SIZE_2);
	          dim3 grid(1, 1 + p_size / BLOCK_SIZE_2);

	          merge_lists<<<grid, block>>>(thrust::raw_pointer_cast(&cu_merged_ranks[0]),
	                                       thrust::raw_pointer_cast(&cu_merged_ids[0]),
	                                       thrust::raw_pointer_cast(&cu_in_ranks[0]),
	                                       thrust::raw_pointer_cast(&cu_in_ids[0]), n, p_size, 0);

	          if ( cudaSuccess != cudaGetLastError() )
	            std::cerr <<  "!WARN - Cuda error : "  << cudaGetLastError() << std::endl;

	          thrust::copy( cu_merged_ranks.begin(),cu_merged_ranks.begin() + cp_length, merged_ranks.begin() + s_offset  );
	          thrust::copy( cu_merged_ids.begin(), cu_merged_ids.begin() + cp_length, merged_ids.begin()+ s_offset );

	     }

	    for(int i=0; i < cusers; i++)
	    {
	      int row = a_map[ user_block_ids[i] ];
	      std::copy( merged_ranks.begin()+i*n, merged_ranks.begin()+i*n + n, user_top_ranks.begin() + row * n );
	      std::copy( merged_ids.begin()+i*n, merged_ids.begin()+i*n + n, out_ids.begin() + row * n );
	    }

*/
}

void euclidian_norm_matrix_mul::calculate(std::ostream& output_stream, int n, int block_size)
{
    normalize(a, features_size, a_size);
    normalize(b, features_size, b_size);
    
    matrix_mul::calculate(output_stream, n, block_size);
}


void euclidian_norm_matrix_mul::normalize(std::vector<float>& b, int vector_length, int m_size)
{
    thrust::device_vector<float> c_device(b);
    dim3 block(NORM_BLOCK_SIZE, 1);
    dim3 grid(1+m_size / NORM_BLOCK_SIZE, 1);
    
    euclidian_normalize<<<grid, block>>>(thrust::raw_pointer_cast(&c_device[0]), m_size, vector_length);
    
    if ( cudaSuccess != cudaGetLastError() )
         std::cerr <<  "euclidian_norm_matrix_mul::normalize:: !WARN - Cuda error : "  << cudaGetLastError() << std::endl;
    
    thrust::copy( c_device.begin(),c_device.end(), b.begin()  );

}

void matrix_mul::print_c_matrix(float* c_device, int a_actual_size)
{
	float* c = (float*)malloc(a_actual_size * b_size * sizeof(float));
	cudaMemcpy(c, c_device, a_actual_size * b_size * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < a_actual_size; i++)
	{
		for (int j = 0; j < b_size; j++)
		{
			printf("%f ", c[i * b_size + j]);
		}
		printf("\n");
	}
	free(c);
}
