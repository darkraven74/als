#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <limits>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cublas_v2.h>
#include <set>
#include <ctime>
#include <omp.h>
#include <iostream>
#include "als.cuh"
#include <curand.h>


als::als( std::istream& tuples_stream, 
          int count_features,
          float alfa,
          float gamma,
          int count_samples,
          int count_error_samples_for_users,
          int count_error_samples_for_items,
          int likes_format,
          int count_gpus) :
          
          _count_users(0),
          _count_items(0),
          _count_features(count_features),          
          status(cublasCreate(&handle)),
          _als_alfa(alfa),
          _als_gamma(gamma),
          alpha(1),
          beta(0),
          _count_error_samples_for_users(count_error_samples_for_users),
          _count_error_samples_for_items(count_error_samples_for_items),
          count_gpus(count_gpus)
          
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
   
   srand(time(NULL));

   read_likes(tuples_stream, count_samples, likes_format);

   _features_users.assign(_count_users * _count_features, 0 );
   _features_items.assign(_count_items * _count_features, 0 );
   YxY.assign(_count_features * _count_features, 0);

//   generate_test_set();

}
 

als::~als()
{
  status = cublasDestroy(handle);
}

void als::read_likes(  std::istream& tuples_stream, int count_simples, int format )
{
    std::string line;
    char const tab_delim = '\t';
    int i=0;
    
    while( getline(tuples_stream, line) )
    {
        std::istringstream line_stream(line);
        std::string value;
        getline(line_stream, value, tab_delim);
        unsigned long uid = atol(value.c_str());
        if ( _users_map.find(uid)  == _users_map.end() )
        {
            _users_map[uid] = _count_users;
            _count_users ++;
            _user_likes.push_back(std::vector<int>() );
            _user_likes_weights.push_back(std::vector<float>() );
            _user_likes_weights_temp.push_back(std::vector<float>());
        }
        
        int user = _users_map[uid];
        
        if( format == 0 )
        {
//          getline(line_stream, value, tab_delim);
          unsigned long gid = atol(value.c_str());
        }
        
        getline(line_stream, value, tab_delim);        
        unsigned long iid = atol(value.c_str());
        float weight=1;
        
        float weight_temp = 1;

        if( format ==1 )
        {
           getline(line_stream, value, tab_delim);  
           weight_temp = atof( value.c_str() );
        }
        
        if ( _items_map.find(iid)  == _items_map.end() )
        {
            _items_map[iid] = _count_items;
            _item_likes.push_back(std::vector<int>() );
            _item_likes_weights.push_back(std::vector<float>() );
            _count_items ++;
        }
        
        int item = _items_map[iid];
        ///
        /// adding data to user likes 
        /// and to item likes
        ///
        _user_likes[user].push_back( item );        
        _user_likes_weights[user].push_back( weight );
    	_user_likes_weights_temp[user].push_back( weight_temp );
        _item_likes[item].push_back( user );
        _item_likes_weights[item].push_back( weight );
        
        if (i % 10000 == 0 ) std::cerr << i << " u: " << _count_users << " i: " << _count_items << "\r";
        
        ///std::cout << "u:" << user << " -> " << item << std::endl; 
        ///std::cout << "i:" << item << " -> " << user << std::endl; 
   
        i++;
        if( count_simples && i > count_simples) break;
    }
    
    std::cerr << " u: " << _count_users << " i: " << _count_items << std::endl;
}

void als::generate_test_set()
{
	for (int i = 0; i < _count_users; i++)
	{
		int size = _user_likes[i].size();
		for (int j = 0; j < size / 2;)
		{
			int id = rand() % _user_likes[i].size();
			if (_user_likes_weights_temp[i][id] < 4)
			{
				continue;
			}
			test_set.push_back(std::make_pair(i, _user_likes[i][id]));


			for (int k = 0; k < _item_likes[_user_likes[i][id]].size(); k++)
			{
				if (_item_likes[_user_likes[i][id]][k] == i)
				{
					_item_likes[_user_likes[i][id]].erase(_item_likes[_user_likes[i][id]].begin() + k);
					_item_likes_weights[_user_likes[i][id]].erase(_item_likes_weights[_user_likes[i][id]].begin() + k);
				}
			}


			_user_likes[i].erase(_user_likes[i].begin() + id);
			_user_likes_weights[i].erase(_user_likes_weights[i].begin() + id);
			_user_likes_weights_temp[i].erase(_user_likes_weights_temp[i].begin() + id);
			break;
		}
	}
}

void als::calculate(int count_iterations)
{
   fill_rnd(_features_users, _count_users);
   fill_rnd(_features_items, _count_items);
   
   if (count_gpus > 1)
   {
	   calculate_multiple_gpus(count_iterations);
   }
   else
   {
	   calculate_one_gpu(count_iterations);
   }

   /// serialize(std::cout);

   /// calc_error();
}

void als::calculate_one_gpu(int count_iterations)
{
	for(int i =0; i < count_iterations; i++)
	{
		time_t start =  time(0);
		std::cerr << "ALS Iteration: " << i << std::endl;

		std::cerr << "Items." << std::endl;
		solve(_item_likes.begin(), _item_likes_weights.begin(), _features_users, _count_users, _features_items, _count_items, _count_items, _count_features );
		std::cerr << "Users." << std::endl;
		solve(_user_likes.begin(), _user_likes_weights.begin(), _features_items, _count_items, _features_users, _count_users, _count_users, _count_features );

		time_t end =  time(0);
		std::cerr << "==== Iteration time : " << end - start << std::endl;

//		hit_rate();
		//calc_error();
	}
}

void als::calculate_multiple_gpus(int count_iterations)
{
	int _count_features_first_part = _count_features / count_gpus;
	int _count_features_last_part = _count_features - _count_features_first_part * (count_gpus - 1);

	int _count_items_first_part = _count_items / count_gpus;
	int _count_items_last_part = _count_items - _count_items_first_part * (count_gpus - 1);

	int _count_users_first_part = _count_users / count_gpus;
	int _count_users_last_part = _count_users - _count_users_first_part * (count_gpus - 1);

	std::vector<int> _count_features_parts(count_gpus, _count_features_first_part);
	_count_features_parts.back() = _count_features_last_part;

	std::vector<int> _count_items_parts(count_gpus, _count_items_first_part);
	_count_items_parts.back() = _count_items_last_part;

	std::vector<int> _count_users_parts(count_gpus, _count_users_first_part);
	_count_users_parts.back() = _count_users_last_part;

	std::vector<int> features_offsets(count_gpus, 0);
	std::vector<int> items_offsets(count_gpus, 0);
	std::vector<int> users_offsets(count_gpus, 0);

	for (int i = 1; i < count_gpus; i++)
	{
		features_offsets[i] = features_offsets[i - 1] + _count_features_parts[i - 1];
		items_offsets[i] = items_offsets[i - 1] + _count_items_parts[i - 1];
		users_offsets[i] = users_offsets[i - 1] + _count_users_parts[i - 1];
	}

	omp_set_num_threads(count_gpus);

	for(int i =0; i < count_iterations; i++)
	{
		time_t start =  time(0);
		std::cerr << "ALS Iteration: " << i << std::endl;

		// Items
		#pragma omp parallel
		{
			int thread_id = omp_get_thread_num();
			int gpu_id;
			cudaSetDevice(thread_id);
			cudaGetDevice(&gpu_id);
			std::cerr << "Items. Thread: " << thread_id << " GPU: " << gpu_id << std::endl;

			solve(_item_likes.begin() + items_offsets[thread_id], _item_likes_weights.begin() + items_offsets[thread_id], _features_users, _count_users, _features_items,
			   _count_items_parts[thread_id], _count_items, _count_features_parts[thread_id], features_offsets[thread_id], items_offsets[thread_id]);
		}

		// Users
		#pragma omp parallel
		{
			int thread_id = omp_get_thread_num();
			int gpu_id;
			cudaSetDevice(thread_id);
			cudaGetDevice(&gpu_id);
			std::cerr << "Users. Thread: " << thread_id << " GPU: " << gpu_id << std::endl;

			solve(_user_likes.begin() + users_offsets[thread_id], _user_likes_weights.begin() + users_offsets[thread_id], _features_items, _count_items, _features_users,
				   _count_users_parts[thread_id], _count_users, _count_features_parts[thread_id], features_offsets[thread_id], users_offsets[thread_id]);
		}

		time_t end =  time(0);

		std::cerr << "==== Iteration time : " << end - start << std::endl;

//		hit_rate();
		//calc_error();
	}
}


void als::solve(
                const likes_vector::const_iterator& likes,
                const likes_weights_vector::const_iterator& weights,
                const features_vector& in_v,
                int in_v_size,
                features_vector& out_v,
                int out_size,
                int out_full_size,
                int _count_features_local,
				int features_local_offset,
                int out_offset
               )
{
   cublasHandle_t handle;
   cublasStatus_t status;

   status = cublasCreate(&handle);


   ///
   /// calculate Y^TxY
   ///
   mulYxY(in_v, in_v_size, handle, status, _count_features_local, features_local_offset);

   #pragma omp barrier
   
   /// serialize_matrix(std::cout, &in_v[0], in_v_size, _count_features);
   
   ///
   /// Solve equals
   ///
   solve_part(likes, weights, in_v, in_v_size, handle, status, out_v, out_size, out_full_size, out_offset);
   
   status = cublasDestroy(handle);

   
    
}

#define RESERVED_MEM 0xA00000

void als::fill_rnd(
                    features_vector& in_v,
                    int in_size
                  )
{
    static long long sed = 1234ULL; 
    sed ++;
    
    size_t cuda_free_mem = 0;
    size_t cuda_total_mem = 0;
       
    cudaMemGetInfo(&cuda_free_mem, &cuda_total_mem);
    
    cuda_free_mem *= 0.8;
    
    std::cerr << "Rnd generator mem: " << cuda_free_mem << std::endl;
    int block_size = cuda_free_mem / (_count_features * sizeof(float) );
    block_size = ((block_size > in_size)? in_size : block_size);    
    
    std::cerr << "Rnd generator block size: " << block_size << std::endl;
    
    thrust::device_vector<float> c_device(block_size * _count_features);
    std::vector<float> tmp ( block_size * _count_features, 0 );
    
    int parts =  in_size / block_size + (((block_size % in_size) == 0) ? 0 : 1);
    
    std::cerr << "Rnd generator parts: " << parts << std::endl;
    
    
    curandGenerator_t gen;    
    
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    
    for( int i=0; i < parts; i ++){
        
        int actual_size = ( (i == (parts -1) && in_size % block_size > 0 ))?  in_size % block_size : block_size;
        
        curandSetPseudoRandomGeneratorSeed(gen, sed);
        
        curandGenerateNormal( gen, thrust::raw_pointer_cast(&c_device.front()), block_size * _count_features, 0, 1 );
        
        if ( cudaSuccess != cudaGetLastError() )
            std::cerr <<  "!WARN - Cuda error (als::fill_rnd -> curandGenerateNormal) : "  << cudaGetLastError() << std::endl;                         

        
        thrust::copy( c_device.begin(), c_device.end(), tmp.begin() );
        
        for(int j=0;  j < _count_features; j++)
        {
           int offset = j * in_size + i * block_size;
           std::copy(tmp.begin() + j * actual_size, 
                     tmp.begin() +  j * actual_size + actual_size,  in_v.begin() + offset );
        }
        
        sed++;
    }
    
    curandDestroyGenerator(gen);    
}

void als::draw_samples_for_error(features_vector& users, features_vector& items, std::vector<float>& r)
{
     srand (time(NULL));
     users.assign(_count_error_samples_for_users * _count_features,0);
     if( users_for_error.size() == 0 )
     {
         for( int i=0;  i < _count_error_samples_for_users; i++)
         {         
//             const int r1 = rand() % _count_users;
           const int r1 = i;
             users_for_error.push_back(r1);
         }
     }
     
     for( int i=0;  i < users_for_error.size(); i++)
     {         
         const int r1 = users_for_error[i];
         for( int c=0; c < _count_features; c++)
            users[CM_IDX(i, c, _count_error_samples_for_users)] = _features_users[CM_IDX(r1, c, _count_users)];
     }
     
     items.assign(_count_error_samples_for_items * _count_features,0);
     
     if( items_for_error.size() == 0 )
     {
         for( int i=0;  i < _count_error_samples_for_items; i++)
         {         
//             const int r1 = rand() % _count_items;
           const int r1 = i;
             items_for_error.push_back(r1);
         }
     }
     
     for( int i=0;  i < items_for_error.size(); i++)
     {         
         const int r1 = items_for_error[i];
         for( int c=0; c < _count_features; c++)
            items[CM_IDX(i, c, _count_error_samples_for_items)] = _features_items[CM_IDX(r1, c, _count_items)];
     }

     /*for (int i = 0; i < _count_error_samples_for_users; i++)
     {
    	 for (int j = 0; j < _count_error_samples_for_items; j++)
    	 {
    		 int user_id = users_for_error[i];
    		 int item_id = items_for_error[j];

    		 for (int k = 0; k < _user_likes[user_id].size(); k++)
    		 {
    			 if (_user_likes[user_id][k] == item_id)
    			 {
                                 r[j * _count_error_samples_for_users + i] = 1;
    			 }
    		 }
    	 }
     }*/
     
	/*for (unsigned int i = 0; i < test_set.size(); i++)
	{
		int user = test_set[i].first;
		int item = test_set[i].second;

		if (item < _count_error_samples_for_items)
			r[item * _count_error_samples_for_users + user] = 1;
	}*/

    for (int i = 0; i < _count_error_samples_for_users; i++)
    {
        for (unsigned int k = 0; k < _user_likes[i].size(); k++)
        {
                if (_user_likes[i][k] < _count_error_samples_for_items)
                        r[_user_likes[i][k] * _count_error_samples_for_users + i] = 1;
        }
    }
     
     
}

#define BLOCK_SIZE 8
///
///  a, b matrixies input
///  c - output mitrix 
///  a_size- a matrix size
///  features_size - features count
///  s3 - b matrix size
///
__global__ void matMulYxYGpuShared(float* Y, float* c, int features_size, int c_rows)
{
	__shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int parts = c_rows / BLOCK_SIZE + 1;
    float ans = 0;

	for (int id = 0; id < parts; id++)
	{
        const int x_idx = id * BLOCK_SIZE + threadIdx.x;
        const int y_idx = id * BLOCK_SIZE + threadIdx.y;
        
        if (y_idx < c_rows && x_idx < features_size )
        {
            a_shared[threadIdx.y][threadIdx.x] =  Y[y_idx * features_size + x_idx];
        }
        else
        {
            a_shared[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            ans += a_shared[k][threadIdx.y] * a_shared[k][threadIdx.x];
        }
        __syncthreads();

    }

    if (i < features_size && j < features_size)
    {
        c[i * features_size + j] = ans;
    }

}


///
/// calculate:
/// (Y^TxY+ Y^Tx(C-I)Y + gamma x I)
///
/// thread.z by features
/// thread.y by features
/// thread.x by matrix

#define IDX_3D(_x, _y, _z)  ((_x) * blockDim.y*blockDim.z + (_z) * blockDim.y + (_y))

///
/// Shared memory variant
///
__global__ void matMulYTxC_IxYGpuShared(float* Y,         /// the features matrix in collumn-major mode
                                        int* C,          /// the C - matrix (squared diagonal), only likes list, and diagonal elements
                                        float* wC,          /// the wC - matrix (squared diagonal), only weights of likes list, and diagonal elements
                                        float* YxY,       /// the Y^TxY matrix pointer
                                        int* c_offset,    /// array of offset of diagonal elements for each C matrix
                                        int* c_size,      /// array of sizes of diagonal elements for each C matrix
                                        int c_mat_count,  /// count of C matrixies
                                        float* pRs,       /// multiplication out is matrixes Y^TxY+ Y^Tx(C-I)Y + gamma x I) has size is features_sizexfeatures_sizexc_mat_count
                                        float alfa,       /// alfa coeff
                                        float gama,         /// regularization coeff 
                                        int features_size,/// count of features 
                                        int rows_per_block,        /// count of rows in Y matrix
                                        int Y_block_offset,      /// offset of cuirrent block
                                        int is_last_block   /// 1 - if current block is last
                                       )
{
    /// int y = blockIdx.y * blockDim.y + threadIdx.y;    
    /// int x = y / features_size; /// this is matrix index
    /// y = y % features_size;
    /// shared memory for likes
    __shared__ int   likes_shared[BLOCK_SIZE][BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float weghts_shared[BLOCK_SIZE][BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE * BLOCK_SIZE];
    
    
    int x = blockIdx.x * blockDim.x + threadIdx.x; /// this is matrix index
    
    ///
    /// multiplicate matrix YxZ
    /// Y is row number
    /// Z is feature number
    ///
    /// int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y; /// this is row index or feature index    
    int z = blockIdx.z * blockDim.z + threadIdx.z; /// row index
    
    ///
    /// up triangle matrix
    ///
    if( x >= c_mat_count ) return;
    
    float *R = pRs + ((( features_size * features_size)) * x );

    float ans = 0;
    int* plikes = C + c_offset[x];
    float* pweights = wC + c_offset[x];
    const int c_likes = c_size[x];
    
    int shared_block_size = BLOCK_SIZE * BLOCK_SIZE;    
    shared_block_size = (c_likes >= shared_block_size)? shared_block_size : c_likes;
    int count_blocks =  c_likes / shared_block_size;
    count_blocks = count_blocks + ((c_likes % shared_block_size > 0 )? 1:0);
    
    for (int i = 0; i < count_blocks; i++)
	{
        ///
        /// Copy likes to shared memory
        int idx_in_likes = i * BLOCK_SIZE * BLOCK_SIZE + threadIdx.z * BLOCK_SIZE + threadIdx.y;
        int shared_idx = threadIdx.z * BLOCK_SIZE + threadIdx.y;
        if(idx_in_likes < c_likes)
        {
           likes_shared[threadIdx.x][shared_idx] = plikes[idx_in_likes];
           weghts_shared[threadIdx.x][shared_idx] = pweights[idx_in_likes];
        }else{
            likes_shared[threadIdx.x][shared_idx] = -1;
            weghts_shared[threadIdx.x][shared_idx]= 0;
        }
        
        __syncthreads();
        
            int count_blocks_occuped = ((i + 1 == count_blocks && (c_likes % (BLOCK_SIZE * BLOCK_SIZE) > 0) )? (1+(c_likes % (BLOCK_SIZE * BLOCK_SIZE)) / BLOCK_SIZE) : BLOCK_SIZE);
            
        
            for(int j =0;  j < count_blocks_occuped; j++)
            {
                
              int s_idx = j * BLOCK_SIZE + threadIdx.y;
              int s_idx_y = j * BLOCK_SIZE + threadIdx.z;
              
              int idx = likes_shared[threadIdx.x][s_idx];
              int idx_y = likes_shared[threadIdx.x][s_idx_y];
             
                 
              if( z >= features_size || idx == -1 || idx < Y_block_offset  || idx >= Y_block_offset + rows_per_block)
              {
                  a_shared[threadIdx.x][threadIdx.z * BLOCK_SIZE + threadIdx.y] = 0;
              }
              else 
              {
                 int row_in_block = idx - Y_block_offset;
                 a_shared[threadIdx.x][threadIdx.z * BLOCK_SIZE + threadIdx.y] = Y[z * rows_per_block + row_in_block];                    
              }
             
              if( y >= features_size || idx_y == -1 || idx_y < Y_block_offset  || idx_y >= Y_block_offset + rows_per_block){
                b_shared[threadIdx.x][threadIdx.y * BLOCK_SIZE + threadIdx.z] = 0;
              }
              else {
                int row_in_block = idx_y - Y_block_offset;
                b_shared[threadIdx.x][threadIdx.y * BLOCK_SIZE + threadIdx.z] = Y[y * rows_per_block + row_in_block];
              }
                 
              __syncthreads();
              
              if(z < features_size && y < features_size && z <= y ) 
              {
                                    
                  for(int k=0; k < BLOCK_SIZE; k++)
                  {
                      int w_idx = j * BLOCK_SIZE + k;
                      float w = weghts_shared[threadIdx.x][w_idx];                      
                      ans += (1 + alfa * w) * a_shared[threadIdx.x][threadIdx.z * BLOCK_SIZE + k] * b_shared[threadIdx.x][threadIdx.y * BLOCK_SIZE + k];
                  }
              }
              __syncthreads();                         
            }
            
        __syncthreads();
    }
    
    if (y < features_size && z < features_size && z <= y)
    {
        /// save matrix in collumn-major
        ans += R[CM_IDX(z,y,features_size)];
        
        if( is_last_block == 1)
        {
          ans += YxY[y * features_size + z];
          /// regularization
          if( z == y ) ans += gama;
        }
        
        /// store matrix to column major
        R[CM_IDX(z,y,features_size)] =  ans;
        R[CM_IDX(y,z,features_size)] =  ans;
    }  
}


///
/// Variant without shared memory
///
__global__ void matMulYTxC_IxYGpu(float* Y,         /// the features matrix in collumn-major mode
                                        int* C,          /// the C - matrix (squared diagonal), only likes list, and diagonal elements
                                        float* wC,          /// the wC - matrix (squared diagonal), only weights of likes list, and diagonal elements
                                        float* YxY,       /// the Y^TxY matrix pointer
                                        int* c_offset,    /// array of offset of diagonal elements for each C matrix
                                        int* c_size,      /// array of sizes of diagonal elements for each C matrix
                                        int c_mat_count,  /// count of C matrixies
                                        float* pRs,       /// multiplication out is matrixes Y^TxY+ Y^Tx(C-I)Y + gamma x I) has size is features_sizexfeatures_sizexc_mat_count
                                        float alfa,       /// alfa coeff
                                        float gama,         /// regularization coeff 
                                        int features_size,/// count of features 
                                        int rows_per_block,        /// count of rows in Y matrix
                                        int Y_block_offset,      /// offset of cuirrent block
                                        int is_last_block   /// 1 - if current block is last
                                       )
{
    /// int y = blockIdx.y * blockDim.y + threadIdx.y;    
    /// int x = y / features_size; /// this is matrix index
    /// y = y % features_size;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x; /// this is matrix index
    
    ///
    /// multiplicate matrix YxZ
    /// Y is row number
    /// Z is feature number
    ///
    /// int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y; /// this is row index or feature index    
    int z = blockIdx.z * blockDim.z + threadIdx.z; /// row index
    
    ///
    /// up triangle matrix
    ///
    if( x >= c_mat_count || z > y) return;
    
    float *R = pRs + ((( features_size * features_size)) * x );

    float ans = 0;
    int* plikes = C + c_offset[x];
    float* pweights = wC + c_offset[x];
    const int c_likes = c_size[x];
    
    
    for (int i = 0; i < c_likes; i++)
	{
        int idx = plikes[i];
        if( idx < Y_block_offset  || idx >= Y_block_offset + rows_per_block ) continue;
        
        int row_in_block = idx - Y_block_offset;
        float w = pweights[i];
        if(z < features_size && y < features_size ) 
         ans += (1 + alfa * w) * Y[z * rows_per_block + row_in_block] * Y[y * rows_per_block + row_in_block];
    }
    
    if (y < features_size && z < features_size && z <= y)
    {
        /// save matrix in collumn-major
        ans += R[CM_IDX(z,y,features_size)];
        
        if( is_last_block == 1)
        {
          ans += YxY[y * features_size + z];
          /// regularization
          if( z == y ) ans += gama;
        }
        
        /// store matrix to column major
        R[CM_IDX(z,y,features_size)] =  ans;
        R[CM_IDX(y,z,features_size)] =  ans;
    }  
}

///
/// calculate:
/// Y^TxCxp(u)
/// x - by features
/// y - by C matrixies
__global__ void matMulYTxCxpGpuShared(
                                      float* Y,         /// the features matrix in collumn-major mode
                                      int* C,           /// the C - matrix (squared diagonal), only likes list
                                      float* wC,        /// the wC - matrix (squared diagonal), only weights of likes list, and diagonal elements
                                      int* c_offset,    /// array of offset of diagonal elements for each C matrix
                                      int* c_size,      /// array of sizes of diagonal elements for each C matrix
                                      int c_mat_count,  /// count of C matrixies
                                      float* pRs,         /// result X vectors
                                      float alfa, 
                                      int features_size, /// count of features 
                                      int rows_per_block,    /// count of rows in block of Y matrix
                                      int Y_block_offset /// offset of cuirrent block
                                    )
{

    float ans = 0;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if( x >= features_size || y >= c_mat_count ) return;
    
    float *R = pRs + ( (( features_size )) * y );
    int* plikes = C + c_offset[y];
    float* pweghts = wC + c_offset[y];
    const int c_likes = c_size[y];
    
	for (int i = 0; i < c_likes; i++)
	{
        int idx = plikes[i];
        if( idx < Y_block_offset  || idx >= Y_block_offset + rows_per_block ) continue;
        
        float w = pweghts[i];
        int row_in_block = idx - Y_block_offset; 
        ans += Y[x * rows_per_block + row_in_block] * (1 + alfa * w);
    }

    if (x < features_size )
    {
        /// store matrix in column-major
        R[x] +=  ans;
    }

}

///
/// Check poiters at matriies
///
__global__ void check_poiters(
                               float* Y[],         
                               int f_size,         
                               int ld,
                               int m_size
                             )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if( z >= m_size ) return ;
    
    float *R = Y[z];
    
    if (x < f_size && y < f_size)
    {
        R[y*ld+ x] = x *1000 +y + z * 1000000;
    }

}


///
/// To do : multiplication by blocks
///
void als::mulYxY(
                 const features_vector& in_v,
                 int in_size,
                 cublasHandle_t& handle,
                 cublasStatus_t& status,
                 int _count_features_local,
                 int features_local_offset
               )
{
  features_vector_device device_YxY;
  device_YxY.assign(_count_features * _count_features, 0);
  float alpha = 1;
  float beta = 1;
  ///
  /// Calculate size of block for input matrix 
  /// input matreix is Y matrix
  ///
  size_t cuda_free_mem = 0;
  size_t cuda_total_mem = 0;
       
  cudaMemGetInfo(&cuda_free_mem, &cuda_total_mem);
  cuda_free_mem -= RESERVED_MEM;
  std::cerr << "Cuda memory YxY free: " << cuda_free_mem << std::endl;
  
       
  ///
  /// detect size of block of Y matrix
  ///
  int count_rows = cuda_free_mem / (_count_features *sizeof(float));
  
  count_rows = count_rows >= in_size? in_size:count_rows;        
  int parts_size = in_size / count_rows + ( (in_size  % count_rows != 0)? 1 : 0);
  thrust::device_vector<float> x_device(count_rows * _count_features, 0);
  
  ///dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  ///dim3 grid(1 + _count_features / BLOCK_SIZE, 1 + _count_features / BLOCK_SIZE); 
           
  ///matMulYxYGpuShared<<<grid, block>>>( thrust::raw_pointer_cast(&x_device[0]),
  ///                                     thrust::raw_pointer_cast(&device_YxY[0]),
  ///                                     _count_features,
  ///                                     in_size);
  for(int part=0; part < parts_size; part++)
  {
    int actual_part_size = ( part == parts_size-1 && in_size  % count_rows != 0) ?  in_size  % count_rows : count_rows;
    

    /// copy to memory
    for(int i=0; i < _count_features; ++i)
    {
       size_t offset = i* in_size + part * count_rows;             
       thrust::copy(in_v.begin()+ offset,  in_v.begin()+ offset + actual_part_size, x_device.begin() + i * actual_part_size) ;
    }
    
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, _count_features_local, _count_features, actual_part_size , &alpha,
                             thrust::raw_pointer_cast(&x_device[0]) + features_local_offset * actual_part_size, actual_part_size , thrust::raw_pointer_cast(&x_device[0]),
                             actual_part_size, &beta, thrust::raw_pointer_cast(&device_YxY[0]) + features_local_offset, _count_features);

    if ( cudaSuccess != cudaGetLastError() )
            std::cerr <<  "!WARN - Cuda error (als::mulYxY -> cublasSgemm) : "  << cudaGetLastError() << std::endl;                         
  }
  

  for(int i = 0; i < _count_features; i++)
  {
	  thrust::copy(device_YxY.begin() + i * _count_features + features_local_offset, device_YxY.begin() + i * _count_features + features_local_offset + _count_features_local,
	 		  YxY.begin() + i * _count_features + features_local_offset);
  }
                         
//   std::cout << "====== YxY === " << std::endl;
  
//   serialize_matrix( std::cout, &YxY.front(), _count_features, _count_features);
  
//   std::cout << "End of ====== YxY === " << std::endl;
                       
}

#define MEM_FOR_MATRIX_LIST 0x1F400000 
#define MEM_FOR_Y_MATRIX 0x40000000

void als::solve_part(
                      const likes_vector::const_iterator& likes,
                      const likes_weights_vector::const_iterator& weights,
                      const features_vector& in_v,
                      int in_size,
                      cublasHandle_t& handle,
                      cublasStatus_t& status,
                      features_vector& out_v,
                      int out_size,
                      int out_full_size,
                      int out_offset
                     )
{   

   features_vector_device device_YxY(YxY);
   ///
   /// Calculate size of single matrix YxY and count number of
   /// matrix in memory
   ///
   int bytes_YxY_matrix_size = _count_features * _count_features * sizeof(float);   
   int YxY_matrix_size = _count_features * _count_features;   
   int count_matrix_in_mem = (long)MEM_FOR_MATRIX_LIST / bytes_YxY_matrix_size ;
   int count_matrix = std::min(count_matrix_in_mem, out_size);       
   
   ///
   /// size of one block of matrxies
   ///
   int part_by_matrix = out_size / count_matrix   + ( ( out_size % count_matrix != 0)? 1 : 0);
   
   ///
   /// Calculate size of likes parts 
   ///
   long large_part = 0;
   size_t max_len = 0;
   for(int m_part=0;  m_part < part_by_matrix;   m_part++ )
   {
       int it_count_matrix = ( (m_part == part_by_matrix -1 && out_size % count_matrix > 0 )?  out_size % count_matrix : count_matrix );
       long l_size=0;
       long len=0;
       for(int i=0; i< it_count_matrix; i++)
       {
         l_size += (*(likes + m_part * count_matrix + i)).size();
         len += (*(likes + m_part * count_matrix + i)).size();
       }
       
       
       len /= it_count_matrix + 1;
       
       if( l_size > large_part)
       {
           large_part = l_size;
           max_len  = len;
       }
   }
   
   ///
   /// Calculate in bytes
   /// and real allowed matrix maybe restricted by count likes
   /// very dencity likes matrix
   ///
   large_part *= 8;
   if(  large_part > ( 0.5 * prop.totalGlobalMem - (count_matrix * bytes_YxY_matrix_size) ) ) 
   {
      std::cerr << "Likes block is too large: " << large_part << max_len << std::endl;
      long mem_for_likes = (0.5 * prop.totalGlobalMem - (count_matrix * bytes_YxY_matrix_size) );
      int real_allowed_matrix = mem_for_likes / (max_len * 8 );
      std::cerr << "Real alowed matrix: " << real_allowed_matrix << " vs " << count_matrix
                << " used mem: " << real_allowed_matrix * max_len * 8 << " mem_for_likes: " << mem_for_likes << std::endl;
      if( real_allowed_matrix < count_matrix) count_matrix = real_allowed_matrix;
      ///
      part_by_matrix = out_size / count_matrix   + ( ( out_size % count_matrix != 0)? 1 : 0);
   }
   
   std::cerr << "Total Parts - split by matrix: " << part_by_matrix << std::endl;
   
   ///
   /// Start iteration by blocks of matrixies
   ///
   for(int m_part=0;  m_part < part_by_matrix;   m_part++ )
   {
       int it_count_matrix = ( (m_part == part_by_matrix -1 && out_size % count_matrix > 0 )?  out_size % count_matrix : count_matrix ); 
       
       std::cerr << "Serve matrix block: " << m_part << " size:  " << it_count_matrix << std::endl;
       
       ///
       /// Allocate memory for output matrixies
       ///
       thrust::device_vector<float> matrix_list_device( it_count_matrix * YxY_matrix_size, 0 );
              
       ///
       /// add likes for matrixies 
       /// They re diagonal elements in matrix C
       ///
       std::vector<int> c_likes_list;          /// list of likes 
       std::vector<float> c_likes_weights_list;  /// weghts of likes
       std::vector<int> c_likes_list_offset;   /// offset of likes by items (matrix)
       std::vector<int> c_likes_list_count;    /// number of likes for matrix
       
       
       ///
       /// copy likes to memory
       ///
       for(int i=0; i< it_count_matrix; i++)
       {
          c_likes_list_offset.push_back(c_likes_list.size());
          c_likes_list.insert(c_likes_list.end(), (*(likes + m_part * count_matrix + i)).begin(), (*(likes + m_part * count_matrix + i)).end());
          c_likes_weights_list.insert(c_likes_weights_list.end(), (*(weights + m_part * count_matrix + i)).begin(), (*(weights + m_part * count_matrix + i)).end());
          c_likes_list_count.push_back((*(likes + m_part * count_matrix + i)).size());
       }
       
       thrust::device_vector<int> matrix_C_likes_device(  c_likes_list );
       thrust::device_vector<float> matrix_C_likes_weghts_device(  c_likes_weights_list );
       thrust::device_vector<int> c_likes_offset_device( c_likes_list_offset );
       thrust::device_vector<int> c_likes_size_device( c_likes_list_count );
       
       
       ///
       /// Calculate size of block for input matrix 
       /// input matreix is Y matrix
       ///
       size_t cuda_free_mem = 0;
       size_t cuda_total_mem = 0;
       
       cudaMemGetInfo(&cuda_free_mem, &cuda_total_mem);
       cuda_free_mem -= RESERVED_MEM;
       std::cerr << "Cuda free mem: " << cuda_free_mem << std::endl;
       
       ///
       /// detect size of block of Y matrix
       ///
       int count_rows = cuda_free_mem / (_count_features *sizeof(float));
       
       count_rows = count_rows >= in_size? in_size:count_rows;        
       int parts_size = in_size / count_rows + ( (in_size  % count_rows != 0)? 1 : 0);                     
       thrust::device_vector<float> y_device(count_rows * _count_features );
       std::vector<float> y_host(count_rows * _count_features, 0);
       
       ///
       /// multiplikate in batch mode
       /// x is dimension of count of matrixies 
       /// y is size of features  by rows
       /// z is size of features  by columns
       ///
       /// dim3 block(BLOCK_SIZE, BLOCK_SIZE);
       /// dim3 grid(1+ _count_features / BLOCK_SIZE , 1 + (it_count_matrix * _count_features) / BLOCK_SIZE); 
       dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
       dim3 grid(1+ it_count_matrix / BLOCK_SIZE , 1 + _count_features / BLOCK_SIZE, 1 + _count_features / BLOCK_SIZE);
       

       std::cerr << "Multiplicate step 1: " << parts_size << std::endl;

       ///
       /// first step is: 
       /// (Y^TxY+ Y^Tx(C-I)Y + gamma x I)=A
       /// otput is fxf
       
       std::time_t t = time(0);

       for(int part=0; part < parts_size;  part++)
       {
         
         std::cerr << " part: " << part << std::endl;
         std::cerr << " grid: " << grid.x << "," << grid.y <<"," << grid.z << std::endl;
         std::cerr << " block: " << block.x << "," << block.y <<"," << block.z << std::endl;
         
         int actual_part_size = ( part == parts_size-1 && in_size  % count_rows != 0) ?  in_size  % count_rows : count_rows;
         int row_offset = count_rows * part;
         
         std::cerr << " actual_part_size: " << actual_part_size << std::endl;
         
         /// copy to memory
         for(int i=0; i < _count_features; ++i)
         {
             size_t offset = i* in_size + part * count_rows;             
             thrust::copy(in_v.begin()+ offset,  in_v.begin()+ offset + actual_part_size, y_host.begin() + i * actual_part_size) ;
         }
         
         thrust::copy(y_host.begin(), y_host.end(), y_device.begin());
         

         matMulYTxC_IxYGpuShared<<<grid, block>>>(
                                                  thrust::raw_pointer_cast(&y_device[0]),
                                                  thrust::raw_pointer_cast(&matrix_C_likes_device[0]),
                                                  thrust::raw_pointer_cast(&matrix_C_likes_weghts_device[0]),
                                                  thrust::raw_pointer_cast(&device_YxY[0]),
                                                  thrust::raw_pointer_cast(&c_likes_offset_device[0]),
                                                  thrust::raw_pointer_cast(&c_likes_size_device[0]),
                                                  it_count_matrix,
                                                  thrust::raw_pointer_cast(&matrix_list_device[0]),
                                                  _als_alfa,                              
                                                  _als_gamma,
                                                  _count_features,
                                                  actual_part_size,
                                                  row_offset,
                                                  (part == parts_size-1)
                                                  );


        std::time_t t1 = time(0);
        
        cudaDeviceSynchronize();
        
        std::time_t t2 = time(0);
       
        std::cerr << " calculated. time: " << t2 - t1 << std::endl;
        
        if ( cudaSuccess != cudaGetLastError() )
            std::cerr <<  "!WARN - Cuda error (als::solve_part -> matMulYTxC_IxYGpu) : "  << cudaGetLastError() << std::endl;
                              
       }

       t = time(0) - t;
       std::cerr << " calculated. part 1 time: " << t << std::endl;



              
       std::vector<float> tmp_matrixies(matrix_list_device.size());       
       
       
       ///
       /// Save all calculated matrixies in temporary storage
       /// 
       ///
       
       thrust::copy(matrix_list_device.begin(), matrix_list_device.end(), tmp_matrixies.begin() );
       cuda_free_mem += matrix_list_device.size() + y_device.size();
       
       
       //std::cout << "====  Y^TxY+ Y^Tx(C-I)Y + gamma x I)=A  ====" << std::endl;
       
        //for( int k=0; k < it_count_matrix; k++)
        //{
         //std::cout << k << " ========\n ";
         //serialize_matrix(std::cout, &tmp_matrixies[k * YxY_matrix_size ], _count_features, _count_features);
        //}
       
        //std::cout << "==== End of:  Y^TxY+ Y^Tx(C-I)Y + gamma x I)=A  ====" << std::endl;
       
       ///
       /// Get new size for vectors list
       ///
       
       
       matrix_list_device.resize(0);
       matrix_list_device.shrink_to_fit();
       matrix_list_device.assign(_count_features * it_count_matrix, 0);
       
       y_device.resize(0);
       y_device.shrink_to_fit();
       
       
       
       cudaMemGetInfo(&cuda_free_mem, &cuda_total_mem);
       cuda_free_mem -= RESERVED_MEM;
       std::cerr << "Cuda free mem: " << cuda_free_mem << std::endl;
       
       ///
       /// detect size of block of Y matrix
       ///
       count_rows = cuda_free_mem / (_count_features *sizeof(float));
       
       count_rows = count_rows >= in_size? in_size:count_rows;        
       parts_size = in_size / count_rows + ( (in_size  % count_rows != 0)? 1 : 0);              
       y_device.assign(count_rows * _count_features, 0);
       
       dim3 block2(BLOCK_SIZE, BLOCK_SIZE);
       dim3 grid2(1 + _count_features / BLOCK_SIZE, 1+ it_count_matrix / BLOCK_SIZE ); 

       std::cerr << "Multiplicate step 2: " << parts_size << std::endl;
       ///
       /// second step is:
       /// Y^TxCxp(u)=B
       ///

       t = time(0);
       for(int part=0; part < parts_size;  part++)
       {
         std::cerr << "part: " << part << std::endl;
         int actual_part_size = ( ( part == parts_size-1 && in_size  % count_rows != 0) ?  in_size  % count_rows : count_rows);
         int row_offset = count_rows * part;
         
         std::cerr << "actual size: " << actual_part_size << " count rows: "<< count_rows << " offset " << row_offset <<  std::endl;
         
         /// copy to memory
         for(int i=0; i < _count_features; ++i)
         {
             size_t offset = i* in_size + part * count_rows;             
             thrust::copy(in_v.begin()+ offset,  in_v.begin()+ offset + actual_part_size, y_device.begin() + i * actual_part_size) ;
         }
         
         matMulYTxCxpGpuShared<<<grid2, block2>>> (
                                                      thrust::raw_pointer_cast(&y_device[0]),
                                                      thrust::raw_pointer_cast(&matrix_C_likes_device[0]),
                                                      thrust::raw_pointer_cast(&matrix_C_likes_weghts_device[0]),
                                                      thrust::raw_pointer_cast(&c_likes_offset_device[0]),
                                                      thrust::raw_pointer_cast(&c_likes_size_device[0]),
                                                      it_count_matrix,                                                  
                                                      thrust::raw_pointer_cast(&matrix_list_device[0]),
                                                      _als_alfa,
                                                      _count_features,
                                                      actual_part_size,
                                                      row_offset
                                                   );
         cudaDeviceSynchronize();
         
         if ( cudaSuccess != cudaGetLastError() )
            std::cerr <<  "!WARN - Cuda error (als::solve_part -> matMulYTxCxpGpuShared) : "  << cudaGetLastError() << std::endl;
           
       }
       
       t = time(0) - t;
       std::cerr << " calculated. part 2 time: " << t << std::endl;


       

       ///
       /// Save all calculated matrixies in temporary storage
       /// 
       ///
       
       cuda_free_mem = matrix_list_device.size() + y_device.size();
       ///
       /// Free part of memory
       ///
       matrix_C_likes_device.resize(0);
       matrix_C_likes_device.shrink_to_fit();
           
       matrix_C_likes_weghts_device.resize(0);
       matrix_C_likes_weghts_device.shrink_to_fit();
       
       c_likes_offset_device.resize(0);
       c_likes_offset_device.shrink_to_fit();
       
       c_likes_size_device.resize(0);
       c_likes_size_device.shrink_to_fit();       
       
       std::vector<float> tmp_matrixies_2(matrix_list_device.size());       
       thrust::copy(matrix_list_device.begin(), matrix_list_device.end(), tmp_matrixies_2.begin() );
       
       //std::cout << "============  Y^TxCxp(u)=B ===================" << std::endl;
       //for( int k=0; k < it_count_matrix; k++)
       //{
         //std::cout << "========\n ";
         //serialize_matrix(std::cout, &tmp_matrixies_2[k * _count_features ], _count_features, 1);
       //}
       //std::cout << "============ End of: Y^TxCxp(u)=B ===================" << std::endl;
       
       y_device.resize(0);
       y_device.shrink_to_fit();       
       
       
       ///
       /// store all pre calculated (Y^TxY+ Y^Tx(C-I)Y + gamma x I) matrixies
       /// to vector
       /// matrix_list_device - is a b matrix
       thrust::device_vector<float> a_matrix_device(tmp_matrixies);
       thrust::device_vector<float> inverce_a_device(_count_features * _count_features * it_count_matrix, 0);
       thrust::device_vector<float> x_device(_count_features * it_count_matrix, 0);

       std::vector<float*> device_pointers_a;
       std::vector<float*> device_pointers_b;
       std::vector<float*> device_pointers_c;
       std::vector<float*> device_pointers_x;
       
       for(int i=0; i < it_count_matrix; i++)
       {
           device_pointers_a.push_back(thrust::raw_pointer_cast(&a_matrix_device[i * YxY_matrix_size ]));
           device_pointers_c.push_back(thrust::raw_pointer_cast(&inverce_a_device[i * YxY_matrix_size ]));
           device_pointers_b.push_back(thrust::raw_pointer_cast(&matrix_list_device[i * _count_features ]) );
           device_pointers_x.push_back(thrust::raw_pointer_cast(&x_device[i * _count_features ]) );
       }
       
       thrust::device_vector<float*> a_matrix_device_ptrs(device_pointers_a);
       thrust::device_vector<float*> b_matrix_device_ptrs(device_pointers_b);
       thrust::device_vector<float*> c_matrix_device_ptrs(device_pointers_c);
       thrust::device_vector<float*> x_matrix_device_ptrs(device_pointers_x);
       thrust::device_vector<int> pivots_device(_count_features * it_count_matrix, 0);
              
       ///
       /// Getting matrix devices pointers
       ///
       
       ///
       /// third part is solving equals
       /// Ax=B
       ///
       /// inverce matrixies A in batch mode


       t = time(0);

       thrust::device_vector<int> info_device(it_count_matrix, 0);
       std::vector<int> infos(it_count_matrix, 0);
       //cublasStatus_t status;
              
       status = cublasSgetrfBatched(handle, 
                                      _count_features, 
                                      thrust::raw_pointer_cast(&a_matrix_device_ptrs[0]),                                    
                                      _count_features, thrust::raw_pointer_cast(&pivots_device[0]),
                                      thrust::raw_pointer_cast(&info_device[0]), it_count_matrix);

                                   
       cudaDeviceSynchronize();
       
       if ( cudaSuccess != cudaGetLastError() )
            std::cerr <<  "!WARN - Cuda error (als::solve_part -> cublasSgetrfBatched) : "  << cudaGetLastError() << std::endl;
        
            
       //std::cout << "======== cublasSgetrfBatched \n ";            
       //{ // DEBUG          
          //tmp_matrixies_2.assign(a_matrix_device.size(), 0 );
          //thrust::copy(a_matrix_device.begin(), a_matrix_device.end(), tmp_matrixies_2.begin() );
          //for( int k=0; k < it_count_matrix; k++)
          //{
            //std::cout << "========\n ";
            //serialize_matrix(std::cout, &tmp_matrixies_2[k * YxY_matrix_size ], _count_features, _count_features);
          //}

       //}
       
       //std::cout << "======== end of cublasSgetrfBatched \n ";            
                                                    
       thrust::copy(info_device.begin(), info_device.end(), infos.begin() ) ;                                       
       
       for(size_t inf=0; inf < infos.size() ; inf++)
         if( infos[inf] != 0 ) { std::cerr <<  "!WARN - Mat Operation error (als::solve_part -> cublasSgetrfBatched) : " << inf << std::endl; }

        status = cublasSgetriBatched(handle, 
                                     _count_features, 
                                     (float **)thrust::raw_pointer_cast(&a_matrix_device_ptrs[0]),
                                     _count_features, 
                                     thrust::raw_pointer_cast(&pivots_device[0]),
                                     thrust::raw_pointer_cast(&c_matrix_device_ptrs[0]), 
                                     _count_features, 
                                     thrust::raw_pointer_cast(&info_device[0]), it_count_matrix
                                     );
                                     
       cudaDeviceSynchronize();      
                                      
       thrust::copy(info_device.begin(), info_device.end(), infos.begin() ) ;                                       
       
       //std::cout << "======== cublasSgetriBatched \n ";            
       //{ // DEBUG          
          //tmp_matrixies_2.assign(inverce_a_device.size(), 0 );
          //thrust::copy(inverce_a_device.begin(), inverce_a_device.end(), tmp_matrixies_2.begin() );
          //for( int k=0; k < it_count_matrix; k++)
          //{
            //std::cout << "========\n ";
            //serialize_matrix(std::cout, &tmp_matrixies_2[k * YxY_matrix_size ], _count_features, _count_features);
          //}

       //}
       
       //std::cout << "======== end of cublasSgetriBatched \n ";            
       
       for(size_t inf=0; inf < infos.size() ; inf++)
         if( infos[inf] != 0 ) { std::cerr <<  "!WARN - Mat Operation error (als::solve_part -> cublasSgetriBatched) : " << inf << std::endl; }
                                     
       if ( cudaSuccess != cudaGetLastError() )
            std::cerr <<  "!WARN - Cuda error (als::solve_part -> cublasSgetriBatched) : "  << cudaGetLastError() << std::endl;
       
        status = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, _count_features, 1, _count_features, &alpha,
  				                    (const float **) thrust::raw_pointer_cast(&c_matrix_device_ptrs[0]), _count_features ,
                                    (const float **) thrust::raw_pointer_cast(&b_matrix_device_ptrs[0]),_count_features, 
                                    &beta, thrust::raw_pointer_cast(&x_matrix_device_ptrs[0]), _count_features, 
                                    it_count_matrix );
                                    
       cudaDeviceSynchronize();                                    
       if ( cudaSuccess != cudaGetLastError() )
            std::cerr <<  "!WARN - Cuda error (als::solve_part -> cublasSgemmBatched) : "  << cudaGetLastError() << std::endl;
                                    
       //std::cout << "======== cublasSgemmBatched (features) \n ";            
       //{ // DEBUG          
          //tmp_matrixies_2.assign(x_device.size(), 0 );
          //thrust::copy(x_device.begin(), x_device.end(), tmp_matrixies_2.begin() );
          //serialize_matrix(std::cout, &tmp_matrixies_2[0], _count_features, it_count_matrix);

       //}
       
        //std::cout << "======== end of cublasSgemmBatched (features) \n ";  
                 
       thrust::device_vector<float> transpose_device(x_device.size(), 0); 
       
       status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, 
                            it_count_matrix, _count_features, 
                            &alpha,
  				            thrust::raw_pointer_cast(&x_device[0]), _count_features ,
                            &beta,
                            thrust::raw_pointer_cast(&x_device[0]),_count_features, 
                            thrust::raw_pointer_cast(&transpose_device[0]), it_count_matrix
                           );
                           
        cudaDeviceSynchronize();

        t = time(0) - t;
        std::cerr << " calculated. part 3 time: " << t << std::endl;

       //std::cout << "======== cublasSgemmBatched -Transpose (features) \n ";            
       //{ // DEBUG          
          //tmp_matrixies_2.assign(transpose_device.size(), 0 );
          //thrust::copy(transpose_device.begin(), transpose_device.end(), tmp_matrixies_2.begin() );
          //serialize_matrix(std::cout, &tmp_matrixies_2[0], it_count_matrix, _count_features);

       //}
       
       //std::cout << "======== end of cublasSgemmBatched - Transpose (features) \n ";                             
                              
       ///
       /// copy data to output by collums
       ///
       int result_offset = m_part * count_matrix;
       std::vector<float> result_tmp(transpose_device.size() );
       thrust::copy( transpose_device.begin(), transpose_device.end(), result_tmp.begin() );
       
       for(int i=0; i < _count_features; i++)
       {
         int offset =  i * it_count_matrix;
         std::copy( result_tmp.begin() + offset, result_tmp.begin() + offset + it_count_matrix, out_v.begin() + i * out_full_size + out_offset + result_offset);
       }
       
       //std::cout << "======== cublas transpose \n ";            
       //{ // DEBUG          
          //serialize_matrix(std::cout, &result_tmp[0 ], it_count_matrix, _count_features);
       //}
       
       //std::cout << "======== cublas transpose === \n ";            
   }
   
   //std::cout << "======== Fetures \n ";            
   //{ // DEBUG          
      //serialize_matrix(std::cout, &out_v[0 ], out_size, _count_features);
   //}
   
   //std::cout << "======== end of Features === \n ";         
}

void als::serialize_vector(std::ostream& out, const float* mat, int size)
{
    
    for(int j=0; j < size;  j++ )
    {
       out << mat[ j ] << " ";
    }
           
    out << std::endl;
    
}

void als::serialize_users_map(std::ostream& out)
{
    serialize_map(out, _users_map);
}

void als::serialize_items_map(std::ostream& out)
{
    serialize_map(out, _items_map);
}

void als::serialize_map(std::ostream& out, std::map<unsigned long, int>& out_map)
{
    std::map<unsigned long, int>::iterator it = out_map.begin();
    for( ; it != out_map.end(); it++)
    {
        out << it->first << "\t" << it->second << std::endl;
    }
}

void als::serialize_items(std::ostream& out)
{
   const als::features_vector& items = get_features_items();
   serialize_matrix(out, &items.front(),  _count_items, _count_features, true); 
}

void als::serialize_users(std::ostream& out)
{
   const als::features_vector& users = get_features_users();

   serialize_matrix(out, &users.front(),  _count_users, _count_features, true);
}

void als::serialize_matrix(std::ostream& out, const float* mat, int crow, int ccol, bool id)
{
   if( ccol == 1) serialize_vector(out, mat, crow);
   else
   for(int i=0; i < crow;  i++ )
   {
       if(id) out << i << "\t";
         
       for(int j=0; j < ccol;  j++ ){
            out << mat[ CM_IDX(i, j, crow) ] << (( j == ccol-1)? "" : "\t");
        }
           
      out << std::endl;
   }
    
}

void als::serialize(std::ostream& out)
{
   const als::features_vector& items = get_features_items();
   const als::features_vector& users = get_features_users();

   
   std::cout << "Items features: " << std::endl;
   serialize_matrix(out, &items.front(),  _count_items, _count_features);
       
       
   out << "User features: " << std::endl;
   serialize_matrix(out, &users.front(),  _count_users, _count_features);
   
   
}
          
void als::calc_error()
{
  als::features_vector users;
  als::features_vector items;
  
  if(_count_error_samples_for_users == 0 || _count_error_samples_for_items == 0)
  {
    return;    
  }
  
  std::vector<float> r(_count_error_samples_for_items * _count_error_samples_for_users, 0);
  draw_samples_for_error(users, items, r);
  
  int final_matrix_size = _count_error_samples_for_users * _count_error_samples_for_items;
  thrust::device_vector<float> x_device(final_matrix_size, 0);
  thrust::device_vector<float> users_device(users);
  thrust::device_vector<float> items_device(items);
  
  float alpha = 1;
  float beta = 0;
 
           
  ///matMulYxYGpuShared<<<grid, block>>>( thrust::raw_pointer_cast(&x_device[0]),
  ///                                     thrust::raw_pointer_cast(&device_YxY[0]),
  ///                                     _count_features,
  ///                                     in_size);
  
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, _count_error_samples_for_users, _count_error_samples_for_items, _count_features, &alpha,
  				       thrust::raw_pointer_cast(&users_device[0]), _count_error_samples_for_users , thrust::raw_pointer_cast(&items_device[0]),
  				       _count_error_samples_for_items, &beta, thrust::raw_pointer_cast(&x_device[0]), _count_error_samples_for_users);
                       
  cudaDeviceSynchronize();                     
  if ( cudaSuccess != cudaGetLastError() )
            std::cerr <<  "!WARN - calc_error - cuda error: "  << cudaGetLastError() << std::endl;

  std::vector<float> mat(final_matrix_size, 0);
  float error = 0;
  thrust::copy(x_device.begin(), x_device.end(), mat.begin());
  float size = 0;
  for(int i=0; i < _count_error_samples_for_users * _count_error_samples_for_items; i++)                       
  {
		if (r[i] == 1)
		{
			size++;
			error += (r[i] - mat[i]) *  ( r[i] - mat[i] );
			//        error += (1 - mat[i]) *  ( 1 - mat[i] );
		}
  }
    
    std::cerr << "ERROR SUM: " << error << std::endl;
    
//    error = error / (float)(_count_error_samples_for_users * _count_error_samples_for_items);

    error = error / size;

//    std::cout << "MSE: " << error << std::endl;
    std::cout << error << std::endl;

    
    error = sqrtf(error / (float)(_count_error_samples_for_users * _count_error_samples_for_items) );
    
    std::cerr << "RMSE: " << error << std::endl;
}

void als::hit_rate()
{
	als::features_vector users;
	als::features_vector items;

	if(_count_error_samples_for_users == 0 || _count_error_samples_for_items == 0)
	{
		return;
	}

	std::vector<float> r(_count_error_samples_for_items * _count_error_samples_for_users, 0);
	draw_samples_for_error(users, items, r);

	int final_matrix_size = _count_error_samples_for_users * _count_error_samples_for_items;
	thrust::device_vector<float> x_device(final_matrix_size, 0);
	thrust::device_vector<float> users_device(users);
	thrust::device_vector<float> items_device(items);

	float alpha = 1;
	float beta = 0;


	///matMulYxYGpuShared<<<grid, block>>>( thrust::raw_pointer_cast(&x_device[0]),
	///                                     thrust::raw_pointer_cast(&device_YxY[0]),
	///                                     _count_features,
	///                                     in_size);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, _count_error_samples_for_users, _count_error_samples_for_items, _count_features, &alpha,
					   thrust::raw_pointer_cast(&users_device[0]), _count_error_samples_for_users , thrust::raw_pointer_cast(&items_device[0]),
					   _count_error_samples_for_items, &beta, thrust::raw_pointer_cast(&x_device[0]), _count_error_samples_for_users);

	cudaDeviceSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			std::cerr <<  "!WARN - calc_error - cuda error: "  << cudaGetLastError() << std::endl;

	std::vector<float> mat(final_matrix_size, 0);
	thrust::copy(x_device.begin(), x_device.end(), mat.begin());

	for (int i = 0; i < _count_users; i++)
	{
		for (unsigned int j = 0; j < _user_likes[i].size(); j++)
		{
			int item_id = _user_likes[i][j];
			mat[item_id * _count_users + i] = -1000000;
		}
	}

//	std::set<std::pair<int, int> > test_set_set(test_set.begin(), test_set.end());

	float sum = 0;
	std::set<std::pair<int, int> > recs;
	for (int i = 0; i < _count_users; i++)
	{
		std::vector<float> v;
		for (int j = 0; j < _count_items; j++)
		{
			v.push_back(mat[j * _count_users + i]);
		}

		for (int j = 0; j < 10; j++)
		{
			std::vector<float>::iterator it = std::max_element(v.begin(), v.end());
			int item = std::distance(v.begin(), it);
			v[item] = -1000000;
			recs.insert(std::make_pair(i, item));
			/*if (test_set_set.count(std::make_pair(i, item)))
			{
				sum += 1.0 / (j + 1);
				break;
			}*/
		}
	}


	float mrr = sum / _count_users;


//	float sum = 0;
	std::set<int> test_u;

	/*for (int u = 0; u < _count_users; u++)
	{

		float tp = 0;
		float size = 0;

		for (unsigned int i = 0; i < test_set.size(); i++)
		{
			int user = test_set[i].first;
			int item = test_set[i].second;

			test_u.insert(user);
			if (user == u)
			{
				size++;
			}

			if (user == u && recs.count(std::make_pair(user, item)))
			{
				tp++;
			}
		}

		if (size != 0)
			sum += tp / size;

	}*/

	//hit-rate10 calc
	std::set<std::pair<int, int> > test_set_set(test_set.begin(), test_set.end());
	float tp = 0;
	for (std::set<std::pair<int, int> >::iterator it = test_set_set.begin(); it != test_set_set.end(); it++)
	{
		if (recs.count(*it))
		{
			tp++;
		}
	}
	float hr10 = tp * 1.0 / test_set_set.size();

	//prec calc
	/*std::set<std::pair<int, int> > test_set_set(test_set.begin(), test_set.end());
	float tp = 0;
	for (std::set<std::pair<int, int> >::iterator it = recs.begin(); it != recs.end(); it++)
	{
		if (test_set_set.count(*it))
		{
			tp++;
		}
	}
	float p = tp * 1.0 / recs.size();
*/
//	float res = sum * 1.0 / test_u.size();

//	float res = tp * 1.0 / test_set.size();

	std::cout << hr10 << std::endl;
}
