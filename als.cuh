#ifndef MATRIX_FACTORIZATION_ALS_CUH
#define MATRIX_FACTORIZATION_ALS_CUH

#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <thrust/device_vector.h>
#include <cublas_v2.h>


#define CM_IDX(i,j,ld) (((j)*(ld))+(i))
///
/// Class for large matrix multiplication 
/// 
///
class als
{
    
public:
    ///
    /// Definition of features vector
    ///
    typedef std::vector<float> features_vector;
    typedef thrust::device_vector<float> features_vector_device;
    typedef std::vector< std::vector<int> >     likes_vector;
    typedef std::vector< std::vector<float> >   likes_weights_vector;
    ///
    /// Ctor
    /// Inputs are:
    /// stream with triplets:
    /// count_users - count users in stream
    /// count_items - count items
    /// count_features - count latent features
    /// format of likes 
    /// 0 - old
    /// 1 - simple
    /// <user> <item> <weight>
    ///
	als( std::istream& tuples_stream, 
         int count_features,
         float alfa,
         float gamma,
         int count_samples,
         int count_error_samples_for_users,
         int count_error_samples_for_items,
         int likes_format,
         int count_gpus = 1);
               
    virtual ~als();
    
    ///
    /// Calculate als (Matrix Factorization)
    /// in
    /// count_iterations - count iterations
    ///
	virtual void calculate(int count_iterations);
    
    ///
    /// Get device info
    ///
    const cudaDeviceProp& getDevice();
    
    ///
    /// Get Items features vector 
    ///
    const features_vector& get_features_items() const { return _features_items; }
    int get_count_items() const { return _count_items; }
    
    ///
    /// Get Users features vector 
    ///
    const features_vector& get_features_users() const { return _features_users; }
    int get_count_users() const { return _count_users; }
    
    void serialize(std::ostream& out);
    void serialize_map(std::ostream& out, std::map<unsigned long, int>& out_map);
    void serialize_matrix(std::ostream& out, const float* mat, int crow, int ccol, bool id = false);
    void serialize_vector(std::ostream& out, const float* mat, int size);
    void serialize_users(std::ostream& out);
    void serialize_items(std::ostream& out);
    void serialize_users_map(std::ostream& out);
    void serialize_items_map(std::ostream& out);
    
    void calc_error();
protected:

    ///
    /// Read likes from stream
    /// if format == 0
    /// user group item
    /// if format == 1
    /// user item weight
    ///
    void read_likes(  std::istream& tuples_stream, int count_simples, int format );
    
    ///
    /// solve one iteration of als
    ///
    void solve(
                const likes_vector::const_iterator& likes,
                const likes_weights_vector::const_iterator& weights,
                const features_vector& in_v,
                int _count_users,
                features_vector& out_v,
                int out_size,
                int out_full_size,
                int _count_features_local,
                int features_local_offset = 0,
                int out_offset = 0
               );
               
    ///
    /// Solve one part of iteration
    ///           
    void solve_part(
                    const likes_vector::const_iterator& likes,
                    const likes_weights_vector::const_iterator& weights,
                    const features_vector& in_v,
                    int in_size,
                    cublasHandle_t& handle, cublasStatus_t& status,
                    features_vector& out_v,
                    int out_size,
                    int out_full_size,
                    int out_offset
                   );
                   
    ///
    /// fill random values to features matrix
    ///               
    void fill_rnd(
                  features_vector& in_v,
                  int in_size
                 );
   ///
   /// calculate const general matrix for iteration
   /// result is stored to device_YxY
   ///                 
   void mulYxY(
                const features_vector& in_v,
                int in_size,
                cublasHandle_t& handle,
                cublasStatus_t& status,
                int _count_features_local,
                int features_local_offset
               );
   ///               
   /// Draw samples to calculate iteration error (MSE)
   ///
   void draw_samples_for_error(features_vector& users, features_vector& items);
   
   ///
   /// Calculate als (Matrix Factorization) with one GPU
   /// in
   /// count_iterations - count iterations
   ///
   virtual void calculate_one_gpu(int count_iterations);

   ///
   /// Calculate als (Matrix Factorization) with multiple GPUS
   /// in
   /// count_iterations - count iterations
   ///
   virtual void calculate_multiple_gpus(int count_iterations);


private :    
    ///
    /// features vectors, for users and items
    ///
    features_vector _features_users;
    int _count_users;
    features_vector _features_items;
    int _count_items;
    
    int count_gpus;

	int _count_features;
    ///
    /// 
    ///
    
    ///
    /// cublas information
    ///
    cublasHandle_t handle;
	cublasStatus_t status;    
    
    ///
    /// Internal data
    ///
    std::map<unsigned long, int> _users_map;
    std::map<unsigned long, int> _items_map;
///    likes_vector              _item_user;
    likes_vector                 _user_likes;
    likes_weights_vector         _user_likes_weights;
    likes_vector                 _item_likes;
    likes_weights_vector         _item_likes_weights;
    
    features_vector YxY;


    /// multiplication args
	float alpha;
	float beta;
    
    float _als_alfa;
    float _als_gamma;
    
    ///
    /// Count samples for calculate error
    ///
    int _count_error_samples_for_users; 
    std::vector<int>   users_for_error;
    int _count_error_samples_for_items;
    std::vector<int>   items_for_error;
    
    
    cudaDeviceProp prop;
    
};


#endif // MATRIX_MUL_CUH

