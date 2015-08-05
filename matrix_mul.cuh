#ifndef MATRIX_MUL_CUH
#define MATRIX_MUL_CUH

#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <thrust/device_vector.h>
#include <cublas_v2.h>

///
///
/// Class for large matrix multiplication 
/// 
///
class matrix_mul
{
public:
    ///
    /// Ctor
    ///
	matrix_mul(const std::string& a_file_name, const std::string& b_file_name, int a_size,
			   int b_size, int features_size);
               
    virtual ~matrix_mul();
    
    ///
    /// Calculate matrix multiplication
    ///
	virtual void calculate(std::ostream& output_stream, int n, int block_size);
    
    ///
    /// Get device info
    ///
    const cudaDeviceProp& getDevice();
    
protected:

    ///
    /// Split matrix a by blocks and to do multiplication
    ///
    void mul_by_block(std::vector<float>& a,
                      std::vector<float>& b,
                      std::vector<int>& b_ids, 
                      int a_size,
                      int b_size,
                      int n, 
                      int block_size,
                      std::vector<float>& items_ranks,
                      std::vector<int>& items_ids                              
                      );

    ///
    /// result matrix is block_size x b_size
    /// n count results
    ///
    virtual void process_block(thrust::device_vector<float>& c_device,
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
                               );
    ///
    /// Read matrix from file
    /// matrix stored in row majority
    /// but we cat translate it to column majority
    ///
    void read_matrix(const std::string& file_name, 
                     int m_size, 
                     std::vector<float>& matrix, 
                     std::vector<int>& ids);

	void print_c_matrix(float* c_device, int a_actual_size);
	std::string a_file_name;
	std::string b_file_name;
	int a_size;
	int b_size;
	int features_size;    
    std::vector<float> b;
    std::vector<float> a;
    std::vector<int> b_ids;
    std::vector<int> a_ids;
    std::vector<int> out_ids;
    std::vector<float> out_ranks;
    
    /// cublas
    cublasHandle_t handle;
	cublasStatus_t status;    
    
    /// multiplication args
	float alpha;
	float beta;
    
    cudaDeviceProp prop;
    
};

///
/// Multiplicate large matrixies splited by part
///
class part_matrix_mul : public matrix_mul
{
    
public :
      ///
      /// a_file_name - path to file with A matrix 
      /// b_file_name - path to file with B matrix
      /// part_descr_file - file with description of parts
      /// a_size - size of a matrix
      /// b_size - size of b matrix
      /// features_size - size of feature vector
      ///
      /// 
      part_matrix_mul(const std::string& a_file_name, 
                      const std::string& b_file_name, 
                      const std::string& part_descr_file,
                      int a_size,
                      int b_size,
                      int features_size);
                      
      
      ///
      /// Calculate multiplication
      ///
      void calculate(std::ostream& output_stream, int n, int block_size);
      
      ///
      /// Set up skip_likes_filter flag
      ///
      void set_skip_likes_filer(bool val) { _skip_likes_filter = val; }
protected:

    ///
    /// result matrix is block_size x b_size
    /// n count results
    ///
    virtual void process_block(thrust::device_vector<float>& c_device,
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
                               );

                      
private:

    
    ///
    /// user lickes - filter of user lickes
    ///
    bool read_next_part_file(std::vector<int>& items, 
                             std::vector<int>& users,
                             std::vector<unsigned char>& user_likes);
    
    ///
    /// Map items identifier to item index
    ///
    void map_ids(const std::vector<int>& ids,                              
                  std::map<int, int>& i_map
                 );

    ///
    /// Merge block  recommends to memory
    ///                 
    void merge_recommends(
                           std::vector<float>& items_ranks,
                           std::vector<int>& items_ids,
                           std::vector<int>& user_block_ids,
                           int n,
                           int cusers
                          );
                 

private:
    std::string _parts_descr_file;
    std::ifstream m_part_stream;    
    int last_part;
    std::vector<int> last_users_set;
    std::vector<int> last_items;
    std::map<int, int> a_map;
    std::map<int, int> b_map;
    /// user lickes for block
    std::vector<unsigned char> user_likes; 
    
    /// user's ranks 
    std::vector<float> user_top_ranks;
    ///
    /// turn on/off likes filter.
    ///
    bool _skip_likes_filter;
};

///
/// multiplicate vectors with euklidian normalization
///
class euclidian_norm_matrix_mul : public matrix_mul
{
public:

    euclidian_norm_matrix_mul(const std::string& a_file_name, const std::string& b_file_name, int a_size,
			   int b_size, int features_size) : matrix_mul (a_file_name, b_file_name, a_size, b_size, features_size){};
               
    ///
    /// Calculate multiplication 
    /// with euclidian normalizaton
    ///               
    void calculate(std::ostream& output_stream, int n, int block_size);
protected:

    ///
    /// Normalization with cuda
    ///
    void normalize(std::vector<float>& b, int vector_length, int m_size);
};

#define NORM_BLOCK_SIZE 128
///
///  Global cuda normalization
///
__global__ void euclidian_normalize(float* c_device,                           
                                     int m_size, int vec_length);


#endif // MATRIX_MUL_CUH
