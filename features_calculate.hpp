#ifndef ___FEATURES_CALCULATE_H___
#define ___FEATURES_CALCULATE_H___

#include <string>
#include <vector>
#include <map>

///
///
///

class features_calculate
{
    typedef  std::pair<int, int> __key;
    typedef  std::map<int, std::vector<float> > __value;
    typedef  std::map<__key, __value > __data_type;
  public :
    
    ///
    /// ctor
    /// u_file - user  features matrix file
    /// i_file - items features -file
    /// in_f_list - input list to generate features
    /// <user> <ref_group> <reco group>
    ///
   features_calculate(std::string& u_file, 
                      int u_size,                      
                      std::string& i_file, 
                      int i_size,
                      std::string& in_f_list,
                      int f_size);
    ///
    ///
    ///
    void compute_features( );                      
    
    ///
    ///
    ///
    void serialize( std::ostream& out);
  protected:
  
    ///
    /// Read matrix
    ///
    void read_matrix(const std::string& file_name, 
                     std::vector<float>& matrix, 
                     std::vector<int>& ids,
                     std::vector<int>& pos);
                     
    ///
    ///
    ///                 
    void read_data_template(const std::string& file_name );
    
    ///
    /// Euclidian normalization
    ///
    void normalize_items( );
    
    ///
    ///
    ///
    float calculate_sim( int rgroup_idx, int group_idx);
    
    ///
    /// calculate prob
    ///
    float calculate_p( int u_idx, int group_idx);
    
 
    
  private:

    std::vector<int> u_id2pos;   /// map vector id to position                       
    std::vector<int> u_pos2id;   /// map vector position to vector id
    std::vector<float> users;    /// user's vector
    
    std::vector<int> i_id2pos;   /// map vector id to position                       
    std::vector<int> i_pos2id;   /// map vector position to vector id
    std::vector<float> items;    /// user's vector
    std::vector<float> norm_items;    /// user's vector
    
    __data_type  data;  /// users ref group, groups, features
    
    int features_size;
    int items_size;
    
};
#endif
