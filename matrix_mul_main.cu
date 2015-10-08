#include "matrix_mul.cuh"
#include "als.cuh"
#include <algorithm>
#include <string>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <ostream>
#include <vector>
#include <omp.h>
#include <sys/time.h>
#include "hashes.hpp"
#include "features_calculate.hpp"

using namespace std;
void ReplaceHashesInFile(std::string& file, std::string& out_file );

void print_help()
{
    std::cout << "Matrix multiplication tools:" << std::endl;
    std::cout << "MatrixMul [--als] [--replace-hashes --reco <file with recomendations>] [--likes <files with likes>] --a_file <user matrix file> --a_size <size of user matrix> \
    --b_file <item matrix file> \
    --b_size <size of item matrix> \
    --f_size <count features> [--out <output file>] [--csamples <count simples>] [--it <count iterations> --als-error <count samples for error>] [--count_gpus <count gpus>] \
    " << std::endl;
    std::cout << " --als  - calculate mtrix factorization by als" << std::endl;
    std::cout << " --likes  - files with user items likes for als" << std::endl;
    std::cout << " --als-error  - if specified error is calculated on each iteration. users:items" << std::endl;
    std::cout << " --replace-hashes  - replace all urls in reco file to hashes" << std::endl;
    std::cout << " --reco - file with recomendations" << std::endl;
    std::cout << " --likes-format [0|1] - format of input file of likes" << std::endl;
    std::cout << " --als-alfa  - alfa for als" << std::endl;
    std::cout << " --skip-likes-filter  - when multiplicate matrixies the likes filter will be ignored" << std::endl;
    std::cout << " --euclidian_norm  - when multiplicate matrixies the euclidian norm will be applied to each vector" << std::endl;
    std::cout << " --create-features  - create features from input triples <user> <ref_group> <group>/ use --p_file as input file path, a_file and b_file as files describes of matrixies " << std::endl;
    
}

int main(int argc, char *argv[] )
{

     
	string a_file_name; // = "/home/d.soloviev/ml/ok/recipes-all-users.txt";
	string b_file_name; // = "/home/d.soloviev/ml/ok/recipes-all-topics.txt";
    string p_file_name; // = "/home/d.soloviev/ml/ok/cluster_item_user.txt";
	string output_file_name; // = "out.txt";
    string likes_file_name;
    string reco_file;
	int a_size = 0; // = 17829029;
//       int a_size = 1782902; 
//        int a_size = 500;
	int b_size = 0; // = 5356624;
//        int b_size = 500;
	int features_size = 50;
	int block_size = 5000;
	int n = 100;
    int csimples = 0;
    int cit = 10;
    bool is_als=false;
    int samples_for_calc_error_users=0;
    int samples_for_calc_error_items=0;
    bool replace_hashes=false;
    int likes_format=0;
    float als_alfa = 5;
    bool skip_likes_filter = false;
    bool euclidian_normalize = false;
    bool create_features = false;
	int count_gpus = 0;
    cudaGetDeviceCount(&count_gpus);
    
    for( int i=1; i <  argc; i++)
    {
        std::string sarg = argv[i];
        if( sarg == "--a_file")
        {
            i++;
            a_file_name = argv[i];
            std::cerr << " A matrix:  " << a_file_name << std::endl;
        }else
        if( sarg == "--a_size")
        {
            i++;
            a_size = atoi(argv[i]);
            std::cerr << " A matrix size:  "<< a_size << std::endl;
        }else
        if( sarg == "--b_file")
        {
            i++;
            b_file_name = argv[i];
            std::cerr << " B matrix:  " << b_file_name << std::endl;

        }
        else 
        if( sarg == "--b_size")
        {
            i++;
            b_size = atoi(argv[i]);
            std::cerr << " B matrix size:  " << b_size << std::endl;            
        }else
        if( sarg == "--p_file")
        {
            i++;
            p_file_name = argv[i];
            std::cerr << " Cluster file:  " << p_file_name << std::endl;  
        }
        else 
        if( sarg == "--f_size")
        {
            i++;
            features_size = atoi(argv[i]);
            std::cerr << " Count features:  " << features_size << std::endl; 
        }else
        if( sarg == "--likes")
        {
            i++;
            likes_file_name = argv[i];
        }
        else
        if( sarg == "--csamples")
        {
            i++;
            csimples = atoi(argv[i]);
        }
        else
        if( sarg == "--skip-likes-filter")
        {
           skip_likes_filter = true; 
        }else        
        if( sarg == "--euclidian_norm")
        {
           euclidian_normalize = true; 
        }else
        if( sarg == "--als")
        {
           is_als = true; 
        }
        else 
        if( sarg == "--als-error")
        {
            i++;
            std::string samples(argv[i]);
            size_t pos = samples.find(":");
            if(pos == std::string::npos)   
             samples_for_calc_error_users  =  samples_for_calc_error_items =  atoi(argv[i]);
            else{
                samples_for_calc_error_users = atoi(samples.substr(0,pos).c_str());
                samples_for_calc_error_items = atoi(samples.substr(pos+1).c_str());
            }
        }else 
        if( sarg == "--it")
        {
            i++;
            cit = atoi(argv[i]);
        }else
        if( sarg == "--out")
        {
            i++;
            output_file_name = argv[i];
        }else
        if( sarg == "--reco")
        {
            i++;
            reco_file = argv[i];
        }else
        if( sarg == "--replace-hashes")
        {
            replace_hashes = true;
        }else
        if( sarg == "--likes-format")
        {
            i++;
            likes_format = atoi(argv[i]);
            
        }else
        if( sarg == "--als-alfa")
        {
            i++;
            als_alfa = atof(argv[i]);
            
        }else
        if( sarg == "--create-features")
        {
            create_features = true;            
        }else
        if( sarg == "--count_gpus")
        {
            i++;
			count_gpus = min(count_gpus, atoi(argv[i]));
        }
                
        if( sarg == "--help")
        {
            print_help();
            exit(0);
        }
    }

    if( create_features 
        && a_file_name.length() != 0
        && b_file_name.length() != 0
        && p_file_name.length() != 0 
        && features_size != 0 )
    {        
      std::cerr << "Start features calculation." << std::endl; 
      features_calculate fc(a_file_name, a_size, b_file_name, b_size, p_file_name, features_size);
      fc.compute_features();
      std::cerr << "Done." << std::endl;

       std::ofstream fout(output_file_name.c_str());
        
       std::ostream& out ((output_file_name.length() == 0)? std::cout : fout );
       std::cerr << "Start features serialization." << std::endl;

       fc.serialize(out);
        
       fout.close();      
       
       std::cerr << "Done." << std::endl;
      return 0;  
    }else if(create_features) {
        
        std::cerr << "Missing one or more input files or features_size" << std::endl;
        print_help();
        exit(1);        
    }
    
    
    if((!replace_hashes && !is_als && (a_file_name.length() == 0 ||
       b_file_name.length() == 0 ||
       (p_file_name.length() == 0 && !skip_likes_filter && !euclidian_normalize) ||
       a_size == 0 || b_size == 0 || features_size == 0 )) ||
       (is_als && features_size == 0 ) )
    {
        std::cerr << "Missing one or more arguments" << std::endl;
        print_help();
        exit(1);
    }
    

    
//	matrix_mul m(a_file_name, b_file_name, a_size, b_size, features_size);
    if(!is_als && !replace_hashes) 
    {
		if (count_gpus == 2)
		{
			int num_gpus = 2;
			std::vector<int> a_sizes(num_gpus);
			a_sizes[0] = a_size / 2;
			a_sizes[1] = a_size - a_sizes[0];

			std::vector<string> output_file_names(num_gpus);
			output_file_names[0] = output_file_name;
			output_file_names[1] = output_file_name + "_temp";

			omp_set_num_threads(num_gpus);
			#pragma omp parallel
			{
				unsigned int cpu_thread_id = omp_get_thread_num();
				unsigned int num_cpu_threads = omp_get_num_threads();

				// set and check the CUDA device for this CPU thread
				int gpu_id = -1;
				cudaSetDevice(cpu_thread_id);
				cudaGetDevice(&gpu_id);
				std::cerr << "CPU thread " << cpu_thread_id << " (of " << num_cpu_threads << ") uses CUDA device " << gpu_id << std::endl;
				std::cerr << "skip_likes_filter: " << skip_likes_filter << std::endl;
				std::cerr << "euclidian_normalize: " << euclidian_normalize << std::endl;

				int skip_lines = cpu_thread_id * a_sizes[0];
				std::auto_ptr<matrix_mul> m;
				if( !skip_likes_filter && !euclidian_normalize)  { m.reset(new  part_matrix_mul(a_file_name, b_file_name,
						p_file_name, a_sizes[cpu_thread_id], b_size, features_size, skip_lines)); }
				else if (!euclidian_normalize)
					{ m.reset(new matrix_mul (a_file_name, b_file_name, a_sizes[cpu_thread_id], b_size, features_size, skip_lines));}
					else { m.reset(new euclidian_norm_matrix_mul (a_file_name, b_file_name, a_sizes[cpu_thread_id], b_size, features_size, skip_lines));}

				std::ofstream fout(output_file_names[cpu_thread_id].c_str());

				std::ostream& out ((output_file_names[cpu_thread_id].length() == 0)? std::cout : fout );

				m->calculate(out, n, block_size);

				fout.close();

			}

			std::ofstream part_1(output_file_names[0].c_str(), std::ios_base::binary | std::ios_base::app);
			std::ifstream part_2(output_file_names[1].c_str(), std::ios_base::binary);

			part_1.seekp(0, std::ios_base::end);
			part_1 << part_2.rdbuf();

			part_1.close();
			part_2.close();

			remove(output_file_names[1].c_str());


		}
		else
		{
		    std::cerr << "skip_likes_filter: " << skip_likes_filter << std::endl;
		    std::cerr << "euclidian_normalize: " << euclidian_normalize << std::endl;

		    std::auto_ptr<matrix_mul> m;
		    if( !skip_likes_filter && !euclidian_normalize)  { m.reset(new  part_matrix_mul(a_file_name, b_file_name, p_file_name, a_size, b_size, features_size)); }
		    else if (!euclidian_normalize)
		        { m.reset(new matrix_mul (a_file_name, b_file_name, a_size, b_size, features_size));}
		        else { m.reset(new euclidian_norm_matrix_mul (a_file_name, b_file_name, a_size, b_size, features_size));}

		    std::ofstream fout(output_file_name.c_str());

		    std::ostream& out ((output_file_name.length() == 0)? std::cout : fout );

		    m->calculate(out, n, block_size);

		    fout.close();
		}

    }else if(is_als)
    {
       std::ifstream f_stream(likes_file_name.c_str() );
       std::istream& in ( (likes_file_name.length() == 0) ? std::cin : f_stream);
       
       std::cerr << " Count ALS iteration " << cit << std::endl;
       std::cerr << " Start Matrix Factorization - ALS " << std::endl;
       std::cerr << " Input file format -  " << likes_format << std::endl;
       std::cerr << " ALS alfa -  " << als_alfa << std::endl;
       std::cerr << " ALS count gpus -  " << count_gpus << std::endl;

       //30
       als als_alg(in, features_size, als_alfa, 30, csimples, samples_for_calc_error_users, samples_for_calc_error_items, likes_format, count_gpus);
       
       ///
       struct timeval t1;
       struct timeval t2;

       cudaDeviceSynchronize();
       gettimeofday(&t1, NULL);

       als_alg.calculate(cit);
       cudaDeviceSynchronize();
       gettimeofday(&t2, NULL);
       std::cout << "als calc time: " << t2.tv_sec - t1.tv_sec << std::endl;


       omp_set_num_threads(4);
#pragma omp parallel
       {
    	   int thread_id = omp_get_thread_num();
    	   if (thread_id == 0)
    	   {
    		   std::ofstream fout_users((output_file_name+".ufea").c_str());
			   als_alg.serialize_users(fout_users);
			   fout_users.close();
    	   }
    	   else if (thread_id == 1)
    	   {
    	       std::ofstream fout_items((output_file_name+".ifea").c_str());
    	       als_alg.serialize_items(fout_items);
    	       fout_items.close();
    	   }
    	   else if (thread_id == 2)
		   {
    	       std::ofstream fout_umap((output_file_name+".umap").c_str());
    	       als_alg.serialize_users_map(fout_umap);
    	       fout_umap.close();
		   }
    	   else if (thread_id == 3)
		   {
    	       std::ofstream fout_imap((output_file_name+".imap").c_str());
    	       als_alg.serialize_items_map(fout_imap);
    	       fout_imap.close();
		   }
       }



    }else if(replace_hashes)
    {
        /// 
        ///
        ///
        ReplaceHashesInFile(reco_file, output_file_name);
    }
    
	return 0;
}

void ReplaceHashesInFile(std::string& file, std::string& out_file )
{
   std::ifstream in_f(file.c_str());
   std::ofstream o_f(out_file.c_str());
   std::string line;
   char const tab_delim = '\t';
   std::istream& in(in_f.good()?in_f:std::cin);
   std::ostream& out(o_f.good()?o_f:std::cout);

   while(std::getline(in, line))
   {
      std::istringstream line_stream(line);
      std::string value;
      int i=0;
      while(getline(line_stream, value, tab_delim))
      {
          if( i== 0) 
          {
              out << value;
          }else
          {
              std::transform(value.begin(), value.end(), value.begin(), ::toupper);
              out << "\t" << hash_64(value.begin(), value.end());
          }
          i++;
      }
      
      out << std::endl;

   }
}
