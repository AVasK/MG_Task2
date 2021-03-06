// AVasK - 2017 - Computer Graphics Course - Task 2

#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"

// Matrix
#include "matrix.h"
#include <tuple>

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef Matrix<std::tuple<uint, uint, uint>> Image; // for the sake of convenience.
typedef Matrix<int> Grayscale;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// C0NSTANTS:
const int SEGMENTS = 12;
const int ANGULAR_SEGMENTS = 12; 
const int CLR_SEGMENTS = 8;
const int LBP_SEGMENTS = 8;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}


/*
|||||||||||||||||||||||||||||||||||||||||||||||
*****  *****    *   *** **
** *   **  *   * *   ****
**  *  ****   *****   **
*****  **  * **   ** **
|||||||||||||||||||||||||||||||||||||||||||||||
*/

// Returns entity of type Grayscale (== Matrix<int>).
Grayscale to_grayscale(BMP image) {
    uint rows = image.TellWidth();
    uint cols = image.TellHeight();
    
	//cout << "size: " << cols << " : " << rows << "\n";
    Grayscale img(cols, rows);
    
    for (uint i = 0; i < rows; i++) {
        for (uint j = 0; j < cols; j++) {
            RGBApixel pixel = image.GetPixel(i,j);
            uint Y = 0.299 * pixel.Red + 0.587 * pixel.Green + 0.114 * pixel.Blue;
            img(j, i) = Y;
        }
    }
    return img;
}

Image BMP_to_Img(BMP image) {
	uint bmp_rows = image.TellWidth();
	uint bmp_cols = image.TellHeight(); // Messy BMP coordinates.
	
	Image temp(bmp_cols, bmp_rows);
	
	for (uint i = 0; i < bmp_rows; i++) {
        for (uint j = 0; j < bmp_cols; j++) {
            RGBApixel pixel = image.GetPixel(i,j);
            temp(j, i) = std::tie(pixel.Red, pixel.Green, pixel.Blue);
        }
    }
	return temp;
}

Image Gray_to_Img(Grayscale gray) {
    auto rows = gray.n_rows;
    auto cols = gray.n_cols;
    
    Image temp(rows, cols);
    
    for (uint r = 0; r < rows; r++) {
        for (uint c = 0; c < cols; c++) {
            temp(r,c) = std::tie(gray(r,c), gray(r,c), gray(r,c));
        }
    }
    return temp;
}

Image Map_to_Img(Matrix<double> gray) {
    auto rows = gray.n_rows;
    auto cols = gray.n_cols;
    
    Image temp(rows, cols);
    
    for (uint r = 0; r < rows; r++) {
        for (uint c = 0; c < cols; c++) {
            temp(r,c) = std::tie(gray(r,c), gray(r,c), gray(r,c));
        }
    }
    return temp;
}

// Copied from previous task's source code.
void save_image(const Image &im, const char *path)
{
    BMP out;
    out.SetSize(im.n_cols, im.n_rows);

    uint r, g, b;
    RGBApixel p;
    p.Alpha = 255;
    for (uint i = 0; i < im.n_rows; ++i) {
        for (uint j = 0; j < im.n_cols; ++j) {
            std::tie(r, g, b) = im(i, j);
            p.Red = r; p.Green = g; p.Blue = b;
            out.SetPixel(j, i, p);
        }
    }

    if (!out.WriteToFile(path))
        throw string("Error writing file ") + string(path);
}

// Histogram of Submatrices, calculated into HOG():

vector<float> hist(Matrix<double> Magn, Matrix<double> Dir) {
    vector<float> histogram(ANGULAR_SEGMENTS, 0);
    
    for (uint r = 0; r < Magn.n_rows; r++) {
        for (uint c = 0; c < Magn.n_cols; c++) {
            int partition = int(Dir(r,c) / double(360/ANGULAR_SEGMENTS));
            if (Dir(r, c) == 360) { partition = 0; } // LOOPed;
			partition = (partition > 0) ? partition : ANGULAR_SEGMENTS;
			partition = (partition < ANGULAR_SEGMENTS) ? partition : 0;
            //cout << Dir(r, c) << "-- Angle\n";
            //cout << partition << "<- PARTITION\n";
            //if (!std::isnan(Dir(r, c)) && !std::isnan(Magn(r, c))) {
                histogram[partition] += (Magn(r, c) <= 200) ? Magn(r, c)/100 : 10;
            //}
        }
        //cout << "Halt!\n";
		int i = 1;
        //for (auto elem : histogram) { cout << i++ << ": " << elem << "\n";};
    }
    return histogram;
}


// Calculating HOG of submatrices:

vector<float> hog(Matrix<double> Magn, Matrix<double> Dir) {
    vector<float> img_hog;
    
    uint rows = Magn.n_rows;
	uint cols = Magn.n_cols;
    
    for (uint i = 0; i < SEGMENTS; i++) {
        for (uint j = 0; j < SEGMENTS; j++) {
            auto hist_part = hist(
                Magn.submatrix(i*rows/SEGMENTS, j*cols/SEGMENTS, 
                    rows/SEGMENTS, cols/SEGMENTS),
                Dir.submatrix(i*rows/SEGMENTS, j*cols/SEGMENTS, 
                    rows/SEGMENTS, cols/SEGMENTS));
            
            for (auto &elem : hist_part) {
                img_hog.push_back(elem);
            }
        }
    }
    return img_hog;
}

// Additional Part 1: LBP:
/*
___          ____   ______    ___
|  \   /\    |   \  | || |     | 
|  |  /  \   |   |    ||       |
|--- /====\  |---     ||       |
|   /      \ | \\     ||      _|_

*/ 

// lazy normalizer
vector<float> normalize (vector<float> arr) { // like a Min-Max normalizer, but without a Min)))
    float max = -1;
    for (auto elem : arr) {
        if (elem > max) {
            max = elem;
        }
    }
    
    if (max != 0) {
        for (auto &elem : arr) {
            elem /= max;
        }
    }
    return arr;
}

class Lbp
{
public:
    int vert_radius = 1; // C++11 syntax;
    int hor_radius = 1;
		
	template<typename T>
    uint operator () (const Matrix<T> &m) const
    {
		 uint h_size = 2 * hor_radius + 1;
		 uint v_size = 2 * vert_radius + 1;

		 vector<uint> temp;
		 uint center = 0;
  		 for (uint i = 0; i < v_size; i++) {
  		     for (uint j = 0; j < h_size; j++) {
  		        if ((i != 1) || (j !=1)) {
					temp.push_back(std::get<0>(m(i, j)));
				} else {
					center = std::get<0>(m(i, j));
				}
  		     }
  		 } 
		 float ret = 0;
		 for (uint i = 0; i < temp.size(); i++) {
			if (temp[i] > center) {
				ret += pow(2, i);
			}
		 }
		
		return ret; 
    }
};

vector<float> histogram_lbp (Image m) 
{
	vector<float> hist(256, 0); // vector of 0 to 255 values, initialized with 0's.
	Matrix<uint> mat = m.unary_map(Lbp());
	for (uint i = 0; i < mat.n_rows; i++) {
		for (uint j = 0; j < mat.n_cols; j++) {
			hist[mat(i, j)]++;
		}
	}
	
	return hist;
}


vector<float> get_lbp (Image m)
{
	uint rows = m.n_rows;
	uint cols = m.n_cols;
	vector<float> ret;
	for (int i = 0; i < LBP_SEGMENTS; i++) {
		for (int j = 0; j < LBP_SEGMENTS; j++) {
			vector<float> hist = histogram_lbp(m.submatrix(i*rows/LBP_SEGMENTS, j*cols/LBP_SEGMENTS,       
                rows/LBP_SEGMENTS, cols/LBP_SEGMENTS));
			hist = normalize(hist);	
			ret.insert(ret.end(), hist.begin(), hist.end());
		}
	}
		
	return ret;
}


// Additional Part 2: Colour Features:
/*

 ____     ____    ___      ____    ____
//   \   /  _ \   | |     /  _ \   | > \
||       | | ||   | |     | | ||   |  _/
||       | |_||   | |__   | |_||   |  \
\\___/   \____/   |____|  \____/   | \_\

*/

vector<float> Avg_Colour(Image cell) {
	auto rows = cell.n_rows;
	auto cols = cell.n_cols;
	
	vector<float> avg(3, 0.0); // 3 elements, all inited w 0.
	int n = 0;
	
	for (uint r = 0; r < rows; r++) {
		for (uint c = 0; c < cols; c++) {
			avg[0] += std::get<0>(cell(r, c));
			avg[1] += std::get<1>(cell(r, c));
			avg[2] += std::get<2>(cell(r, c));
			n++;
		}
	}
	avg[0] /= n;
	avg[1] /= n;
	avg[2] /= n;
	
	return avg;
}
// Calculating Colour Features:
vector<float> colour_features(Image img) {
	vector<float> img_clr;
	auto rows = img.n_rows;
	auto cols = img.n_cols;
	
	for (uint i = 0; i < CLR_SEGMENTS; i++) {
        for (uint j = 0; j < CLR_SEGMENTS; j++) {
            auto avg = Avg_Colour(
                img.submatrix(i*rows/CLR_SEGMENTS, j*cols/CLR_SEGMENTS, 
                    rows/CLR_SEGMENTS, cols/CLR_SEGMENTS));
            
			for (auto elem : avg) {img_clr.push_back(elem/255);} // normalizing
        }
    }
	return img_clr;
}

/*
|||||||||||||||||||||||||||||||||||||||||||||||
***********************************************
|||||||||||||||||||||||||||||||||||||||||||||||
*****  *****  *****  *****  **
**     **  *  **  *  **     **
 **   **  *  *****  *****  **
 `**  **  *  **  *  **     **
*****  *****  *****  *****  ******
|||||||||||||||||||||||||||||||||||||||||||||||
***********************************************
|||||||||||||||||||||||||||||||||||||||||||||||
*/


// Exatract features from dataset.
// You should implement this function by yourself =)
// Roger that! 

void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        
        // PLACE YOUR CODE HERE
        // Remove this sample code and place your feature extraction code here
        
        // 0. Getting images & Grayscaling them:
        auto img = data_set[image_idx].first; // Should be (BMP *)
        Grayscale gray_img = to_grayscale(*img); // << GRAYSCALE IMAGE;
        
        // 1. Using sobel Vertical & Horizontal filters:
        
        /*
         /----------\
         | -1, 0, 1 |  <- HORIZONTAL
         \----------/
        */ 
        
        /* 
         
         ____
         |1 |
         |0 |  <- VERTICAL
         |-1|
         ----
        
        
        */
        /*
        Matrix<double> Vertical(3, 1);
        Vertical(0,0) = 1;
        Vertical(1,0) = 0;
        Vertical(2,0) = -1;
        Matrix<double> Horizontal = {-1, 0, 1};
        */
        
        // Actually making convolutions:
        auto rows = gray_img.n_rows;
        auto cols = gray_img.n_cols;
        
        Matrix<double> res_vert(rows, cols);
        Matrix<double> res_horiz(rows, cols);
        
        // VERTICAL:
		// r = 0: <MIRRORING EDGE-HANDLING> <if commented, gives "crop" method :) >
		for (uint c = 0; c < cols; c++) { // mirroring looks strange btw.
			res_vert(0, c) = gray_img(1, c) - gray_img(1, c);
		}
		
		
        for (uint r = 1; r < rows - 1; r++) {
            for (uint c = 0; c < cols; c++) {
                res_vert(r, c) = gray_img(r-1, c) - gray_img(r+1, c);
            }
        }
        
        // HORIZONTAL:
		// c = 0: <MIRRORING EDGE-HANDLING>
		for (uint r = 0; r < rows; r++) {
			res_horiz(r, 0) = gray_img(r, 1) - gray_img(r, 1);
		}
		
        for (uint r = 0; r < rows; r++) {
            for (uint c = 1; c < cols - 1; c++) {
                res_horiz(r, c) = gray_img(r, c+1) - gray_img(r, c-1);

            }
        }
        
        // Calculating MAGNITUDE & DIRECTION.
        Matrix<double> magnitude(rows, cols);
        Matrix<double> direction(rows, cols);
        
        for (uint r = 0; r < rows - 1; r++) {
            for (uint c = 0; c < cols - 1; c++) {
                double g_x = res_horiz(r, c);
                double g_y = res_vert(r, c);
                magnitude(r, c) = std::sqrt(g_x * g_x + g_y * g_y);
                direction(r, c) = int(180/3.1415926 * std::atan2(g_y, g_x)) % 360;
				direction(r, c) = (direction(r, c) > 0.0) ? direction(r, c) : (360.0 + direction(r, c));
            }
        }
        
        //DEBUGGING STAGE:
        Image temp(Map_to_Img(magnitude));
        save_image(temp, "magn.bmp");
        Image temp2(Map_to_Img(direction));
        save_image(temp2, "dir.bmp");
        //Image temp(Map_to_Img(res_vert));
        //save_image(temp, "temp.bmp");
        //================

		// 1 Rad = 180/pi Deg
		// 1 Deg = pi/180 Rad
		
        /*
		double max = 0;
		for (uint r = 0; r < rows - 1; r++) {
			for(uint c = 0; c < cols - 1; c++) {
				cout << int(direction(r, c)) << " ";
				if (direction(r,c) > max) max = direction(r,c);
			}
			cout << '\n';
		}
        
		cout << "max angle (deg) " << max;
        */
        
        // 2. Calculating HOG:
        auto mfv = hog(magnitude, direction);
		for (auto elem : mfv) {
			//cout << elem << ", ";
		}
        
        // 3. 
        
        
        vector<float> one_image_features = hog(magnitude, direction);
		auto clr = colour_features(BMP_to_Img(*img));
		one_image_features.insert(one_image_features.end(), clr.begin(), clr.end());
        
        auto lbp = get_lbp(BMP_to_Img(*img));
        one_image_features.insert(one_image_features.end(), lbp.begin(), lbp.end());
        //one_image_features.push_back(1.0);
        features->push_back(make_pair(one_image_features, data_set[image_idx].second));
        // End of sample code
        

    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}