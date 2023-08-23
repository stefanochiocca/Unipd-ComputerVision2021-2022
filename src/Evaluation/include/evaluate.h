#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

double bb_intersection_over_union(Rect bb1, Rect bb2, Mat img);

vector<int> convert_str_to_arr(string str);

vector<vector<Rect>> get_bbs(string folder);

vector<Mat> load_dataset(string dir);
