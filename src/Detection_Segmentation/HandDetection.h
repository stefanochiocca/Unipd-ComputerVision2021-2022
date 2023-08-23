#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <memory>
#include "meanshift/MeanShift.h"

using namespace std;
using namespace cv;

Mat image, image_clone;
struct DetectionParams {
    Mat input_image;
    int min_neighbors = 0;
    int min_size = 120;
    int kernel_bandwidth = 50;
    int min_detections = 40;
    int min_percentage = 3;
    string file_number;
};

void segmendAndColorHands(vector<Rect> &rects, Mat &img, string &file_number);
void saveOurDetFile(vector<Rect> &rects, string &file_number);
void saveOurMaskFile(Mat &mask, string &file_number);
void saveOurColoredFile(Mat &image, string &file_number);
void saveOurDetectionFile(Mat &image, string &file_number);
void doProcessImage( DetectionParams *params );

/** Global variables */
CascadeClassifier cascadeClassifier;
vector<String> classifiers;
//red, green, blue, yellow, black
const vector<Scalar> colors = {Scalar(0,0,255),Scalar(0,255,0),Scalar(255,0,0),Scalar(0,255,255),Scalar(0,0,0)};
const Scalar black = Scalar(0, 0, 0);
const Scalar white = Scalar(255, 255, 255);
const vector<string> file_numbers = {"01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"};
const string image_path = "Detection_Segmentation/rgb/", image_extension = ".jpg";
const string our_det_path = "Detection_Segmentation/our_det/", our_det_extension = ".txt";
const string our_mask_path = "Detection_Segmentation/our_mask/", our_mask_extension = ".png";
const string our_colored_path = "Detection_Segmentation/our_colored/", our_colored_extension = ".jpg";
const string our_detection_path = "Detection_Segmentation/our_detection/", our_detection_extension = ".jpg";


double dist(double ax, double ay,double bx, double by){
    return sqrt(pow((ax - bx),2) + pow((ay - by),2));
}