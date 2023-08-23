//
// Created by stefano on 18/07/22.
//

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>

#include "include/evaluate.h"

using namespace std;
using namespace cv;


const string gt_folder = "Evaluation/images/det";
const string pred_folder = "Evaluation/images/our_det";
const string gt_masks_folder = "Evaluation/images/mask";
const string segmented_masks_folder = "Evaluation/images/our_mask";
const string images_dir = "Detection_Segmentation/rgb";



int main(int argc, char** argv){

    // -------------------------------- DETECTION ------------------------------------ //

    // Load ground truth bounding boxes 
	
    
	vector<vector<Rect>> gt_bounding_boxes = get_bbs(gt_folder);

    // Load bounding boxes from the classifier

    vector<vector<Rect>> predicted_bounding_boxes = get_bbs(pred_folder);


    // Load test set images

    vector<Mat> dataset_images = load_dataset(images_dir);

    vector<vector<double>> IoUs;
	
    //Compute IoU and draw BBs
    
	for (int i = 0; i < dataset_images.size(); i++){
        
		vector<double> IoUtemp;
        
		for (int j = 0; j < gt_bounding_boxes[i].size(); j++){

            double iou = bb_intersection_over_union(gt_bounding_boxes[i][j], predicted_bounding_boxes[i][j], dataset_images[i]);

            rectangle(dataset_images[i], gt_bounding_boxes[i][j],
                      Scalar(0,255,0), 3,8,0);  //GT bounding boxes

            rectangle(dataset_images[i], predicted_bounding_boxes[i][j],
                      Scalar(0,0,255), 3,8,0); // Predicted bounding boxes

            string s = "IoU: " + to_string(iou);

            putText(dataset_images[i],s,Point(gt_bounding_boxes[i][j].x,gt_bounding_boxes[i][j].y-3),
                    FONT_HERSHEY_COMPLEX,0.65,Scalar(0,255,0), 1);

            IoUtemp.push_back(iou);
        }
		if(argv[1]){
			imshow("Detection", dataset_images[i]);
			waitKey(0);	
		}
		if(argv[2]==string("save")){
			char title [100]; 
			string s;
			s = sprintf(title, "Evaluation/images/output_detection/%02d.jpg",i+1);
			cout << "Printing image " << i<<endl;
			imwrite(title, dataset_images[i]);
		}
        IoUs.push_back(IoUtemp);
    }
	
	for (int i = 0; i < IoUs.size(); i++){
		
		cout << "Image# " << i << "	";
               
		for (int j = 0; j < IoUs[i].size(); j++){
	
			cout << IoUs[i][j] << "		";
		}
		cout << endl;
	}
			
	vector<double> avg_IoU_single_image;
	
	for(int i = 0; i < IoUs.size(); i++){
		double acc_single_image = 0.0;
		for(int j = 0; j < IoUs[i].size(); j++){
			acc_single_image += IoUs[i][j];
		}
		acc_single_image = acc_single_image / IoUs[i].size();
		cout << "AVG IoU Image #" << i+1 << ": ----- "<< acc_single_image << " -------" << endl;
		avg_IoU_single_image.push_back(acc_single_image);
	}
	
	double avg_IoU = 0.0;
	
	for(int i = 0; i < avg_IoU_single_image.size(); i++){
		avg_IoU += avg_IoU_single_image[i]; 		
	}
	
	avg_IoU = avg_IoU / avg_IoU_single_image.size();
	
	cout << "AVG IoU Total: ----- "<< avg_IoU << " -------" << endl;
	

	/*--------------------------------------------------------------------------------------
	|																						|	
	|																						|
	|																						|
	|																						|	
    // -------------------------------- SEGMENTATION ------------------------------------ //
	|																						|	
	|																						|
	|																						|
	|																						|	
	 -------------------------------------------------------------------------------------*/
	
	// Load GT masks

    vector<Mat> gt_masks = load_dataset(gt_masks_folder);
    vector<Mat> segmented_masks = load_dataset(segmented_masks_folder);
 
	// Compute pixel accuracy of segmentation
	
	vector<double> accuracies;
	
    for( int  i = 0; i < gt_masks.size(); i++){
        
		int hand_pixels = 0;
        int non_hand_pixels = 0;

		Mat gt_mask = gt_masks[i];
		Mat segm_mask = segmented_masks[i]; 
        for(int r = 0; r < gt_mask.rows; r++){
			for(int c = 0; c < gt_mask.cols; c++){
				if(segm_mask.at<uchar>(r,c) == 0 && gt_mask.at<uchar>(r,c) == 0) non_hand_pixels++;
				if(segm_mask.at<uchar>(r,c) == 255 && gt_mask.at<uchar>(r,c) == 255) hand_pixels++;
				
			}
		}
		//cout << hand_pixels<<endl;
		//cout << non_hand_pixels<<endl;
		double pixel_accuracy = (double)(hand_pixels + non_hand_pixels) / (gt_mask.rows*gt_mask.cols);
		cout << "Pixel Accuracy Image #" << i << ": ------ " << pixel_accuracy << " ------" << endl;
		accuracies.push_back(pixel_accuracy);
	}
    
	double avg_accuracy = 0.0;
    
	for( int  i = 0; i < accuracies.size(); i++){
		avg_accuracy += accuracies[i];
	}
	
	avg_accuracy = avg_accuracy / accuracies.size();
	
	cout << "Average Pixel Accuracy For Segmentation: ------" << avg_accuracy << " ------" << endl; 
	
	return 0;
}