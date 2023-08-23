#include "evaluate.h"


double bb_intersection_over_union(Rect bb1, Rect bb2, Mat img){
    
	// determine the (x, y)-coordinates of the intersection rectangle
    int x1, y1, x2, y2;
    x1 = max(bb1.x, bb2.x);
    y1 = max(bb1.y, bb2.y);
    x2 = min(bb1.x+bb1.width, bb2.x+bb2.width);
    y2 = min(bb1.y+bb1.height, bb2.y+bb2.height);
    
    // compute the area of intersection rectangle
    int interArea =  max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1);

    // compute the area of both the prediction and ground-truth rectangles
    int box1Area = ((bb1.x+bb1.width) - bb1.x + 1) * ((bb1.y+bb1.height) - bb1.y + 1);
    int box2Area = ((bb2.x+bb2.width) - bb2.x + 1) * ((bb2.y+bb2.height) - bb2.y + 1);

    //  compute the intersection over union by taking the intersection
    //	area and dividing it by the sum of prediction + ground-truth
    //	areas - the interesection area
    double iou = interArea / float(box1Area + box2Area - interArea);

    return iou;

}


vector<int> convert_str_to_arr(string str) {
    
	vector<int> bb;
    istringstream iss(str, istringstream::in);
    string word;
    vector<string> ints;
    while( iss >> word )
    {
        bb.push_back(stoi(word));
    }
    return bb;
}


vector<vector<Rect>> get_bbs(string folder){
	
    vector<String> fn;
    glob(folder, fn, true);
    vector<vector<Rect>> bounding_boxes;

    for( int i = 0; i < fn.size(); i++ ){
        fstream gt;
        gt.open(fn[i],ios::in);
        if (gt.is_open()){ //checking whether the file is open
            string tp;
            vector<Rect> temp;
            while(getline(gt, tp)) { //read data from file object and put it into string.
                vector<int> arr = convert_str_to_arr(tp); 
                Rect box(arr[0], arr[1], arr[2], arr[3]);
                temp.push_back(box);
            }
            bounding_boxes.push_back(temp);
        }
        gt.close(); //close the file object.
    }

	return bounding_boxes;	
}


vector<Mat> load_dataset(string dir){
    
	vector<Mat> dataset;
    vector<String> fn;
    glob(dir, fn, true);
    for (size_t i = 0; i < fn.size(); i++){
        Mat img = imread(fn[i], IMREAD_ANYCOLOR);
        if (img.empty()) continue;
        dataset.push_back(img);
    }


    return dataset;
}