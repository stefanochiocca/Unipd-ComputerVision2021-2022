#include "HandDetection.h"

static void processImage(int, void *params){
    //just taking the paramentes and doing the actual processing
    auto *filter_params = static_cast<DetectionParams*>(params);
    doProcessImage(filter_params);
}

int main( int argc, const char** argv ) {
    //here are a huge list of many tried classifiers

    //String cascade_name = "rgb/cacade_reduced_LBP_30px_30times/cascade.xml";
    //String cascade_name = "rgb/cacade_reduced_LBP_10px_25times/cascade.xml";
    //String cascade_name = "rgb/cacade_reduced_Haar_10px_25times/cascade.xml";
    //String cascade_name = "rgb/cacade_reduced_LBP_40px_30times/cascade.xml";
    //String cascade_name = "rgb/cacade_reduced_LBP_30px_20times_black/cascade.xml";
    //String cascade_name = "rgb/cacade_reduced_LBP_24px_20times_white_negatives/cascade.xml";
    //String cascade_name = "rgb/cascade1/cascade.xml";
    //String cascade_name = "rgb/cascade2/cascade.xml";
    //String cascade_name = "rgb/cascade3/cascade.xml";
    //String cascade_name = "rgb/cascade40/cascade.xml";
    //String cascade_name = "rgb/cascade24/cascade.xml";
    //String cascade_name = "rgb/cascade2_24/cascade.xml";
    //String cascade_name = "rgb/cascade_00_24/cascade.xml";
    //String cascade_name = "rgb/cascade_haar_00_20/cascade.xml";
    //String cascade_name = "rgb/cascade_haar_00_20_17stages/cascade.xml";
    //String cascade_name = "rgb/cascade50/cascade.xml";
    //String cascade_name = "rgb/cascade_black_white/cascade.xml";
    //String cascade_name = "rgb/cascade_black1/cascade.xml";



    //String cascade_name = "Detection_Segmentation/traincascade_classifiers/expanded/24/your/normal/cascade/cascade.xml";
    //String cascade_name = "Detection_Segmentation/traincascade_classifiers/expanded/24/your/black_bg/cascade/cascade.xml";

    //String cascade_name = "Detection_Segmentation/traincascade_classifiers/expanded/filtered/32/your/cascade/cascade.xml";



    //these are the ones working best
    //String yourright = "Detection_Segmentation/traincascade_classifiers/expanded/24/yourright/cascade/cascade.xml";
    //String yourright32 = "Detection_Segmentation/traincascade_classifiers/expanded/32/yourright/cascade/cascade.xml";
    String yourright40 = "Detection_Segmentation/traincascade_classifiers/expanded/40/yourright/cascade/cascade.xml";
    //String yourleft = "Detection_Segmentation/traincascade_classifiers/expanded/24/yourleft/cascade/cascade.xml";
    //String yourright46 = "Detection_Segmentation/traincascade_classifiers/expanded/filtered/46/yourright/cascade/cascade.xml";
    String yourleft46 = "Detection_Segmentation/traincascade_classifiers/expanded/filtered/46/yourleft/cascade/cascade.xml";
    //String myleft46 = "Detection_Segmentation/traincascade_classifiers/expanded/filtered/46/myleft/cascade/cascade.xml";
    //String myright46 = "Detection_Segmentation/traincascade_classifiers/expanded/filtered/46/myright/cascade/cascade.xml";
    String myright = "Detection_Segmentation/traincascade_classifiers/expanded/24/myright/cascade/cascade.xml";
    String myleft = "Detection_Segmentation/traincascade_classifiers/expanded/24/myleft/cascade/cascade.xml";

    //String cascade_name = "Detection_Segmentation/traincascade_classifiers/expanded/24/hands_over_face/cascade/cascade.xml";
    //String cascade_name = "Detection_Segmentation/traincascade_classifiers/expanded/12/hands_over_face/cascade/cascade.xml";
    //String cascade_name = "Detection_Segmentation/traincascade_classifiers/expanded/6/hands_over_face/cascade/cascade.xml";
    //String cascade_name = "Detection_Segmentation/traincascade_classifiers/expanded/32/hands_over_face/cascade/cascade.xml";
    //String cascade_name = "Detection_Segmentation/traincascade_classifiers/expanded/40/hands_over_face/cascade/cascade.xml";
    String hof50 = "Detection_Segmentation/traincascade_classifiers/expanded/50/hands_over_face/cascade/cascade.xml";

    //classifiers = {yourleft,yourright,myleft,myright};
    //classifiers = {yourleft,yourright,myleft,myright};
    //classifiers = {yourright,yourright32,yourright40};
    //classifiers = {yourright32,yourright40};
    //classifiers = {yourleft};

    //the program will run with these classifiers (so in the end will be 5 of them, for now, just to test, only 2)
    classifiers = {yourleft46, yourright40,myleft, myright, hof50};
    //classifiers = {yourleft46, yourright46,myleft46, myright46, hof50};
    //classifiers = {myleft, myright};
    //classifiers = {yourright40};
    //classifiers = {myright};
    //classifiers = {hof50};

    //show a window just to put some trackbars usefull for developement
    namedWindow("image");
    DetectionParams params;
    //loop thru them
    //while(true) {
        for (int i = 0; i < file_numbers.size(); i++){
            const string &file_number = file_numbers[i];
            cout << file_number + image_extension << endl;

            //we're using other parameters for the hands over face images
            if (i > 19) {
                params.min_size = 50;
                params.min_detections = 10;
            }

            //this has to always be zero
            createTrackbar("Detection min neighbors", "image", &params.min_neighbors, 1, processImage, (void*)&params);
            //found that a detection min size of 100 to 150 works best
            createTrackbar("Detection min size", "image", &params.min_size, 300, processImage, (void*)&params);
            //this can be 30-100 without problems
            createTrackbar("Mean shift kernel", "image", &params.kernel_bandwidth, 200, processImage, (void*)&params);
            //the percentage of points inside the best cluster over all the points (in order to be accepted, has to be above 30%)
            createTrackbar("Percentage of total", "image", &params.min_percentage, 10, processImage, (void*)&params);
            //total number of detected features in the image (must be at least 30, or 10 fro hof images)
            createTrackbar("Min mumber of detections", "image", &params.min_detections, 100, processImage, (void*)&params);

            image = imread(image_path + file_number + image_extension);
            params.file_number = file_number;
            //and launch the processing
            processImage(0, &params);
            waitKey(0);
        }
    //}
}

//this will use the mean shift algorithm
void doProcessImage( DetectionParams *params ){
    //here we save all the rects that rappresents the rec sorrounding a detected hand
    vector<Rect> rects;
    image_clone = image.clone();
    int image_height = image.rows;
    int image_width = image.cols;
    //we normally will run 5 classifiers (one for each hand and one for hands over face)
    for (int k = 0; k < classifiers.size(); k++) {
        String cascade_name = classifiers[k];
        if (k < 4 && image_width < 1000) { continue;} //first 4 classifiers don't have to work on small images
        if (k == 4 && image_width > 1000) { continue;} //hand over face classifier doesn't have to work on large images
        //load the classifier
        cascadeClassifier.load(cascade_name);
        //found hands by the classifier
        vector<Rect> hands;
        Size min_size{params->min_size, params->min_size};
        //so here there was a super sneaky bug. Using the same image_clone for detection and for putting the detection circles and rects
        //made so than the subsequent classifiers were detection their feautures in a tainted image, and so detecting less
        //so i always need the main image to pass to the classifier, not the clone
        cascadeClassifier.detectMultiScale(image, hands, 1.05, params->min_neighbors, 0, min_size);

        //we'll save here all the points found by the classifier (the centers of the hands)
        vector<vector<double>> mean_shift_points;
        //these are all the detected "hands" of the classifier
        int detections = hands.size();
        //we go thru them, to calculate the "true" hand = the mean shift of all these points
        for (size_t i = 0; i < detections; i++) {
            Rect hand = hands[i];
            vector<double> point;
            //the central point of the "hand", because the classifier returns rects not points
            point.push_back(hand.x + hand.width / 2);
            point.push_back(hand.y + hand.height / 2);
            mean_shift_points.push_back(point);
            //rectangle(image_clone, hands[i],Scalar(0,0,0),1);
            Point2i center = Point2i(hand.x + hand.width / 2, hand.y + hand.height / 2);
            circle(image_clone, center, 1, black, 1, LINE_AA, 0);
        }

        //run the mean shift algorithm to find the "centers"
        auto *msp = new MeanShift();
        double kernel_bandwidth = params->kernel_bandwidth;
        vector<Cluster> clusters = msp->cluster(mean_shift_points, kernel_bandwidth);


        int max_1 = 0;
        int max_1_index = -1;
        //run thru all the clusters to find the max (the "center" of the cluster)
        for (int i = 0; i < clusters.size(); i++) {
            Cluster cluster = clusters[i];
            int shifted_points = (int)cluster.original_points.size();
            if (shifted_points > max_1) {
                max_1 = shifted_points;
                max_1_index = i;
            }
        }

        //run again thru the clusters to put the colored circle in the "center"
        //and find a suitable bounding box of the hand
        for (int i = 0; i < clusters.size(); i++) {
            Cluster cluster = clusters[i];
            vector<double> mode = cluster.mode;
            //center of the cluster (there are many clusters. We here have the "biggest" one
            Point2i center = Point2i((int)mode[0], (int)mode[1]);
            //so if this is the biggest cluster = most probably the hand
            //here i am at the "center" of the cluster
            //i will accept this center as the center of the cluster if :
            //it's the biggest cluster, so max_1 (there were max_2 and max_3, but then removed because the classifier got better)
            //at least 30% of the points found by the classifier belong to this cluster (min_percentage = 30)
            //there have to be at lest some min_detections number of points in the cluster (the classifier will always find something,
            //but I will accept only if the total nr is above 30
            if (i == max_1_index && ((double) max_1 / detections > (double)params->min_percentage/10) && max_1 > params->min_detections) {
                //calculate the avg distance from the center
                //this is used to know how "big" the cluster is.
                //Probably bigger the cluster than bigger tha hand.
                double full_distance_from_center = 0;
                for (int j = 0; j < cluster.original_points.size(); j++) {
                    vector<double> original_point = cluster.original_points[j];
                    Point2i centerj = Point2i((int)original_point[0], (int)original_point[1]);
                    //so at each point of this cluster we put a black point in the image (actually a black little circle)
                    //circle(image_clone, centerj, 1, black, 1, LINE_AA, 0);
                    //and sum up the distances from the center
                    full_distance_from_center += dist(center.x, center.y, centerj.x, centerj.y);
                }
                //the avg distance
                full_distance_from_center = full_distance_from_center / (double)cluster.original_points.size();
                //now i consider only the points within like 2 times the avg distance from center to perform a new calculation
                //this way I hope to remove some outliars and get the real "dimension" of the cluster
                double distance_from_center = 0;
                int number_of_points = 0;
                for (int j = 0; j < cluster.original_points.size(); j++) {
                    vector<double> original_point = cluster.original_points[j];
                    Point2i centerj = Point2i((int)original_point[0], (int)original_point[1]);
                    //circle(image_clone, centerj, 3, black, 1, LINE_AA, 0);
                    double new_distance_from_center = dist(center.x, center.y, centerj.x, centerj.y);
                    //so now i consider only the points less than 2 times the avg distance from the center
                    if (new_distance_from_center < full_distance_from_center*2) {
                        distance_from_center += new_distance_from_center;
                        number_of_points++;
                    }
                }
                //so this number is the distance from the center of the most important points in the cluster
                //i will consider this as the "dimension" of the cluster
                distance_from_center = distance_from_center / number_of_points;

                //now i try to guess a suitable bounding box from the "dimension" of the cluster

                //some max dimensions for the bounding box (not to be too large)
                int max_width = 200;
                int max_height = 200;
                //so, by observation, i found that the rect 12 times bigger than the avg distance from the center is a suitable bounding box
                //one can play with those numbers (the 2 and the 12 )
                //12 for the other hands, 20 for my hands
                int factor = 12;
                if (k == 2 || k == 3) {
                    factor = 20;
                    max_width = 300;
                    max_height = 300;
                }
                if (k == 4) {
                    factor = 10;
                    max_width = 100;
                    max_height = 100;
                }
                int rect_width = (int)distance_from_center*factor;
                int rect_height = (int)distance_from_center*factor;
                if (rect_width > max_width) rect_width = max_width;
                if (rect_height > max_height) rect_height = max_height;
                int rect_origin_x = center.x - rect_width/2;
                int rect_origin_y = center.y - rect_height/2;
                //so i draw a rect (bounding box) of the hand in the image
                //checking if the rect overflows the image
                if (rect_origin_y + rect_height > image_height) {
                    rect_height = image_height - rect_origin_y;
                }
                if (rect_origin_x + rect_width > image_width) {
                    rect_width = image_width - rect_origin_x;
                }
                Rect rect = Rect(rect_origin_x,rect_origin_y,rect_width,rect_height);
                rectangle(image_clone,rect,colors[k],3);
                //and put this rect in this vector in order to segment the hand contained within
                rects.push_back(rect);
                circle(image_clone, center, 10, colors[k], 5, FILLED, 0);
            } else {
                //else draw a white circle (just to see the other clusters
                circle(image_clone, center, 5, white, 3, FILLED, 0);
            }
        }
    }
    //show the detections
    string window_name = "Detections";
    namedWindow(window_name,WINDOW_NORMAL);
    imshow( window_name, image_clone);
    saveOurDetectionFile(image_clone, params->file_number);

    //save our det files for Intersection Over Union
    saveOurDetFile(rects,params->file_number);

    //now segment the hand
    segmendAndColorHands(rects, image, params->file_number);
}

//save our det file to disk
void saveOurDetFile(vector<Rect> &rects, string &file_number) {
    string content;
    for (const Rect& rec : rects) {
        content += to_string(rec.x) + " " + to_string(rec.y) + " " + to_string(rec.width) + " " + to_string(rec.height) + "\n";
    }
    string path = our_det_path + file_number + our_det_extension;
    ofstream our_det(path);
    our_det << content;
    our_det.close();
}



//i'm passing the original image here and all the rects(hands=blunding boxes) found
void segmendAndColorHands(vector<Rect> &rects, Mat &img, string &file_number) {
    Mat img_clone = image.clone();
    //this will be the final black and white segmented image
    Mat bw = Mat::zeros(image.size(), 0);
    for (int r = 0; r < rects.size(); r++) {
        Rect rect = rects[r];
        //for every bounding box I take the sub image
        Mat bilateral_source = image(rect);
        //show the rect
        String rect_image_name = "Rect : " + to_string(r);
        namedWindow(rect_image_name, WINDOW_NORMAL);
        imshow(rect_image_name, bilateral_source);


        //bilateral filter test
        Mat source;
        int d = 37;
        int sigcol = 17;
        int sigspa = 147;
        bilateralFilter(bilateral_source,source,d,sigcol,sigspa,BORDER_DEFAULT);
        //end bilateral filter test


        //this is the segmented image inside the bounding box
        Mat dest = Mat::zeros(rect.size(), source.type());

        //here starts the algorithm to create a vignette out of the hand image
        double radius = 0.6;
        double cx = (double) source.cols / 2, cy = (double) source.rows / 2;
        double maxDis = radius * dist(0, 0, cx, cy);
        double temp;
        for (int y = 0; y < source.rows; y++) {
            for (int x = 0; x < source.cols; x++) {
                temp = cos(dist(cx, cy, x, y) / maxDis);
                temp *= temp;
                dest.at<Vec3b>(y, x)[0] = saturate_cast<uchar>((source.at<Vec3b>(y, x)[0]) * temp);
                dest.at<Vec3b>(y, x)[1] = saturate_cast<uchar>((source.at<Vec3b>(y, x)[1]) * temp);
                dest.at<Vec3b>(y, x)[2] = saturate_cast<uchar>((source.at<Vec3b>(y, x)[2]) * temp);
            }
        }//end of vignette algorithm

        //show the vignette
        String vignette_image_name = "Vignette : " + to_string(r);
        namedWindow(vignette_image_name, WINDOW_NORMAL);
        imshow(vignette_image_name, dest);

        //now the segmentation (thresholding)
        Mat src_gray, dst;
        cvtColor(dest, src_gray, COLOR_BGR2GRAY);
        //binary and otsu thresholding
        threshold(src_gray, dst, 0, 255, 8);
        //show the image of the thresholding
        String threshold_image_name = "Threshold : " + to_string(r);
        namedWindow( threshold_image_name,WINDOW_NORMAL);
        imshow(threshold_image_name, dst);

        //now i loop on the threshold sub-image (the hands) and put them in the bigger black and white picture
        //actually putting only the white pixels
        for (int y = 0; y < rect.width; y++) {
            for (int x = 0; x < rect.height; x++) {
                unsigned char val = dst.at<unsigned char>(y, x);
                if (val == (unsigned char )255) { //if white
                    bw.at<unsigned char>(y + rect.y, x + rect.x) = val;
                    //color the image
                    for (int c = 0; c < image.channels(); c++) {
                        img_clone.at<Vec3b>(y + rect.y, x + rect.x)[c] = colors[r][c];
                    }
                }
            }
        }
    }
    //show the segmented image
    //TODO : remove or filter out the errors of the classifier (sometimes a shoulder will be detected and segmented as a hand)
    namedWindow( "BW", WINDOW_NORMAL );
    imshow("BW", bw);

    //save our mask files for Pixel Accuracy
    saveOurMaskFile(bw,file_number);

    //show the colored image
    namedWindow( "Colours", WINDOW_NORMAL );
    imshow("Colours", img_clone);
    saveOurColoredFile(img_clone,file_number);
}

//saves our mask image to disk
void saveOurMaskFile(Mat &mat, string &file_number) {
    string path = our_mask_path + file_number + our_mask_extension;
    imwrite(path, mat);
}

//saves colored image to disk
void saveOurColoredFile(Mat &mat, string &file_number) {
    string path = our_colored_path + file_number + our_colored_extension;
    imwrite(path, mat);
}

//saves detected image to disk
void saveOurDetectionFile(Mat &mat, string &file_number) {
    string path = our_detection_path + file_number + our_detection_extension;
    imwrite(path, mat);
}