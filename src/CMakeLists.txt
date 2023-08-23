cmake_minimum_required(VERSION 3.16)
project(HandDetection)

set(CMAKE_CXX_STANDARD 14)


find_package(OpenCV REQUIRED PATHS)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(HandDetection Detection_Segmentation/HandDetection.cpp Detection_Segmentation/meanshift/MeanShift.cpp)

#add_executable(HandDetection Evaluation/main.cpp Evaluation/include/evaluate.h Evaluation/include/evaluate.cpp)


target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})





#add_executable(HandDetection thresholding.cpp)
#add_executable(HandDetection playground.cpp)
#add_executable(HandDetection main.cpp)
#add_executable(HandDetection hough.cpp)
#add_executable(HandDetection kmeans.cpp)
#add_executable(HandDetection ok-means.cpp)
#add_executable(HandDetection kmeans_algo.cpp)
#add_executable(HandDetection region_growing_auto.cpp)
#add_executable(HandDetection region_growing.cpp)
#add_executable(HandDetection evaluate.cpp)
#add_executable(HandDetection matching.cpp)
#add_executable(HandDetection rg_rgb.cpp)
#add_executable(HandDetection rg_click.cpp)
#add_executable(HandDetection dt.cpp)
#add_executable(HandDetection noise.cpp)
#add_executable(HandDetection canny2.cpp)
#add_executable(HandDetection grabcut.cpp)
#add_executable(HandDetection tree_project.cpp tree_project.h)
#add_executable(HandDetection meanshift/MeanShift_cpp/MeanShift_cpp-master/MeanShift.cpp meanshift/MeanShift_cpp/MeanShift_cpp-master/cpp_test.cpp)
#add_executable(HandDetection dehaze/main.cpp dehaze/hazeremoval.cpp dehaze/guidedfilter.cpp)
#add_executable(HandDetection skin_detect/skin-detect.cpp skin_detect/skin-detect.h)

