// #include <iostream>
// #include <stdio.h>
#ifndef OPENPOSE_H
#define OPENPOSE_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "darknet.h"

using namespace std;
using namespace cv;

template<typename T>
inline int intRound(const T a);

template<typename T>
inline T fastMin(const T a, const T b);

// float *run_net(network * net,float *indata);
void connect_bodyparts(float * pose_keypoints,const float* const map,const float* const peaks,int mapw,int maph,const int inter_min_above_th,const float inter_th,const int min_subset_cnt,const float min_subset_score,int * keypoint_shape);
void find_heatmap_peaks(const float *src,float *dst,const int SRCW,const int SRCH,const int SRC_CH,const float TH);
Mat create_netsize_im(const Mat &im,const int netw,const int neth,float *scale);

// extern "C"{
//     void free_memory(sk pose_struct);
// 	sk openpose_forward(network * const net, unsigned char* img_src, long* img_shape, long* img_strides);
// }
#endif