#include <opencv2/opencv.hpp>
#include "quickopencv.h"


int main(int argc, char** argv) 
{
	cv::Mat src = cv::imread("C:/Users/70756/Desktop/1.png");
	cv:imshow("src", src);
	QuickDemo qd;
	qd.bifilter_Demo(src);
	cv::waitKey(0);
	return 0;
}