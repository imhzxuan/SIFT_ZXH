#include<iostream> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/opencv.hpp> 
using namespace cv;
int main()
{
	VideoCapture capture(0);
	while (1)
	{
		Mat frame;
		Mat edge,grayimage;
		capture >> frame;
		if (frame.empty()) break;
		cvtColor(frame, grayimage, CV_BGR2GRAY);
		blur(grayimage,edge, Size(6, 6));
		Canny(edge,edge,3,9,3);
		imshow(" ”∆µ÷°", edge);
		waitKey(30);

	}
	return 0;
}