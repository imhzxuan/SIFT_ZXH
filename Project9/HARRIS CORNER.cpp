#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include<math.h>
using namespace cv;
using namespace std;

void Harris1(Mat &srcImage,Mat &dstImage)
{
	int col, row;
	double trace, det;
	double temp1, temp2,temp3;
	int i, j;
	Mat dstImagex, dstImagey,dstImagexy;
	dstImagex.create(srcImage.rows, srcImage.cols, srcImage.type());
	dstImagey.create(srcImage.rows, srcImage.cols, srcImage.type());
	dstImagexy.create(srcImage.rows, srcImage.cols, srcImage.type());
	col = srcImage.cols;
	row = srcImage.rows;
	for (i = 0; i < row; i++)
	{
		uchar *ptr1 = srcImage.ptr(i);
		uchar *ptr2 = dstImagex.ptr(i);
		for (j = 2; j < col - 2; j++)
		{
			ptr2[j] = abs(ptr1[j + 1] - ptr1[j-1]+2*ptr1[j+2]-2*ptr1[j-2])/3;
		}
		ptr2[0] = 0;
		ptr2[1] =0;
		ptr2[j] =0;
		ptr2[j + 1] = 0;
	}

	//get Y

	for (i = 2; i < row-2; i++)
	{
		uchar *ptr0 = srcImage.ptr(i - 2);
		uchar *ptr3 = srcImage.ptr(i + 2);
		uchar *ptr1 = srcImage.ptr(i-1);
		uchar *ptr12 = srcImage.ptr(i + 1);
		uchar *ptr2 = dstImagey.ptr(i);
		for (j = 0; j < col ; j++)
		{
			ptr2[j] = abs(ptr12[j] - ptr1[j]+ptr3[j]*2-ptr0[j]*2)/3;
		}
	}
	uchar *ptr1 = srcImage.ptr(i);
	uchar *ptr2 = dstImagey.ptr(i);
	for (j = 0; j < col; j++)
	{
		ptr2[j] = 0;
	}
	ptr1 = srcImage.ptr(i+1);
	ptr2 = dstImagey.ptr(i+1);
	for (j = 0; j < col; j++)
	{
		ptr2[j] =0;
	}
	ptr1 = srcImage.ptr(0);
	ptr2 = dstImagey.ptr(0);
	for (j = 0; j < col; j++)
	{
		ptr2[j] = 0;
	}
	ptr1 = srcImage.ptr(1);
	ptr2 = dstImagey.ptr(1);
	for (j = 0; j < col; j++)
	{
		ptr2[j] = 0;
	}

	for (i = 0; i < row; i++)
	{
		uchar *ptr = dstImagex.ptr(i);
		uchar *ptr2 = dstImagey.ptr(i);
		uchar *ptr3 = dstImagexy.ptr(i);
		for (j = 0; j < col; j++) ptr3[j] = ptr[j] * ptr2[j];
	}
	blur(dstImagexy, dstImagexy, Size(5, 5));
	for (i = 0; i < row; i++)
	{
		uchar *ptr = dstImagex.ptr(i);
		for (j = 0; j < col; j++) ptr[j] = ptr[j] * ptr[j];
	}
	blur(dstImagex, dstImagex, Size(5, 5));

	for (i = 0; i < row; i++)
	{
		uchar *ptr = dstImagey.ptr(i);
		for (j = 0; j < col; j++) ptr[j] = ptr[j] * ptr[j];
	}
	blur(dstImagey, dstImagey, Size(5, 5));



	//H MATRIX
	for (i = 0; i < row ; i++)
	{
		uchar *ptr = dstImagex.ptr(i);
		uchar *ptr2 = dstImagey.ptr(i);
		uchar *ptr3 = dstImagexy.ptr(i);
		uchar *ptr4 = dstImage.ptr(i);
		for (j = 0; j < col ; j++)
		{
			temp1 = ptr[j];
			temp2 = ptr2[j];
			temp3 = ptr3[j];
			trace = temp1 + temp2;
			det = abs(temp1*temp2 - temp3*temp3);
			ptr4[j] =(int)abs(det/trace);
		}
	}

}

void drawcircle(Mat &srcImage, Mat &dstImage)
{
	int col, row,i,j;
	col = srcImage.cols;
	row = srcImage.rows;
	for (i = 1; i < row-1; i++)
	{
		uchar *ptr1 = srcImage.ptr(i);
		uchar *ptr2 = dstImage.ptr(i);
		uchar *ptr3 = dstImage.ptr(i-1);
		uchar *ptr4 = dstImage.ptr(i + 1);
		for (j =5; j < col - 5; j++)
		{
			if (
				ptr2[j]>50 && (ptr2[j]>ptr2[j - 1]) && (ptr2[j]>ptr2[j + 1]) 
				&& (ptr2[j]>ptr3[j - 1]) && (ptr2[j]>ptr3[j + 1]) && (ptr2[j]>ptr3[j])
				&& (ptr2[j]>ptr3[j - 1]) && (ptr2[j]>ptr3[j + 1]) && (ptr2[j]>ptr3[j])
				) circle(srcImage, Point(j, i), 3, Scalar(0), 2, 8);
		}
	}
}
void main()
{
	Mat srcImage1, dstImage1;
	srcImage1 = imread("C:\\Users\\×ÓìÓ\\Desktop\\8.jpg",0);

	dstImage1.create(srcImage1.rows, srcImage1.cols, CV_8UC1);
	Harris1(srcImage1, dstImage1);
	drawcircle(srcImage1, dstImage1);
//	imshow("src", dstImage1);
	imshow("src", srcImage1);
	waitKey(0);
	
//	imshow("dst", dstImage1);
//	
}
/*	
	system("Pause");
}*/

