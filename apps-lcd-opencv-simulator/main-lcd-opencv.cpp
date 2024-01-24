#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


void pix(Mat po,int y,int x,int a,int b, int c){
	po.at<uchar>(y,(x*3)) = a;
	po.at<uchar>(y,(x*3)+1) = b;
	po.at<uchar>(y,(x*3)+2) = c;
}
void po(Mat pe,int y0,int x0, int y1, int x1,int a,int b, int c){

	        int dx =  abs(x1 - x0), sx = x0 < x1 ? 1 : -1;

	        int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;

	        int err = dx + dy, e2; //error value e_xy

	        for(;;){  //loop

	        	pix(pe, y0,x0,a,b,c);
	        	if (x0 == x1 && y0 == y1) break;

	        	            e2 = 2*err;

	        	            if (e2 >= dy) { err += dy; x0 += sx; } //e_xy+e_x > 0

	        	            if (e2 <= dx) { err += dx; y0 += sy; } //e_xy+e_y < 0
	        }
}

int main(int argc, const char * argv[]) {

	cv::Mat imageRed(200,200, CV_8UC3, Scalar(0,0,0));

	Mat c2 = imread("index.jpeg");
	pix(imageRed,0,2,255,255,255);
//	cout << imageRed << endl;

	po(imageRed,40,60,120,100,255,255,255);

	//c2.at<uchar>(100,200) = (255,255,255);
	po(c2,40,60,120,100,255,255,255);
    imshow("Image RED", imageRed);
    imshow("hasagi",c2);

    waitKey();

    imwrite("filename.png", imageRed);
return 0;
}
