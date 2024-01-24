// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
// Image transformation from RGB to BW schema. 
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_runtime.h>
#include "opencv2\opencv.hpp"

using namespace cv;

// Function prototype from .cu file
void run_grayscale( uchar4 *color_pic, uchar4 *bw_pic, int sizex, int sizey );

int main( int numarg, char **arg )
{
	if ( numarg < 2 ) 
	{
		printf( "Enter picture filename!\n" );
		return 1;
	}

	// Load image
	Mat bgr_img = imread( arg[ 1 ], CV_LOAD_IMAGE_COLOR );
	//IplImage *bgr_img = cvLoadImage( arg[ 1 ] );
	Size bgr_img_size = bgr_img.size();
	int sizex = bgr_img_size.width;
	int sizey = bgr_img_size.height;

	// Arrays alocation for images
	uchar4 *bgr_pole = new uchar4[ bgr_img_size.width * bgr_img_size.height ];
	uchar4 *bw_pole = new uchar4[ bgr_img_size.width * bgr_img_size.height ];
	for ( int y = 0; y < bgr_img_size.height; y++ )
		for ( int x  = 0; x < bgr_img_size.width; x++ )
		{
			Vec3b v3 = bgr_img.at<Vec3b>( y, x );
			uchar4 bgr = {  v3[ 0 ], v3[ 1 ], v3[ 2 ] };
			bgr_pole[ y * bgr_img_size.width + x ] = bgr;

		}

	// Calling function from .cu file
	run_grayscale( bgr_pole, bw_pole, bgr_img_size.width, bgr_img_size.height );

	Mat bw_img( bgr_img_size, CV_8UC3 );
	//IplImage *bw_img = cvCreateImage( cvSize( sizex, sizey ), IPL_DEPTH_8U, 3 );

	// Store data from GPU to new image
	for ( int y = 0; y < bgr_img_size.height; y++ )
		for ( int x  = 0; x < bgr_img_size.width; x++ )
		{
			uchar4 bgr = bw_pole[ y * bgr_img_size.width + x ];
			Vec3b v3 = { bgr.x, bgr.y, bgr.z };
			bw_img.at<Vec3b>( y, x ) = v3;
		}

	// Show the Color and BW image
	imshow( "Color", bgr_img );
	imshow( "GrayScale", bw_img );
	waitKey( 0 );
}

