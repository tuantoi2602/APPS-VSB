// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
// Image creation and its modification using CUDA.
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"

using namespace cv;

// Prototype of function in .cu file
void run_animation( uchar4 *bgr_pic, int sizex, int sizey, int elemx, int elemy );

// Image size
#define SIZEX 432 // Width of image
#define	SIZEY 321 // Heigth of image
// Block size for threads
#define BLOCKX 40 // block width
#define BLOCKY 25 // block height

int main()
{
	// Array is created to store all points from image with size SIZEX * SIZEY. 
	// Image is stored line by line. 
	uchar4 *bgr_pole = new uchar4[ SIZEX * SIZEY ];

	// Creation of empty image
	Mat img( SIZEY, SIZEX, CV_8UC3 );

	// Image filling by color gradient blue-green-red
	for ( int y = 0; y < SIZEY; y++ )
		for ( int x  = 0; x < SIZEX; x++ )
		{
			uchar4 bgr = { 0, 0, 0 }; // black
			if ( x < SIZEX / 2 )
			{
				bgr.y = 255 * x / ( SIZEX / 2 );
				bgr.x = 255 - bgr.y;
			}
			else
			{
				bgr.y = 255 * ( SIZEX - x ) / ( SIZEX / 2 );
				bgr.z = 255 - bgr.y;
			}
			// store points to array for transfer to GPU device
			bgr_pole[ y * SIZEX + x ] = bgr;

			// store points to image
			Vec3b v3bgr = { bgr.x, bgr.y, bgr.z };
			img.at<Vec3b>( y, x ) = v3bgr;
		}

	// Show image before modification
	imshow( "B-G-R Gradient", img );

	// Function calling from .cu file
	run_animation( bgr_pole, SIZEX, SIZEY, BLOCKX, BLOCKY );

	// Store modified data to image
	for ( int y = 0; y < SIZEY; y++ )
		for ( int x  = 0; x < SIZEX; x++ )
		{
			uchar4 bgr = bgr_pole[ y * SIZEX + x ];
			Vec3b v3bgr = { bgr.x, bgr.y, bgr.z };
			img.at<Vec3b>( y, x ) = v3bgr;
		}

	// Show modified image
	imshow( "B-G-R Gradient & Color Rotation", img );
	waitKey( 0 );
}

