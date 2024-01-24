// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
// Image transformation from RGB to BW schema. 
//
// ***********************************************************************

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Demo kernel to tranfrom RGB color schema to BW schema
__global__ void kernel_grayscale( uchar4 *color_pic, uchar4* bw_pic, int sizex, int sizey )
{
	// X,Y coordinates and check image dimensions
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if ( y >= sizey ) return;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( x >= sizex ) return;

	// Get point from color picture
	uchar4 bgr = color_pic[ y * sizex + x ];

	// All R-G-B channels will have the same value
	bgr.x = bgr.y = bgr.z = bgr.x * 0.11 + bgr.y * 0.59 + bgr.z * 0.30;

	// Store BW point to new image
	bw_pic[ y * sizex + x ] = bgr;

}

void run_grayscale( uchar4 *color_pic, uchar4* bw_pic, int sizex, int sizey )
{
	cudaError_t cerr;
	// Memory allocation in GPU device
	uchar4 *cudaColorPic;
	uchar4 *cudaBWPic;
	cerr = cudaMalloc( &cudaColorPic, sizex * sizey * sizeof( uchar4 ) );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );	

	cerr = cudaMalloc( &cudaBWPic, sizex * sizey * sizeof( uchar4 ) );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );	

	// Copy color image to GPU device
	cerr = cudaMemcpy( cudaColorPic, color_pic, sizex * sizey * sizeof( uchar4 ), cudaMemcpyHostToDevice );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );	

	int block = 16;
	dim3 blocks( ( sizex + block - 1 ) / block, ( sizey + block - 1 ) / block );
	dim3 threads( block, block );

	// Grid creation, size of grid must be greater than image
	kernel_grayscale<<< blocks, threads >>>( cudaColorPic, cudaBWPic, sizex, sizey );

	if ( ( cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Copy new image from GPU device
	cerr = cudaMemcpy( bw_pic, cudaBWPic, sizex * sizey * sizeof( uchar4 ), cudaMemcpyDeviceToHost );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );	

	// Free memory
	cudaFree( cudaColorPic );
	cudaFree( cudaBWPic );
}