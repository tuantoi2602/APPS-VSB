// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
// Manipulation with prepared image.
//
// ***********************************************************************

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Every threads identifies its position in grid and in block and modify image
__global__ void kernel_animation( uchar4 *pic, int sizex, int sizey )
{
	// X,Y coordinates 
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( x >= sizex ) return;
	if ( y >= sizey ) return;

	// Point [x,y] selection from image
	uchar4 bgr = pic[ y * sizex + x ];

	// Color rotation inside block
	int x2 = blockDim.x / 2;
	int y2 = blockDim.y / 2;
	int px = __sad( x2, threadIdx.x, 0 ); // abs function
	int py = __sad( y2, threadIdx.y, 0 );

	if ( px < x2 * ( y2 - py ) / y2 ) 
	{
		uchar4 tmp = bgr;
		bgr.x = tmp.y;
		bgr.y = tmp.z;
		bgr.z = tmp.x;
	}

	// Store point [x,y] back to image
	pic[ y * sizex + x ] = bgr;

}

void run_animation( uchar4 *pic, int sizex, int sizey, int blockx, int blocky )
{
	cudaError_t cerr;

	// Memory allocation in GPU device
	uchar4 *cudaPic;
	cerr = cudaMalloc( &cudaPic, sizex * sizey * sizeof( uchar4 ) );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );	

	// Copy data to GPU device
	cerr = cudaMemcpy( cudaPic, pic, sizex * sizey * sizeof( uchar4 ), cudaMemcpyHostToDevice );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );	

	// Grid creation with computed organization
	dim3 mrizka( ( sizex + blockx - 1 ) / blockx, ( sizey + blocky - 1 ) / blocky );
	kernel_animation<<< mrizka, dim3( blockx, blocky ) >>>( cudaPic, sizex, sizey );

	if ( ( cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Copy data from GPU device to PC
	cerr = cudaMemcpy( pic, cudaPic, sizex * sizey * sizeof( uchar4 ), cudaMemcpyDeviceToHost );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );	

	// Free memory
	cudaFree( cudaPic );

	// For printf
	//cudaDeviceSynchronize();

}