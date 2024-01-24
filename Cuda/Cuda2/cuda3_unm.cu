// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Manipulation with prepared image.
//
// ***********************************************************************

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_img.h"
// Every threads identifies its position in grid and in block and modify image
__global__ void kernel_animation( CudaImg t_cuda_img )
{
	// X,Y coordinates 
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_x >= t_cuda_img.m_size.x ) return;
	if ( l_y >= t_cuda_img.m_size.y ) return;

	// Point [l_x,l_y] selection from image
	//uchar3 l_bgr, l_tmp = t_cuda_img.m_p_uchar3[ l_y * t_cuda_img.m_size.x + l_x ];
	uchar3 l_bgr, l_tmp = t_cuda_img.at3(l_y,l_x);

	// color rotation
    l_bgr.x = l_tmp.y;
    l_bgr.y = l_tmp.z;
    l_bgr.z = l_tmp.x;

	// Store point [l_x,l_y] back to image
	//t_cuda_img.m_p_uchar3[ l_y * t_cuda_img.m_size.x + l_x ] = l_bgr;
    t_cuda_img.at3(l_y,l_x) = l_bgr;
}
__global__ void kernel_mirror( CudaImg t_pic_in, CudaImg t_pic_out )
{
	// X,Y coordinates
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_x >= t_pic_in.m_size.x) return;
	if ( l_y >= t_pic_in.m_size.y) return;

	// Point [l_x,l_y] selection from image
	//uchar3 l_bgr, l_tmp = t_cuda_img.m_p_uchar3[ l_y * t_cuda_img.m_size.x + l_x ];
	uchar3 l_bgr,l_tmp = t_pic_in.at3(l_y,l_x);
    t_pic_out.at3(l_y,t_pic_in.m_size.x-l_x) = l_tmp;
}
__global__ void kernel_remove( CudaImg t_pic_in, CudaImg t_pic_out )
{
	// X,Y coordinates
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_x >= t_pic_in.m_size.x) return;
	if ( l_y >= t_pic_in.m_size.y) return;

	uchar3 l_bgr,l_tmp = t_pic_in.at3(l_y,l_x);


	if(!(l_tmp.y == l_tmp.z && l_tmp.x == 0)){
		l_tmp.x = 0;
		l_tmp.y = 0;
		l_tmp.z = 0;
	}


	 t_pic_out.at3(l_y,l_x) = l_tmp;
	// Store point [l_x,l_y] back to image
	//t_cuda_img.m_p_uchar3[ l_y * t_cuda_img.m_size.x + l_x ] = l_bgr;


}
__global__ void kernel_separation( CudaImg t_pic_in, CudaImg t_pic_out_r, CudaImg t_pic_out_g, CudaImg t_pic_out_b)
{
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_x >= t_pic_in.m_size.x ) return;
	if ( l_y >= t_pic_in.m_size.y ) return;

	uchar3 l_tmp = t_pic_in.at3(l_y,l_x);

	t_pic_out_r.at3(l_y,l_x) = {0, 0, l_tmp.x};
	t_pic_out_g.at3(l_y,l_x) = {0,l_tmp.y,0};
	t_pic_out_b.at3(l_y,l_x) = {l_tmp.z,0,0};
}
void cu_run_animation( CudaImg t_cuda_img, uint2 t_block_size )
{
	cudaError_t l_cerr;

	// Grid creation with computed organization
	dim3 l_grid( ( t_cuda_img.m_size.x + t_block_size.x - 1 ) / t_block_size.x,
			     ( t_cuda_img.m_size.y + t_block_size.y - 1 ) / t_block_size.y );
	kernel_animation<<< l_grid, dim3( t_block_size.x, t_block_size.y ) >>>( t_cuda_img );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();

}
void cu_run_rgb_separation( CudaImg &t_pic_in, CudaImg &t_pic_out_r, CudaImg &t_pic_out_g, CudaImg &t_pic_out_b )
{
	cudaError_t l_cerr;

	uint2 l_block_size = {32,32};
	dim3 l_grid( ( t_pic_in.m_size.x + l_block_size.x - 1 ) / l_block_size.x,
			       ( t_pic_in.m_size.y + l_block_size.y - 1 ) / l_block_size.y );

	kernel_separation<<< l_grid,dim3(l_block_size.x,l_block_size.y)>>>( t_pic_in, t_pic_out_r, t_pic_out_g, t_pic_out_b );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

		cudaDeviceSynchronize();
}
void cu_run_my_mirror( CudaImg &t_pic_in, CudaImg &t_pic_out )
{
	cudaError_t l_cerr;
	uint2 t_block_size = {32,32};
	// Grid creation with computed organization
	dim3 l_grid( ( t_pic_in.m_size.x + t_block_size.x - 1 ) / t_block_size.x,
			     ( t_pic_in.m_size.y + t_block_size.y - 1 ) / t_block_size.y );
	kernel_mirror<<< l_grid, dim3( t_block_size.x, t_block_size.y ) >>>( t_pic_in, t_pic_out);

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();

}
void cu_run_my_remove( CudaImg &t_pic_in, CudaImg &t_pic_out )
{
	cudaError_t l_cerr;
	uint2 t_block_size = {32,32};
	// Grid creation with computed organization
	dim3 l_grid( ( t_pic_in.m_size.x + t_block_size.x - 1 ) / t_block_size.x,
			     ( t_pic_in.m_size.y + t_block_size.y - 1 ) / t_block_size.y );
	kernel_remove<<< l_grid, dim3( t_block_size.x, t_block_size.y ) >>>( t_pic_in, t_pic_out);

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();

}
