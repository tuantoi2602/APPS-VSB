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
__global__ void kernel_run_img_rotate90( CudaImg t_pic_in, CudaImg t_pic_out )
{
	// X,Y coordinates
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_x >= t_pic_in.m_size.x) return;
	if ( l_y >= t_pic_in.m_size.y) return;




	// Point [l_x,l_y] selection from image
	//uchar3 l_bgr, l_tmp = t_cuda_img.m_p_uchar3[ l_y * t_cuda_img.m_size.x + l_x ];
	uchar3 l_bgr,l_tmp = t_pic_in.at3(l_y,l_x);
    t_pic_out.at3(l_x,t_pic_in.m_size.y -l_y) = l_tmp;
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
		l_tmp = {0,0,0};
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

__global__ void kernel_img_ins_sel( CudaImg t_big_img, CudaImg t_small_img, int2 t_position, int t_ins_or_select )
{
	// X,Y coordinates
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_x >= t_big_img.m_size.x ) return;
	if ( l_y >= t_big_img.m_size.y ) return;

	int2 l_xy_small;
	l_xy_small.x = l_x - t_position.x;
	l_xy_small.y = l_y - t_position.y;

	if(l_xy_small.x < 0) return;
	if(l_xy_small.y < 0) return;
	if(l_xy_small.x >= t_small_img.m_size.x) return;
	if(l_xy_small.y >= t_small_img.m_size.y) return;

	//insertion
//	t_big_img.at3(l_y,l_x)=t_small_img.at3(l_xy_small.y,l_xy_small.x);
	//selection
	//t_small_img.at3(l_xy_small.y,l_xy_small.x) = t_big_img.at3(l_y,l_x);
	if( t_ins_or_select == 0){
		t_big_img.at3(l_y,l_x)=t_small_img.at3(l_xy_small.y,l_xy_small.x); //insert
	}else{
		t_small_img.at3(l_xy_small.y,l_xy_small.x) = t_big_img.at3(l_y,l_x); //select
	}

/*	if(t_ins_or_sel == 0){
		t_big_img.at3(l_y,l_x)=t_small_img.at3(l_xy_small.y,l_xy_small.x); //insert
	}
	else{
		t_small_img.at3(l_xy_small.y,l_xy_small.x) = t_big_img.at3(l_y,l_x); //select
	}*/
}

__global__ void kernel_draw_char( CudaImg t_img, int2 t_pos, uchar3 t_color, char t_char, CudaImg t_font, uint2 t_font_size ){
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_img.m_size.y ) return;
	if ( l_x >= t_img.m_size.x ) return;

	int line = ( (int *) t_font.m_p_void)[t_char * t_font_size.y + l_y];

	if(line & (1<< l_x))
		t_img.at3(l_y + t_pos.y, l_x + t_pos.x) = t_color;
}
void cu_draw_char( CudaImg &t_img, int2 t_pos, uchar3 t_color, char t_char, CudaImg &t_font, uint2 t_font_size ){
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images

	dim3 l_blocks_size = { (t_font_size.x + 1) /2, (t_font_size.y + 1)/ 2, 1 };
	dim3 l_grid = {2,2,1};
	kernel_draw_char<<< l_grid, l_blocks_size >>>(t_img, t_pos, t_color, t_char, t_font, t_font_size );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}
__global__ void kernel_insertimage( CudaImg t_big_pic, CudaImg t_small_pic, int2 t_position )
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_small_pic.m_size.y ) return;
	if ( l_x >= t_small_pic.m_size.x ) return;
	int l_by = l_y + t_position.y;
	int l_bx = l_x + t_position.x;
	if ( l_by >= t_big_pic.m_size.y || l_by < 0 ) return;
	if ( l_bx >= t_big_pic.m_size.x || l_bx < 0 ) return;

	// Get point from small image
	uchar4 l_fg_bgra = t_small_pic.m_p_uchar4[ l_y * t_small_pic.m_size.x + l_x ];
	uchar3 l_bg_bgr = t_big_pic.m_p_uchar3[ l_by * t_big_pic.m_size.x + l_bx ];
	uchar3 l_bgr = { 0, 0, 0 };

	// compose point from small and big image according alpha channel
	l_bgr.x = l_fg_bgra.x * l_fg_bgra.w / 255 + l_bg_bgr.x * ( 255 - l_fg_bgra.w ) / 255;
	l_bgr.y = l_fg_bgra.y * l_fg_bgra.w / 255 + l_bg_bgr.y * ( 255 - l_fg_bgra.w ) / 255;
	l_bgr.z = l_fg_bgra.z * l_fg_bgra.w / 255 + l_bg_bgr.z * ( 255 - l_fg_bgra.w ) / 255;

	// Store point into image
	t_big_pic.m_p_uchar3[ l_by * t_big_pic.m_size.x + l_bx ] = l_bgr;
}

void cu_insertimage( CudaImg &t_big_pic, CudaImg &t_small_pic, int2 t_position )
{
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 32;
	dim3 l_blocks( ( t_small_pic.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( t_small_pic.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_insertimage<<< l_blocks, l_threads >>>( t_big_pic, t_small_pic, t_position );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

void cu_run_img_ins_sel( CudaImg &t_big_img, CudaImg &t_small_img, int2 t_position,int t_ins_or_select)
{
	cudaError_t l_cerr;
	uint2 t_block_size = {32,32};
	// Grid creation with computed organization
	dim3 l_grid( ( t_big_img.m_size.x + t_block_size.x - 1 ) / t_block_size.x,
			     ( t_big_img.m_size.y + t_block_size.y - 1 ) / t_block_size.y );
	kernel_img_ins_sel<<< l_grid, dim3( t_block_size.x, t_block_size.y ) >>>( t_big_img, t_small_img, t_position,t_ins_or_select);

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();

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

void cu_run_img_rotate90( CudaImg &t_pic_in, CudaImg &t_pic_out )
{
	cudaError_t l_cerr;
	uint2 t_block_size = {32,32};
	// Grid creation with computed organization
	dim3 l_grid( ( t_pic_in.m_size.x + t_block_size.x - 1 ) / t_block_size.x,
			     ( t_pic_in.m_size.y + t_block_size.y - 1 ) / t_block_size.y );
	kernel_run_img_rotate90<<< l_grid, dim3( t_block_size.x, t_block_size.y ) >>>( t_pic_in, t_pic_out);

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
