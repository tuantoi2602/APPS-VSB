// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage without unified memory.
//
// Image creation and its modification using CUDA.
// Image manipulation is performed by OpenCV library.
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"
#include "font32x53_lsb.h"

using namespace cv;
// Prototype of function in .cu file
void cu_run_animation( CudaImg t_pic, uint2 t_block_size );
void cu_run_rgb_separation(CudaImg &t_pic_in,CudaImg &t_pic_out_r, CudaImg &t_pic_out_g, CudaImg &t_pic_out_b);
void cu_run_img_rotate90( CudaImg &t_in_img, CudaImg &t_out_img );
// Image size
void cu_run_my_remove( CudaImg &t_pic_in, CudaImg &t_pic_out );

void cu_run_img_ins_sel( CudaImg &t_big_img, CudaImg &t_small_img, int2 t_position,int t_ins_or_select );
void cu_insertimage( CudaImg &t_big_pic, CudaImg &t_small_pic, int2 t_position );
void cu_draw_char( CudaImg &t_img, int2 t_pos, uchar3 t_color, char t_char, CudaImg &t_font, uint2 t_font_size );

void draw_text( CudaImg &t_img, int2 t_pos, uchar3 t_color, char* t_text, CudaImg &t_font, uint2 t_font_size ){
	for ( int inx = 0; inx < strlen( t_text ); inx++ ){
	cu_draw_char( t_img, t_pos, t_color, t_text[inx], t_font, t_font_size );
	t_pos.x = t_pos.x + t_font_size.x;
	}
}
int main()
{

	// Uniform Memory allocator for Mat
	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator( &allocator );
	// Creation of empty image.

	// Image filling by color gradient blue-green-red
   // CudaImg l_cuda_img(l_cv_img);
	// Show modified image
	//cv::imshow( "B-G-R Gradient & Color Rotation", l_cv_img );


	cv::Mat l_cv_img =cv::imread("logo.jpg",cv::IMREAD_COLOR );
	cv::Mat l_cv_small_img =cv::imread("index.jpeg",cv::IMREAD_COLOR );
	cv::Mat l_cv_empty(l_cv_img.size().width, l_cv_img.size().width, CV_8UC3);
	cv::Mat l_cv_font(256,53, CV_32S);
	memcpy(l_cv_font.data,font,sizeof(font));

	char* text1 = "GPU";
	char* text2 = "Computing";


	CudaImg l_cuda_img(l_cv_img);
	CudaImg l_cuda_font(l_cv_font);
	CudaImg l_cuda_small_img(l_cv_small_img);
	CudaImg l_cuda_empty(l_cv_empty);

	draw_text( l_cuda_empty, {50,10}, {255,255,255},text1, l_cuda_font, {32,53});
	draw_text( l_cuda_empty, {50,70}, {255,255,255},text2, l_cuda_font, {32,53});


	cu_insertimage(l_cuda_empty,l_cuda_small_img,{1300,90});
	cv::imshow("Image", l_cv_empty);

	cv::waitKey( 0 );

}
