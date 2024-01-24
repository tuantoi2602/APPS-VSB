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
using namespace cv;
// Prototype of function in .cu file
void cu_run_animation( CudaImg t_pic, uint2 t_block_size );
void cu_run_rgb_separation(CudaImg &t_pic_in,CudaImg &t_pic_out_r, CudaImg &t_pic_out_g, CudaImg &t_pic_out_b);
void cu_run_my_mirror( CudaImg &t_pic_in, CudaImg &t_pic_out );
// Image size
void cu_run_my_remove( CudaImg &t_pic_in, CudaImg &t_pic_out );

int main()
{
	// Uniform Memory allocator for Mat
	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator( &allocator );
	// Creation of empty image.
	cv::Mat l_cv_img =cv::imread("gdg.png",cv::IMREAD_COLOR );
	// Image filling by color gradient blue-green-red
   // CudaImg l_cuda_img(l_cv_img);
	// Show modified image
	//cv::imshow( "B-G-R Gradient & Color Rotation", l_cv_img );



/*
	cv:Mat gdg= cv::imread("gdg.png",cv::IMREAD_COLOR);
	cv::imshow("Google Developer Group",gdg);
	CudaImg l_gdg(gdg);
	cv::Mat l_gdg_r(gdg.size().height,gdg.size().width,CV_8UC3);
	cv::Mat l_gdg_g(gdg.size().height,gdg.size().width,CV_8UC3);
	cv::Mat l_gdg_b(gdg.size().height,gdg.size().width,CV_8UC3);
	CudaImg l_gdg_r1(l_gdg_r);
	CudaImg l_gdg_g1(l_gdg_g);
	CudaImg l_gdg_b1(l_gdg_b);
	cu_run_rgb_separation(l_gdg,l_gdg_r1,l_gdg_g1,l_gdg_b1);
	cv::imshow("Red",l_gdg_r);
	cv::imshow("Green",l_gdg_g);
	cv::imshow("Blue",l_gdg_b);
*/


/*
	cv:Mat key= cv::imread("key.png",cv::IMREAD_COLOR);
	cv::imshow("Keyboard",key);
	CudaImg l_key(key);
	cv::Mat l_key_r(key.size().height,key.size().width,CV_8UC3);
	CudaImg l_key_r1(l_key_r);
	cu_run_my_mirror(l_key,l_key_r1);
	cv::imshow("dsasdas",l_key_r);
*/


	cv:Mat key= cv::imread("mix.png",cv::IMREAD_COLOR);
	cv::imshow("Keyboard",key);
	CudaImg l_key(key);
	cv::Mat l_key_r(key.size().height,key.size().width,CV_8UC3);
	CudaImg l_key_r1(l_key_r);
	cu_run_my_remove(l_key,l_key_r1);
	cv::imshow("dsasdas",l_key_r);



	cv::waitKey( 0 );

}
