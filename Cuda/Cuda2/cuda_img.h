// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
//
// Image interface for CUDA
//
// ***********************************************************************

#pragma once
#include <opencv2/opencv.hpp>

// Structure definition for exchanging data between Host and Device
struct CudaImg
{
  uint3 m_size;				// size of picture
  union {
	  void   *m_p_void;		// data of picture
	  uchar1 *m_p_uchar1;	// data of picture
	  uchar3 *m_p_uchar3;	// data of picture
	  uchar4 *m_p_uchar4;	// data of picture

  };

__host__ CudaImg(cv::Mat &t_img){
	  this->m_size.x = t_img.cols;
	  this->m_size.y = t_img.rows;
	  this->m_p_void = t_img.data;

 }
__device__ uchar1 &at1(int t_y, int y_x){
	return m_p_uchar1[t_y * m_size.x + y_x];
}
__device__ uchar3 &at3(int t_y, int y_x){
	return m_p_uchar3[t_y * m_size.x + y_x];
}
__device__ uchar4 &at4(int t_y, int y_x){
	return m_p_uchar4[t_y * m_size.x + y_x];
}
};


/*
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
void cu_run_rgb_separation(CudaImg t_pic_in,CudaImg t_pic_out_r, CudaImg t_pic_out_g, CudaImg t_pic_out_b);
// Image size


int main()
{
	// Uniform Memory allocator for Mat
	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator( &allocator );
	// Creation of empty image.
	cv::Mat l_cv_img =cv::imread("gdg.png",cv::IMREAD_COLOR );
	// Image filling by color gradient blue-green-red
    CudaImg l_cuda_img(l_cv_img);
    	//t_pic_out.at3(l_y,t_pic_in.m_size.x-l_x) = l_bgr;
	// Show modified image
	cv::imshow( "B-G-R Gradient & Color Rotation", l_cv_img );



	cv:Mat gdg= cv::imread("gdg.png",cv::IMREAD_COLOR);
	cv::imshow("Google Developer Group",gdg);
	CudaImg l_gdg(gdg);
	cv::Mat l_gdg_r(gdg.rows,gdg.cols,CV_8UC3);
	cv::Mat l_gdg_g(gdg.rows,gdg.cols,CV_8UC3);
	cv::Mat l_gdg_b(gdg.rows,gdg.cols,CV_8UC3);
	CudaImg l_gdg_r1(l_gdg_r);
	CudaImg l_gdg_g1(l_gdg_g);
	CudaImg l_gdg_b1(l_gdg_b);
	cu_run_rgb_separation(l_gdg,l_gdg_r1,l_gdg_g1,l_gdg_b1);
	cv::imshow("Red",l_gdg_r);
	cv::imshow("Green",l_gdg_g);
	cv::imshow("Blue",l_gdg_b);
	cv::waitKey( 0 );

}

*/
/*

*/

