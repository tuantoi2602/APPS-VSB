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
void cu_run_img_rotate90( CudaImg &t_in_img, CudaImg &t_out_img );
// Image size
void cu_run_my_remove( CudaImg &t_pic_in, CudaImg &t_pic_out );

void cu_run_img_ins_sel( CudaImg &t_big_img, CudaImg &t_small_img, int2 t_position,int t_ins_or_select );
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
	cv:: Mat l_cv_img_big = cv::imread("big.jpg",cv::IMREAD_COLOR);
	cv:: Mat l_cv_img_small = cv::imread("tree.jpg", cv::IMREAD_COLOR);


	cv::imshow("Big", l_cv_img_big);
	cv::imshow("Small", l_cv_img_small);



	CudaImg l_cuda_big_img(l_cv_img_big);
	CudaImg l_cuda_small_img(l_cv_img_small);

	cu_run_img_ins_sel(l_cuda_big_img,l_cuda_small_img, {120,120});
	cv::imshow("New Big", l_cv_img_big);
	cv::imshow("New Small", l_cv_img_small);
*/



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

	cv:Mat in= cv::imread("chim.jpg",cv::IMREAD_COLOR);
	cv::Mat out(in.size().width,in.size().height,CV_8UC3);
	cv::imshow("Keyboard",in);
	CudaImg l_pic_in(in);
	CudaImg l_pic_out(out);
	cu_run_img_rotate90(l_pic_in,l_pic_out);
	cv::imshow("Rotate",out);



	cv::Mat out1(in.size().height,in.size().width,CV_8UC3);
	cv::imshow("Keyboard",out);
	CudaImg l_pic_out1(out1);
	cu_run_img_rotate90(l_pic_out,l_pic_out1);
	cv::imshow("Rotate",out1);


	cv:Mat key= cv::imread("mix.png",cv::IMREAD_COLOR);
	cv::imshow("Keyboard",key);
	CudaImg l_key(key);
	cv::Mat l_key_r(key.size().height,key.size().width,CV_8UC3);
	CudaImg l_key_r1(l_key_r);
	cu_run_my_remove(l_key,l_key_r1);
	cv::imshow("dsasdas",l_key_r);
*/
// 3

	cv:: Mat img1 = cv::imread("chim.jpg",cv::IMREAD_COLOR);
	cv:: Mat img2 = cv::imread("chim1.jpg",cv::IMREAD_COLOR);
	CudaImg l_pic_in(img1);
	CudaImg l_pic_out3(img2);

	//rotate 90
	cv::Mat img_rot(img1.size().width,img1.size().height,CV_8UC3);
	CudaImg l_pic_out(img_rot);
	cu_run_img_rotate90(l_pic_in,l_pic_out);
	//rotate 180
	cv::Mat out1(img1.size().height,img1.size().width,CV_8UC3);
	CudaImg l_pic_out1(out1);
	cu_run_img_rotate90(l_pic_out,l_pic_out1);
	cv::imshow("Rotate",out1);
	//select
	cv::Mat l_cv_tmp3(img1.size().height*50/100, img1.size().width*50/100, CV_8UC3);
	CudaImg l_cuda_tmp3(l_cv_tmp3);
	cu_run_img_ins_sel(l_pic_out1,l_cuda_tmp3,{img1.size().width/2,0},1);
	cv::imshow("1/4",l_cv_tmp3);


	cv::Mat l_cv_tmp4(img1.size().height*50/100, img1.size().width*50/100, CV_8UC3);
	CudaImg l_cuda_tmp4(l_cv_tmp4);
	cu_run_img_ins_sel(l_pic_out1,l_cuda_tmp4,{0,img1.size().height/2},1);
	cv::imshow("1/4 other",l_cv_tmp4);


	//insert
	cu_run_img_ins_sel(l_pic_out3,l_cuda_tmp3,{img1.size().width/2,0},0);
	cu_run_img_ins_sel(l_pic_out3,l_cuda_tmp4,{0,img1.size().height/2},0);

	cv::imshow("full image",img2);



	cv::waitKey( 0 );

}
