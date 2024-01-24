// **************************************************************************
//
//               Demo program for labs
//
// Subject:      Computer Architectures and Parallel systems
// Author:       Petr Olivka, petr.olivka@vsb.cz, 09/2019
// Organization: Department of Computer Science, FEECS,
//               VSB-Technical University of Ostrava, CZ
//
// File:         OpenCV simulator of LCD
//
// **************************************************************************
#ifndef __LCD_LIB_H
#define __LCD_LIB_H

#include <opencv2/opencv.hpp>

#define LCD_WIDTH       320
#define LCD_HEIGHT      240
#define LCD_NAME        "Virtual LCD"

// LCD Simulator

// Virtual LCD
extern cv::Mat g_canvas;

// Put color pixel on LCD (canvas)
void lcd_put_pixel( int t_x, int t_y, int t_rgb_565 );

// Clear LCD
void lcd_clear();

// LCD Initialization 
void lcd_init();

#endif // __LCD_LIB_H

