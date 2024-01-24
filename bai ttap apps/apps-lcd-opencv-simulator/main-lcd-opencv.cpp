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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "lcd_lib.h"
#include "font8x8.h"

//#include "graph_struct.hpp"
#include "graph_class.hpp"
Point2D point1 = {70,120};
Point2D point2 = {250,120};

Point2D point3 = {160,200};
Point2D point4 = {160, 60};

Point2D point5 = {55, 119};

Point2D point6 = {586,119};
Point2D point7 = {160,44};
Point2D point8 = {160,216};

Point2D st = {20,220};

RGB black = {0, 0, 0};
RGB white = {255, 255, 255};
RGB bordo = {128, 0, 32};
RGB cyan = {0, 255, 255};
RGB green = {0, 255, 0};
RGB blue = {0, 0, 255};
RGB red = {255, 0, 0};
RGB navajowhite = {255,222,173};


/*
Line l2(point3,point4,blue,white);
Circle c1(point5,15, red, black);
Circle c2(point6,15, red, black);
Circle c3(point7,15,red,black);
Circle c4(point8,15,red,black);
Character ch1(point5,'N',white,blue);
Character ch2(point6,'U',white,blue);
Character ch3(point7,'0',white,blue);
Character ch4(point8,'G',white,blue);
/*Character p1(point2,'o',navajowhite,blue);*/



/*Circle c2(point1,20,bordo,blue);*/

int main(){




    lcd_init();                     // LCD initialization

    lcd_clear();                    // LCD clear screen
while(1){
	for(float x = 0; x < 360; x+=0.01){
		if(x >= 0 && x <= 359){
    float sin1 = sin(x);
    float cos1 = cos(x);
    int sx = 120;
    int sy = 120;
    int r1 = 90;
    int r2 = 90;
    int sx1 = 140;
    int sy1 = 140;


    Line line1({(sx + sin1 * r1), (sy + cos1 * r2)},{(sx-sin1 *r1),(sy-cos1*r1)}, green,black);
    Line line2({(sx - cos1 * r1), (sy + sin1 * r2)},{(sx+cos1 *r1),(sy-sin1*r1)}, blue, black);

    Circle c1({(sx + sin1 * (r1+13)), (sy + cos1 * (r2+13))},13, red, black);
    Circle c2({(sx-sin1 *(r1+13)),(sy-cos1*(r1+13))},13, red, black);
    Circle c3({(sx - cos1 * (r1+13)), (sy + sin1 * (r2+13))},13, red, black);
    Circle c4({(sx+cos1 *(r1+13)),(sy-sin1*(r1+13))},13, red, black);

    Character ch1({(sx + sin1 * (r1+13)), (sy + cos1 * (r2+13))},'N',white,black);
    Character ch2({(sx-sin1 *(r1+13)),(sy-cos1*(r1+13))},'U',white,black);
    Character ch3({(sx - cos1 * (r1+13)), (sy + sin1 * (r2+13))},'0',white,black);
    Character ch4({(sx+cos1 *(r1+13)),(sy-sin1*(r1+13))},'G',white,black);

     line1.draw();
     line2.draw();
     c1.draw();
     c2.draw();
     c3.draw();
     c4.draw();
     ch1.draw();
     ch2.draw();
     ch3.draw();
     ch4.draw();
     cv::imshow( LCD_NAME, g_canvas );   // refresh content of "LCD"
     cv::waitKey(10);
     line1.hide();
     line2.hide();
     c1.hide();
     c2.hide();
     c3.hide();
     c4.hide();
     ch1.hide();
     ch2.hide();
     ch3.hide();
     ch4.hide();
		}
	}
}





    /*   int l_color_red = 0xF800;
       int l_color_green = 0x07E0;
       int l_color_blue = 0x001F;
       int l_color_white = 0xFFFF;
       // simple animation display four color square using LCD_put_pixel function
       int l_limit = 200;
       for ( int ofs = 0; ofs < 20; ofs++ ) // square offset in x and y axis
           for ( int i = 0; i < l_limit; i++ )
           {
               lcd_put_pixel(ofs + i, ofs + 0, l_color_red);
               lcd_put_pixel(ofs + 0, ofs + i, l_color_green);
               lcd_put_pixel(ofs + i, ofs + l_limit, l_color_blue);
               lcd_put_pixel(ofs + l_limit, ofs + i, l_color_white);
           }*/
   /* p9.draw();
    p1.draw();*/
    /*

    c1.draw();
    c2.draw();
    c3.draw();
    c4.draw();
    l1.draw();
    l2.draw();*/


}



