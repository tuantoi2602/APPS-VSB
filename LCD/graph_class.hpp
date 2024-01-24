

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


#include "lcd_lib.h"


// Simple graphic interface


struct Point2D 

{

    int32_t x, y;

};


struct RGB

{

    uint8_t r, g, b;

};


class GraphElement

{

public:

    // foreground and background color

    RGB m_fg_color, m_bg_color;


    // constructor

    GraphElement( RGB t_fg_color, RGB t_bg_color ) : 

        m_fg_color( t_fg_color ), m_bg_color( t_bg_color ) {}


    // ONLY ONE INTERFACE WITH LCD HARDWARE!!!

    void drawPixel( int32_t t_x, int32_t t_y ) { lcd_put_pixel( t_x, t_y, convert_RGB888_to_RGB565( m_fg_color ) ); }

    

    // Draw graphics element

    virtual void draw() = 0;

    uint16_t convert_RGB888_to_RGB565( RGB t_color ) {
    union URGB {struct {int b:5; int g:6; int r:5;}; short rgb565; } urgb;
    urgb.r = (t_color.r >> 3) & 0x1F;
    urgb.g = (t_color.g >> 2) & 0x3F;
    urgb.b = (t_color.b >> 3) & 0x1F;
    return urgb.rgb565;; /* green color */ }

    // Hide graphics element

    virtual void hide() { swap_fg_bg_color(); draw(); swap_fg_bg_color(); }

private:

    // swap foreground and backgroud colors

    void swap_fg_bg_color() { RGB l_tmp = m_fg_color; m_fg_color = m_bg_color; m_bg_color = l_tmp; }


    // IMPLEMENT!!!

    // conversion of 24-bit RGB color into 16-bit color format


};



class Pixel : public GraphElement

{

public:

    // constructor

    Pixel( Point2D t_pos, RGB t_fg_color, RGB t_bg_color ) : GraphElement( t_fg_color, t_bg_color ), m_pos( t_pos ) {}

    // Draw method implementation

    virtual void draw() { drawPixel( m_pos.x, m_pos.y ); }

    // Position of Pixel

    Point2D m_pos;

};



class Circle : public GraphElement

{

public:

    // Center of circle

    Point2D m_center;

    // Radius of circle

    int32_t radius;


    Circle( Point2D t_center, int32_t t_radius, RGB t_fg, RGB t_bg ) :

        GraphElement( t_fg, t_bg ), m_center( t_center ), radius( t_radius ) {}


    void draw() {

    	int f = 1 - radius;

    	int df_x = 0;

    	int df_y = -2 * radius;

    	int x = 0;

    	int y = radius;


    	int x0 = m_center.x;

    	int y0 = m_center.y;


    	drawPixel(x0,y0 + radius);

    	drawPixel(x0,y0 - radius);

    	drawPixel(x0 + radius, y0);

    	drawPixel(x0 - radius,y0);


    	while(x<y){

    		if(f >=0){

    			y--;

    			df_y +=2;

    			f+=df_y;

    		}

    		x++;

    		df_x += 2;

    		f+= df_x;


    		drawPixel(x0 + x, y0 + y);

    		drawPixel(x0 - x, y0 + y);

    		drawPixel(x0 + x, y0 - y);

    		drawPixel(x0 - x, y0 - y);

    		drawPixel(x0 + y, y0 + x);

    		drawPixel(x0 - y, y0 + x);

    		drawPixel(x0 + y, y0 - x);

    		drawPixel(x0 - y, y0 - x);

    		}

    } // IMPLEMENT!!!

};

class Character : public GraphElement
{
public:
    // position of character
    Point2D pos;
    // character
    char character;

    Character( Point2D t_pos, char t_char, RGB t_fg, RGB t_bg ) :
      pos( t_pos ), character( t_char ), GraphElement( t_fg, t_bg ) {};

    void draw() {
    	for(int i = 0; i <8 ; i++){
    		for(int j = 0; j <8 ; j++ ){
    			if(font8x8[character][i]&(1<<j)){
    				drawPixel(j+pos.x,i+pos.y);
    			}
    		}
    	}
     };
};


class Line : public GraphElement
{
public:
    // the first and the last point of line
    Point2D pos1, pos2;
    Line( Point2D t_pos1, Point2D t_pos2, RGB t_fg, RGB t_bg ) :
      pos1( t_pos1 ), pos2( t_pos2 ), GraphElement( t_fg, t_bg ) {}
    void draw()
    {
        int x0 = pos1.x;
        int y0 = pos1.y;
        int x1 = pos2.x;
        int y1 = pos2.y;
        int dx =  abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy, e2; //error value e_xy
        for(;;){  //loop
            drawPixel(x0, y0);
            if (x0 == x1 && y0 == y1) break;
            e2 = 2*err;
            if (e2 >= dy) { err += dy; x0 += sx; } //e_xy+e_x > 0
            if (e2 <= dx) { err += dx; y0 += sy; } //e_xy+e_y < 0
        }
    }
}; // IMPLEMENT!!!

