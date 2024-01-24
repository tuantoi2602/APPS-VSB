// **************************************************************************
//
//               Demo program for labs
//
// Subject:      Computer Architectures and Parallel systems
// Author:       Petr Olivka, petr.olivka@vsb.cz, 08/2016
// Organization: Department of Computer Science, FEECS,
//               VSB-Technical University of Ostrava, CZ
//
// File:         Main program for LCD module
//
// **************************************************************************

#include "mbed.h"
#include "lcd_lib.h"

// Serial line for printf output
Serial pc(USBTX, USBRX);

// two dimensional array with fixed size font
extern uint8_t g_font8x8[256][8];


int main()
{

	// Serial line initialization
	//g_pc.baud(115200);

	lcd_init();				// LCD initialization

	lcd_clear();			// LCD clear screen

	int l_color_red = 0xF800;
	int l_color_green = 0x07E0;
	int l_color_blue = 0x001F;
	int l_color_white = 0xFFFF;

	// simple animation display four color square using LCD_put_pixel function
	int l_limit = 200;
	/*for (int ofs = 0; ofs < 20; ofs++) // square offset in x and y axis
		for (int i = 0; i < l_limit; i++)
		{
			lcd_put_pixel(ofs + i, ofs + 0, l_color_red);
			lcd_put_pixel(ofs + 0, ofs + i, l_color_green);
			lcd_put_pixel(ofs + i, ofs + l_limit, l_color_blue);
			lcd_put_pixel(ofs + l_limit, ofs + i, l_color_white);

		}*/
/*	for (int ofs = 0; ofs < 20; ofs++) // square offset in x and y axis
		for (int i = 0; i < l_limit; i++)
		{
			lcd_put_pixel(ofs + i, ofs + 0, l_color_red);
		}*/
	/*for(int i = 0; i < 240; i++){
		lcd_put_pixel(i+10,i+20,l_color_red);
	}*/
	 for (int i = 0; i < 360; i++){
		 for(int y = 0; y < 360; y++)
		 lcd_put_pixel(i,i+y,l_color_red);
	 }
	return 0;
}
