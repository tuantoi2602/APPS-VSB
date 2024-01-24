// **************************************************************************
//
//               Demo program for labs
//
// Subject:      Computer Architectures and Parallel systems
// Author:       Petr Olivka, petr.olivka@vsb.cz, 09/2019
// Organization: Department of Computer Science, FEECS,
//               VSB-Technical University of Ostrava, CZ
//
// File:         Programming interface for LCD module
//
// **************************************************************************

#include "mbed.h"
#include "lcd_lib.h"

DigitalOut g_cs(PTC0, 0);		// chip select
DigitalOut g_reset(PTC1, 0);	// reset
DigitalOut g_rs(PTC2, 0);		// command/data
DigitalOut g_bl(PTC3, 0);		// backlight

// **************************************************************************
// internal functions

#define Nop() { asm( "nop" ); }

// SPI interface
static SPI _spi(PTD2, PTD3, PTD1);

// send command to LCD controller
void lcd_write_command(uint8_t Data)
{
	g_cs = 0;
	Nop();
	g_rs = 0;
	_spi.write(Data);
	g_cs = 1;
}

// send one byte to LCD controller
void lcd_write_data(uint8_t Data)
{
	g_cs = 0;
	Nop();
	g_rs = 1;
	_spi.write(Data);
	g_cs = 1;
}

// send two bytes to LCD controller
void lcd_write_data_16(uint32_t Data)
{
	g_cs = 0;
	Nop();
	g_rs = 1;
	_spi.write(Data >> 8);
	_spi.write(Data);
	g_cs = 1;
}

// **************************************************************************
// LCD programming interface

// HW reset of LCD controller
void lcd_reset()
{
	g_reset = 0;
	wait_ms(100);
	g_reset = 1;
	wait_ms(100);
}

// clear screen
void lcd_clear()
{
	lcd_write_command(0x2c);
	for (int i = 0; i < 320 * 240; i++)
		lcd_write_data_16(0);
}

// LCD controller initialization
void lcd_init()
{
	// init SPI interface
	_spi.format(8, 0);
	_spi.frequency(20000000);

	// HW reset of LCD controller
	lcd_reset();

	// backlight ON
	g_bl = 0;

	// initialization sequence... see documentation
	lcd_write_command(0xCB);
	lcd_write_data(0x39);
	lcd_write_data(0x2C);
	lcd_write_data(0x00);
	lcd_write_data(0x34);
	lcd_write_data(0x02);

	lcd_write_command(0xCF);
	lcd_write_data(0x00);
	lcd_write_data(0XC1);
	lcd_write_data(0X30);

	lcd_write_command(0xE8);
	lcd_write_data(0x85);
	lcd_write_data(0x00);
	lcd_write_data(0x78);

	lcd_write_command(0xEA);
	lcd_write_data(0x00);
	lcd_write_data(0x00);

	lcd_write_command(0xED);
	lcd_write_data(0x64);
	lcd_write_data(0x03);
	lcd_write_data(0X12);
	lcd_write_data(0X81);

	lcd_write_command(0xF7);
	lcd_write_data(0x20);

	lcd_write_command(0xC0);    	//Power control
	lcd_write_data(0x23);   		//VRH[5:0]

	lcd_write_command(0xC1);    	//Power control
	lcd_write_data(0x10);   		//SAP[2:0];BT[3:0]

	lcd_write_command(0xC5);    	//VCM control
	lcd_write_data(0x3e);   		//Contrast
	lcd_write_data(0x28);

	lcd_write_command(0xC7);    	//VCM control2
	lcd_write_data(0x86);   		//--

	lcd_write_command(0x36);    	// Memory Access Control
	//lcd_write_data( 0x48 );
	lcd_write_data(0xE8);

	lcd_write_command(0x2A);
	lcd_write_data_16(0);
	lcd_write_data_16(320 - 1);

	lcd_write_command(0x2B);
	lcd_write_data_16(0);
	lcd_write_data_16(240 - 1);

	lcd_write_command(0x3A);
	lcd_write_data(0x55);

	lcd_write_command(0xB1);
	lcd_write_data(0x00);
	lcd_write_data(0x18);

	lcd_write_command(0xB6);    	// Display Function Control
	lcd_write_data(0x08);
	lcd_write_data(0x82);
	lcd_write_data(0x27);
	/*
	 lcd_write_command( 0xF2 );    	// 3Gamma Function Disable
	 lcd_write_data( 0x00 );

	 lcd_write_command( 0x26 );    	//Gamma curve selected
	 lcd_write_data( 0x01 );

	 lcd_write_command( 0xE0 );    	//Set Gamma
	 lcd_write_data( 0x0F );
	 lcd_write_data( 0x31 );
	 lcd_write_data( 0x2B );
	 lcd_write_data( 0x0C );
	 lcd_write_data( 0x0E );
	 lcd_write_data( 0x08 );
	 lcd_write_data( 0x4E );
	 lcd_write_data( 0xF1 );
	 lcd_write_data( 0x37 );
	 lcd_write_data( 0x07 );
	 lcd_write_data( 0x10 );
	 lcd_write_data( 0x03 );
	 lcd_write_data( 0x0E );
	 lcd_write_data( 0x09 );
	 lcd_write_data( 0x00 );

	 lcd_write_command( 0XE1 );    	//Set Gamma
	 lcd_write_data( 0x00 );
	 lcd_write_data( 0x0E );
	 lcd_write_data( 0x14 );
	 lcd_write_data( 0x03 );
	 lcd_write_data( 0x11 );
	 lcd_write_data( 0x07 );
	 lcd_write_data( 0x31 );
	 lcd_write_data( 0xC1 );
	 lcd_write_data( 0x48 );
	 lcd_write_data( 0x08 );
	 lcd_write_data( 0x0F );
	 lcd_write_data( 0x0C );
	 lcd_write_data( 0x31 );
	 lcd_write_data( 0x36 );
	 lcd_write_data( 0x0F );
	 */
	lcd_write_command(0x11);    	//Exit Sleep
	wait_ms(120);

	lcd_write_command(0x29);    	//Display on
	lcd_write_command(0x2c);
}

// draw one pixel to LCD screen
void lcd_put_pixel(int x, int y, int color)
{
	lcd_write_command(0x2A);
	lcd_write_data_16(x);
	lcd_write_data_16(x);

	lcd_write_command(0x2B);
	lcd_write_data_16(y);
	lcd_write_data_16(y);

	lcd_write_command(0x2C);

	lcd_write_data_16(color);
}

