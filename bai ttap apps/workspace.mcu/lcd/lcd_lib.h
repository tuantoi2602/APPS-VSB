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
#ifndef __LCD_LIB_H
#define __LCD_LIB_H

// HW reset of LCD controller
void lcd_reset();

// clear screen
void lcd_clear();

// LCD controller initialization
void lcd_init();

// draw one pixel to LCD screen
void lcd_put_pixel( int x, int y, int color );

#endif // __LCD_LIB_H
