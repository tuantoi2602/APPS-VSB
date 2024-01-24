// **************************************************************************
//
//               Demo program for labs
//
// Subject:      Computer Architectures and Parallel systems
// Author:       Petr Olivka, petr.olivka@vsb.cz, 09/2019
// Organization: Department of Computer Science, FEECS,
//               VSB-Technical University of Ostrava, CZ
//
// File:         Main program for LEDs
//
// **************************************************************************

#include "mbed.h"

// Serial line for printf output
Serial g_pc(USBTX, USBRX);

// LEDs on K64F-KIT - instances of class DigitalOut
DigitalOut g_led1(PTA1);
DigitalOut g_led2(PTA2);

// Button on K64F-KIT - instance of class DigitalIn
DigitalIn g_but9(PTC9);

int main()
{
	// Serial line initialization
	//g_pc.baud(115200);

	while (1)
	{
		int l_delay = 500;

		g_led1 = !g_led1; 		// invert LED1 state

		if (g_but9 == 0) 		// button pressed?
		{
			l_delay /= 10;		// speed up blinking
			g_led2 = !g_led2;
		}
		else
			g_led2 = 0; 		// LED2 off

		wait_ms(l_delay);
	}
}
