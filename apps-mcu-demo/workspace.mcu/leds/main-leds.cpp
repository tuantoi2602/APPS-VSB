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
DigitalOut g_ptc0(PTC0);
DigitalOut g_ptc1(PTC1);
DigitalOut g_ptc2(PTC2);
DigitalOut g_ptc3(PTC3);
DigitalOut g_ptc4(PTC4);
DigitalOut g_ptc5(PTC5);
DigitalOut g_ptc7(PTC7);
DigitalOut g_ptc8(PTC8);
DigitalOut g_red1(PTB9);
DigitalOut g_green1(PTB3);
DigitalOut g_blue1(PTB2);
DigitalOut g_red2(PTB19);
DigitalOut g_green2(PTB18);
DigitalOut g_blue2(PTB11);
DigitalOut Color[6] = {g_red1,g_green1,g_blue1,g_red2,g_green2,g_blue2};

// Button on K64F-KIT - instance of class DigitalIn
DigitalIn g_but9(PTC9);
DigitalIn g_but10(PTC10);
DigitalIn g_but11(PTC11);
DigitalIn g_but12(PTC12);

void my_blink_function1()
{
	g_led1= !g_led1;
}
void my_blink_function2()
{
	g_led2= !g_led2;
	if(g_but9 == 1)
		g_led2=!g_led2;
	else
		g_led2=0;
}
void my_blink_functionptc0()
{
	g_led2= !g_led2;
	if(g_but10 == 0)
		g_led2=!g_led2;
	else
		g_led2=0;
}

DigitalOut LED[8] = {g_ptc0,g_ptc1,g_ptc2,g_ptc3,g_ptc4,g_ptc5,g_ptc7,g_ptc8};
int brightness1[8] = {0,10,20,30,40,50,60,70};
int brightness_c[6] = {0,0,0,0,0,0};

int timer = 0;
void ledtinker(){
	if(timer >= 20){
		timer = 0;
	}

		for(int j=0; j<6;j++){
			int T1 = 20 * brightness_c[j] / 100;
			if(timer < T1){
				Color[j] = 1;
			}
			else{
				Color[j] = 0;
			}
		}
		timer++;
}

void ledf(){
		while(1){
			for(int i = 0; i < 25;i++){
				for(int j = 0; j < 3; j++){
					int T1 = 20 * brightness_c[j]/100;
					if(i < T1)
						Color[j] = 1;
					else
						Color[j] = 0;
				}
			}
			if(!g_but9){
				brightness_c[0]++;
				wait_ms(20);
			}
			if(g_but9 == 0){
				brightness_c[0]--;
				wait_ms(20);
			}
			if(!g_but10){
				brightness_c[1]++;
				wait_ms(20);
			}
			if(!g_but11){
				brightness_c[2]++;
				wait_ms(20);
			}
			}
}
int main(){
	Ticker t;
	t.attach_us(ledtinker, 1000);
	int count = 0;
	while(1){
		if(g_but12 %2 == 1){
		if(!g_but9){
			if(brightness_c[0] >= 100)
				brightness_c[0] = 100;
			brightness_c[0]++;
			if(brightness_c[0] >= 100)
				brightness_c[3]++;
			wait_ms(20);
		}
		if(!g_but10){
			if(brightness_c[1] >= 100)
				brightness_c[1] = 100;
			brightness_c[1]++;
			if(brightness_c[1] >= 100)
				brightness_c[4]++;
			wait_ms(20);
		}
		if(!g_but11){
			if(brightness_c[2] >= 100)
				brightness_c[2] = 100;
			brightness_c[2]++;
			if(brightness_c[2] >= 100)
				brightness_c[5]++;
			wait_ms(20);
		}
	}if(g_but12 %2 == 0){
		if(!g_but9){
			if(brightness_c[0] <= 0)
				brightness_c[0] = 0;
			brightness_c[3]--;
			if(brightness_c[3] <= 0)
				brightness_c[0]--;
			wait_ms(20);
		}
		if(!g_but10){
			if(brightness_c[1] <= 0)
				brightness_c[1] = 0;
			brightness_c[4]--;
			if(brightness_c[4] <= 0)
				brightness_c[1]--;
			wait_ms(20);
		}
		if(!g_but11){
			if(brightness_c[2] <= 0)
				brightness_c[2] = 0;
			brightness_c[5]--;
			if(brightness_c[5] <= 0)
				brightness_c[2]--;
			wait_ms(20);
		}
	}if(!g_but12){
		count++;
		wait_ms(200);}

	if(count == 2){

		brightness_c[0] = 0;
		brightness_c[1] = 0;
		brightness_c[2] = 0;
		brightness_c[3] = 0;
		brightness_c[4] = 0;
		brightness_c[5] = 0;


		count = count - 2;
	}
}
}





