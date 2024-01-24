// ***********************************************************************
//
// Demo program for subject Computer Architectures and Paralel systems
// Petr Olivka, Dept. of Computer Science, FEECS, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Threads programming example for Linux (11/2019)
// For the propper testing is necessary to have at least 2 cores CPU
//
// ***********************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

struct my_data{
	char * m_name;
	int m_limit;
	int m_timeout;
};

void *my_first_thread(void *arg){
	my_data *l_pmd = (my_data * ) arg;

	for(int i = 0; i < l_pmd ->m_limit;i++){
		printf("Code of my thread %s: i = %d\n",l_pmd->m_name,i);
		sleep(l_pmd->m_timeout );
	}
	return nullptr;
}

int main(int num_arg, char **args){
	pthread_t l_th1, l_th2;
	my_data l_md1 = {"tyger",5,2};
	my_data l_md2 = {"lion",7,1};
	pthread_create (&l_th1, nullptr, my_first_thread,&l_md1);
	pthread_create (&l_th2, nullptr, my_first_thread,&l_md2);

	pthread_join(l_th1,nullptr);
	pthread_join(l_th2,nullptr);

	printf("main finished!\n");
}
