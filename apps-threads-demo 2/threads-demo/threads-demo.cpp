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
#include <time.h>
#include <sys/time.h>
#include <sys/param.h>
#include <pthread.h>

#define TYPE float

// class for one part of task
class TaskPart
{
public:
	int m_id;                 // user thread identification
	int m_from, m_length;     // data range
	TYPE *m_data;             // array
	TYPE m_max;               // result

	TaskPart( int t_myid, int t_from, int t_length, TYPE *t_data ) :
		m_id( t_myid ), m_from( t_from ), m_length( t_length ), m_data( t_data ) {}

	TYPE get_result() { return m_max; }

	// function search_max search the largest number in part of array
	// from the left (included) up to the right element
	TYPE search_max()
	{
		TYPE l_max = m_data[ m_from ];
		for ( int i = 1; i < m_length; i++ )
			if ( l_max < m_data[ m_from + i ] )
				l_max = m_data[ m_from + i ];
		return l_max;
	}
	void bubbleSort(){


		for(int i = m_from; i < m_length+m_from-1; i++)
		{
			for(int j = m_from; j < m_length+m_from-1; j++)
			{
				if(m_data[j] > m_data[j+1])
				{
					TYPE x = m_data[j];
					m_data[j] = m_data[j+1];
					m_data[j+1] = x;
				}
			}
		}
	}

	void bubbleSort_desc(){
		for(int i = m_from; i < m_length+m_from-1; i++)
		{
			for(int j = m_from; j < m_length+m_from-1; j++)
			{
				if(m_data[j] < m_data[j+1])
				{
					TYPE x = m_data[j];
					m_data[j] = m_data[j+1];
					m_data[j+1] = x;
				}
			}
		}
	}

	void print_Array(){
		for(int i = m_from; i < m_length+m_from; i++){
			printf("Sort :%.2f\n",m_data[i]);
		}
	}
	void print_Array2(){
		for(int i = m_from; i < m_length+m_from; i++){
			printf("Copy array :%.2f\n",m_data[i]);
		}
	}
	void print_Array4(){

		for(int i = m_from; i < m_length+m_from; i++){
			printf("Sort array without threads:%.2f\n",m_data[i]);
		}
	}
	void print_Array1(){
		printf("---------Merge-------------\n");
		for(int i = m_from; i < m_length+m_from; i++){
			printf("Merge :%.2f\n",(float) m_data[i]);
		}
	}
	void print_Array3(){
		for(int i = m_from; i < m_length+m_from; i++){
			printf("Begin :%.2f\n",m_data[i]);
		}
	}

};


void bubbleSort_desc2(int arr[], int n){
	for(int i = n; i >=0; i--)
		{
			for(int j = n; j > n-i; j--)
			{
				if(arr[j] > arr[j-1])
				{
					TYPE x = arr[j];
					arr[j] = arr[j-1];
					arr[j-1] = x;
				}
			}
		}
}
void bubbleSort_desc1(TYPE *m_data,int m_from,int m_length){
	for(int i = m_length+m_from-1; i >= m_from; i--)
	{
		for(int j = m_length+m_from-i-1; j > m_from; j--)
		{
			if(m_data[j] > m_data[j-1])
			{
				TYPE x = m_data[j-1];
				m_data[j-1] = m_data[j];
				m_data[j] = x;
			}
		}
	}
}
void bubbleSort(TYPE *m_data,int m_from,int m_length){


	for(int i = m_from; i < m_length+m_from-1; i++)
	{
		for(int j = m_from; j < m_length+m_from-1; j++)
		{
			if(m_data[j] > m_data[j+1])
			{
				TYPE x = m_data[j+1];
				m_data[j+1] = m_data[j];
				m_data[j] = x;
			}
		}
	}
}
void print_Array5(TYPE *m_data,int m_from,int m_length){
	printf("---------------SORT---------------------\n");
	for(int i = m_from; i < m_length+m_from; i++){
		printf("Sort array without threads:%.2f\n",m_data[i]);
	}
}
TYPE * mergeArrays(TaskPart *a, TaskPart *b){
		int i = a->m_from;
		int j = b->m_from;
		int k = 0;
		TYPE *arr = new TYPE [ a->m_length+b->m_length ];
		while(i<a->m_length && j< (a->m_length + b->m_length))
		{
			if(a->m_data[i] > b->m_data[j])
			{
				arr[k] = a->m_data[i];
				i++; k++;
			}
			else
			{
				arr[k] = b->m_data[j];
				j++; k++;
			}
		}
		while(i<a->m_length)
			arr[k++] = a->m_data[i++];
		while(j<b->m_length+a->m_length)
			arr[k++] = b->m_data[j++];


		for(int h = 0; h < b->m_length + b->m_from; h++){
			a->m_data[h] = arr[h];
		}
		a->m_length = a->m_length + b->m_length;
	}


void *my_thread( void *t_void_arg )
{
	TaskPart *lp_task = ( TaskPart * ) t_void_arg;

	printf( "Thread %d started...\n",
		lp_task->m_id);

	lp_task->bubbleSort_desc();
	//lp_task->print_Array1();

  //  printf( "Found maximum in thread %d is %d\n", lp_task->m_id, lp_task->m_max );

	return NULL;
}
bool compareArray(TYPE* m_data1,TYPE* l_my_array,int l_my_length){
	for(int i = 0; i <l_my_length ; i++){
	   if ( m_data1[i] != l_my_array[i]){
		 //  printf("%d %f %f\n",i, m_data1[i],l_my_array[i]);
	   }
   }
	   return 1;
}
// Time interval between two measurements converted to ms
int timeval_diff_to_ms( timeval *t_before, timeval *t_after )
{
	timeval l_res;
	timersub( t_after, t_before, &l_res );
	return 1000 * l_res.tv_sec + l_res.tv_usec / 1000;
}



#define LENGTH_LIMIT 10000

int main( int t_na, char **t_arg )
{

	int l_my_length = atoi(t_arg[1]);
	if(t_na >= 2)
	{
		printf( "Specify number of elements, at least %d.\n", l_my_length );
	}
	int P  = 0;
	if(t_na == 2){
		P = 2;
		printf("Number of threads: %d\n",P);
	}
	else if(t_na ==3){
		P = atoi(t_arg[2]);
		printf("Number of threads: %d\n",P);
	}

	 // array allocation
	 TYPE *l_my_array = new TYPE [ l_my_length ];
	 if ( !l_my_array )
	 {
		 printf( "Not enough memory for array!\n" );
		 return 1;
	 }

	 // Initialization of random number generator
	 srand( ( int ) time( NULL ) );

	 printf( "Random numbers generetion started..." );
	 for ( int i = 0; i < l_my_length; i++ )
	 {
		l_my_array[ i ] = (float)(((rand() % 200000) - 100000 ) / 100.0);
		 if ( !( i % LENGTH_LIMIT ) )
		 {
		  //   printf( "." );
			 fflush( stdout );
		 }
		 //l_my_array[i] = LENGTH_LIMIT-i;
	 }

	 TYPE *m_data1 = new TYPE [l_my_length];
	 for(int i = 0; i < l_my_length; i++){
		 m_data1[i] = l_my_array[i];
	 }

	 timeval l_time_before, l_time_after;

	 gettimeofday( &l_time_before, NULL );

	 TaskPart *l_tp = (TaskPart*) malloc(sizeof(TaskPart)*P);
	 pthread_t l_pt[P];
	 for(int i = 0; i < P; i++){
		 l_tp[i] =TaskPart(i,  i*l_my_length / P,l_my_length / P, l_my_array );

	 }
	 for(int i = 0; i < P; i++){

		  pthread_create( &l_pt[i], NULL, my_thread, &l_tp[i] );
	  }
	 for(int i = 0; i < P; i++){
		 pthread_join( l_pt[i], NULL );
	 }
	 gettimeofday( &l_time_after, NULL );
	 printf( "The search time of THREADS: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );

	 gettimeofday( &l_time_before, NULL );
	 for(int m = 1; m < P ; m++){
		 mergeArrays(&l_tp[0],&l_tp[m]);
		// l_tp[0].print_Array1();
	 }
	  gettimeofday( &l_time_after, NULL );
	  printf( "The merge search time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );


	  gettimeofday( &l_time_before, NULL );
	  TaskPart l_single1(1, 0, l_my_length, m_data1 );
	  l_single1.bubbleSort_desc();
	  gettimeofday( &l_time_after, NULL );
	  printf( "The search time of COPY: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );


	  if(compareArray(m_data1,l_my_array,l_my_length)==1){
		  printf("Equal\n");
	  }else{
		  printf("Not equal\n");
	  }
	 gettimeofday( &l_time_before, NULL );
	 TaskPart l_single2(2, 0, l_my_length, m_data1 );
	 l_single2.bubbleSort_desc();
	 gettimeofday( &l_time_after, NULL );
	 printf( "The new desc again search time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );

	 gettimeofday( &l_time_before, NULL );
	 TaskPart l_single3(3, 0, l_my_length, m_data1 );
	 l_single3.bubbleSort();
	 gettimeofday( &l_time_after, NULL );
	 printf( "The new asc search time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );
}

