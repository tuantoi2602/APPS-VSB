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

    //from the lowest to the biggest
    void insertion_sort_asc(){
    	for(int i=m_from; i<m_length+m_from-1;i++){
    		int j = i+1;
    		TYPE tmp = m_data[j];
    		while (j > m_from && tmp < m_data[j-1]){
    			m_data[j] = m_data[j-1];
    			j--;
    		}
    		m_data[j] = tmp;
    	}
    }

    void print_array_asc(){
		printf("\n---------------sorting asc------------------\n");
		for(int i = m_from; i < m_length + m_from; i++){
			printf("%.2f ,", (float) m_data[i] );
		}
		printf("\n");

    }

    //from the biggest to the lowest
    void insertion_sort_desc(){
    	for(int i = m_from; i < m_length + m_from - 1; i++){
    		int j = i+1;
    		TYPE tmp = m_data[j];
    		while (j > m_from && tmp > m_data[j-1]){
    			m_data[j] = m_data[j-1];
    			j--;
    		}
    		m_data[j] = tmp;
    	}
    }

    void print_array_desc(){
    	printf("\n---------------sorting desc------------------\n");
    	for(int i=m_from; i<m_length+m_from;i++){
    		printf("%.2f ,", (float) m_data[i] );
    	}
    	printf("\n");
    }

    void print_array(){
    	printf("\n-------------- merge ------------------\n");
    	for(int i=m_from; i<m_length+m_from;i++){
    		printf("%.2f ,", (float) m_data[i] );
    	}
    	printf("\n");
    }

};

void merge( TaskPart* a , TaskPart* b){
	int i = a->m_from;
	int j = b->m_from;
	int k = 0;
	TYPE *arr = new TYPE [ a->m_length+b->m_length ];
	while(i<a->m_length && j< (a->m_length + b->m_length))
	{
		if(a->m_data[i] < b->m_data[j])
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

bool compare(TYPE* l_my_array, TYPE* l_my_array_copy, int length){
	for(int i = 0; i < length; i++){
	   if ( l_my_array_copy[i] != l_my_array[i] ){
		   return 0;
		   break;
	   }
   }
	   return 1;
}


// Thread will search the largest element in array
// from element arg->m_from with length of arg->m_length.
// Result will be stored to arg->m_max.
void *my_thread( void *t_void_arg )
{
    TaskPart *lp_task = ( TaskPart * ) t_void_arg;

    printf( "Thread %d started from %d with length %d...\n",
        lp_task->m_id, lp_task->m_from, lp_task->m_length );

    //lp_task->m_max = lp_task->search_max();
    lp_task->insertion_sort_desc();
    //lp_task->insertion_sort_asc();
    //lp_task->print_array_desc();


   // printf( "Found maximum in thread %d is %d\n", lp_task->m_id, lp_task->m_max );
    //insertion_sort_asc();
    //print_array(lp_task->m_data, lp_task->m_from, lp_task->m_length);


    return NULL;
}


// Time interval between two measurements converted to ms
int timeval_diff_to_ms( timeval *t_before, timeval *t_after )
{
    timeval l_res;
    timersub( t_after, t_before, &l_res );
    return 1000 * l_res.tv_sec + l_res.tv_usec / 1000;
}


#define LENGTH_LIMIT 100000


int main( int t_na, char **t_arg )
{
    // The number of elements must be used as program argument
	int l_my_length = atoi( t_arg[ 1 ] );
	if ( t_na >= 2 )
	{
		printf( "Number of elemenents: %d.\n", l_my_length );
	}

    int P = 0;
    if (t_na == 2){
     	P=2;
    	printf( "Number of threads: %d.\n", P );
    }

    else if (t_na == 3){
    	P = atoi( t_arg[ 2 ] );
    	printf( "Number of threads: %d.\n", P );
    }
    // array allocation
    TYPE *l_my_array = new TYPE [ l_my_length ];

    if ( !l_my_array ) 
    {
        printf( "Not enought memory for array!\n" );
        return 1;
    }

    // Initialization of random number generator
    srand( ( int ) time( NULL ) );

    //printf( "Random numbers generetion started..." );
   for ( int i = 0; i < l_my_length; i++ )
    {
	   	l_my_array[ i ] = ( (float) (rand()% 200000)-100000)/100.0;
        if ( !( i % LENGTH_LIMIT ) ) 
        {
            printf( "." ); 
            fflush( stdout );
        }
    }
   TYPE *l_my_array_copy = new TYPE [ l_my_length ];
   for(int i = 0; i < l_my_length-1; i++){
		l_my_array_copy[i] = l_my_array[i];
   }


   printf("\n--------------------Sorting using %d threads----------------------\n", P);

   timeval l_time_before, l_time_after;
   //Time recording before searching
   gettimeofday( &l_time_before, NULL );
    //printf( "\nMaximum number search using two threads...\n" );
   TaskPart* l_tp = (TaskPart*) malloc(sizeof(TaskPart)*P);
   pthread_t l_pt[P];

   for(int i = 0; i < P; i++){
	   l_tp[i] = TaskPart( i, i*l_my_length/P, l_my_length / P, l_my_array);
   }
   for(int i = 0; i < P; i++){
	   // Threads starting
	   pthread_create( &l_pt[i], NULL, my_thread, &l_tp[i] );
   }
   for(int i = 0; i < P; i++){
	   // Waiting for threads completion
	   pthread_join( l_pt[i], NULL );
   }
   // Time recording after searching
   gettimeofday( &l_time_after, NULL );
   printf( "The sort time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );


   gettimeofday( &l_time_before, NULL );
   for(int j = 1; j < P; j++){
       	merge(&l_tp[0],&l_tp[j]);
       	//l_tp[0].print_array();
   }
   gettimeofday( &l_time_after, NULL );
   printf( "The MERGE time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );


   printf("\n--------------------Sorting without using thread----------------------\n");
   gettimeofday( &l_time_before, NULL );

   // Searching in single thread
   TaskPart l_single( 333, 0, l_my_length, l_my_array_copy );
   l_single.insertion_sort_desc();

   gettimeofday( &l_time_after, NULL );
   printf( "The sort time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );

   gettimeofday( &l_time_before, NULL );

   // Searching in single thread
   TaskPart l_single1( 334, 0, l_my_length, l_my_array_copy );
   l_single1.insertion_sort_desc();

   gettimeofday( &l_time_after, NULL );
   printf( "The sort time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );

   // Searching in single thread
   TaskPart l_single2( 335, 0, l_my_length, l_my_array_copy );
   l_single2.insertion_sort_asc();

   gettimeofday( &l_time_after, NULL );
   printf( "The sort time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );


    /*timeval l_time_before, l_time_after;

    // Time recording before searching
    gettimeofday( &l_time_before, NULL );

    // Threads starting
    pthread_create( &l_pt1, NULL, my_thread, &l_tp1 );
    pthread_create( &l_pt2, NULL, my_thread, &l_tp2 );
    pthread_create( &l_pt3, NULL, my_thread, &l_tp3 );
    pthread_create( &l_pt4, NULL, my_thread, &l_tp4 );

    // Waiting for threads completion 
    pthread_join( l_pt1, NULL );
    pthread_join( l_pt2, NULL );
    pthread_join( l_pt3, NULL );
    pthread_join( l_pt4, NULL );*/

   /* // Time recording after searching
    gettimeofday( &l_time_after, NULL );

    printf( "The found maximum: %d\n", MAX( l_tp1.get_result(), l_tp2.get_result() ) );
    printf( "The search time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );

    printf( "\nMaximum number search using one thread...\n" );

    gettimeofday( &l_time_before, NULL );

    // Searching in single thread
    TaskPart l_single( 333, 0, l_my_length, l_my_array );
    TYPE l_res = l_single.search_max();

    gettimeofday( &l_time_after, NULL );

    printf( "The found maximum: %d\n", l_res );
    printf( "The search time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );*/
}
