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

#define TYPE int

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
};

// Thread will search the largest element in array 
// from element arg->m_from with length of arg->m_length.
// Result will be stored to arg->m_max.
void *my_thread( void *t_void_arg )
{
    TaskPart *lp_task = ( TaskPart * ) t_void_arg;

    printf( "Thread %d started from %d with length %d...\n",
        lp_task->m_id, lp_task->m_from, lp_task->m_length );

    lp_task->m_max = lp_task->search_max();

    printf( "Found maximum in thread %d is %d\n", lp_task->m_id, lp_task->m_max );

    return NULL;
}

// Time interval between two measurements converted to ms
int timeval_diff_to_ms( timeval *t_before, timeval *t_after )
{
    timeval l_res;
    timersub( t_after, t_before, &l_res );
    return 1000 * l_res.tv_sec + l_res.tv_usec / 1000;
}

#define LENGTH_LIMIT 10000000

int main( int t_na, char **t_arg )
{
    // The number of elements must be used as program argument
    if ( t_na != 2 ) 
    { 
        printf( "Specify number of elements, at least %d.\n", LENGTH_LIMIT ); 
        return 0; 
    }
    int l_my_length = atoi( t_arg[ 1 ] );
    if ( l_my_length < LENGTH_LIMIT ) 
    { 
        printf( "The number of elements must be at least %d.\n", LENGTH_LIMIT ); 
        return 0; 
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

    printf( "Random numbers generetion started..." );
    for ( int i = 0; i < l_my_length; i++ )
    {
        l_my_array[ i ] = rand() % ( l_my_length * 10 );
        if ( !( i % LENGTH_LIMIT ) ) 
        {
            printf( "." ); 
            fflush( stdout );
        }
    }

    printf( "\nMaximum number search using two threads...\n" );
    pthread_t l_pt1, l_pt2;
    TaskPart l_tp1( 1, 0, l_my_length / 2, l_my_array );
    TaskPart l_tp2( 2, l_my_length / 2, l_my_length - l_my_length / 2, l_my_array );

    timeval l_time_before, l_time_after;

    // Time recording before searching
    gettimeofday( &l_time_before, NULL );

    // Threads starting
    pthread_create( &l_pt1, NULL, my_thread, &l_tp1 );
    pthread_create( &l_pt2, NULL, my_thread, &l_tp2 );

    // Waiting for threads completion 
    pthread_join( l_pt1, NULL );
    pthread_join( l_pt2, NULL );

    // Time recording after searching
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
    printf( "The search time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );
}
