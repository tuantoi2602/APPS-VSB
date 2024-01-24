/*
	In my opinion, measure time is correct when using more threads and threads will split time, measure time is smaller
	when using more threads.
	Measure time is not correct when matrices multiply is wrong or we use little number, the result is 0 [ms]
	is the same and we can't compare time.
	Sometimes we need base on many expensive computer, if core is better, measure time will have too fast when we use
	more threads.
	Like matrices in expansive computer we can plus 2 - 4 -6 base on we will use threads. Example: Without threads is 10000 ms
	threads will be 5000 (2 threads) or 2500 (4 threads),...
	Or in cheap computer  we can plus 2 - 4 -6 base on we will use threads. Example: Without threads is 25000 ms
	threads will be 12500 (2 threads) or 6250 (4 threads),...
	For that many reason, the measure time when use threads or not is not too much different



	*/
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




// class for one part of task

class matrix{
public:
	int rows;
	int cols;
	float** matrix5;
	matrix(int x,int y){
		rows = x;
		cols = y;
		matrix5 = new float*[rows];
		for (int i = 0; i < rows; i++){
		    matrix5[i] = new float[cols];
		}
/*		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
			   matrix5[i][j] = (float)(((rand() % 200000) - 100000 ) / 100);
			 }
		}*/
	}

	void generate(){
		 for (int i = 0; i < rows; i++) {
		    for (int j = 0; j < cols; j++) {
		    	matrix5[i][j] = (float)(((rand() % 2000) - 1000 ));
		    }
		  }
	}

	void unit_matrix(){
		 for (int i = 0; i < rows; i++) {
		    for (int j = 0; j < cols; j++) {
		    	if(i == j){
		    		matrix5[i][j] = 1;
		    	}
		    	else{
		    		matrix5[i][j] = 0;
		    	}
		    }
		  }
	}
	void print(){
		 for (int i = 0; i < rows; i++) {
		    for (int j = 0; j < cols; j++) {
		    	printf("%.0f ",matrix5[i][j]);
		    }
		    printf("\n");
		 }
	}

};


class TaskPart
{
public:

	matrix m1;
	matrix m2;
	matrix out;
	int m1_from;
	int m1_to;

	TaskPart(matrix t_m1,matrix t_m2,matrix t_out, int t_m1from, int t_m1to ) :
		m1(t_m1), m2(t_m2),out(t_out),m1_from(t_m1from),m1_to(t_m1to){}

/*	void print(){
		 for (int i = 0; i <out.rows; i++) {
		    for (int j = 0; j < out.cols; j++) {
		    	printf("%.0f ",out.matrix5[i][j]);
		    }
		    printf("\n");
		 }
	}*/
/*	void multiply(){
		    //if ( m1.cols != m2.rows ){
		    for (int i = m1_from; i < m1_to; i++){
		        for (int j= 0; j < m2; j++ ){
		        	out[i][j] = 0;
		            for (int inx = 0; inx < m2;inx++ )
		            	out[i][j] += m1[i][inx] * m2[inx][j];
		        }
		    }
	}*/


};

void multiply(matrix m1,matrix m2,matrix out,int m1_from,int m1_to){
	if(m1.cols != m2.rows){
		printf("Bad matrix");
		return;
	}
	else{
	for(int row = m1_from; row < m1_to;row++){

		for(int col = 0;col < m2.cols;col++){
			out.matrix5[row][col] = 0;
			for(int inx = 0; inx < m2.rows; inx++){
				out.matrix5[row][col] += m1.matrix5[row][inx] * m2.matrix5[inx][col];
				}
			}
		}
	}
}
bool compareMatrix(matrix m1,matrix out){
	if(m1.cols != out.cols){
		return 0;
	}
	if(m1.rows != out.rows){
		return 0;
	}
	for(int i = 0; i <m1.rows ; i++){
		for(int j = 0; j <m1.cols;j++){
			if ( m1.matrix5[i][j] != out.matrix5[i][j]){
				return 0;
	   	   }
		}
	}
	   return 1;
}
/*void *my_thread_matrix(void *t_void_arg){
	TaskPart_matrix *t_lp_matrix = (TaskPart_matrix *) t_void_arg;
	printf( "Thread %d started...\n",
		t_lp_matrix->m_id);
	t_lp_matrix->multiply();
	return NULL;
}*/

void *my_thread( void *t_void_arg )
{
	TaskPart *lp_task = ( TaskPart * ) t_void_arg;


	multiply(lp_task->m1,lp_task->m2,lp_task->out,lp_task->m1_from,lp_task->m1_to);
	//lp_task->print();
	//lp_task->bubbleSort_desc();
	//lp_task->print_Array1();

  //  printf( "Found maximum in thread %d is %d\n", lp_task->m_id, lp_task->m_max );

	return NULL;
}

// Time interval between two measurements converted to ms
int timeval_diff_to_ms( timeval *t_before, timeval *t_after )
{
	timeval l_res;
	timersub( t_after, t_before, &l_res );
	return 1000 * l_res.tv_sec + l_res.tv_usec / 1000;
}



#define LENGTH_LIMIT 10000

int main( int t_na, char **t_arg ) {

	int m1_x = atoi(t_arg[1]);

	int m1_y = atoi(t_arg[2]);

	int m2_x = atoi(t_arg[3]);

	int m2_y = atoi(t_arg[4]);

	int n = atoi(t_arg[5]);

	 timeval l_time_before, l_time_after;


	// printf("FIRST MATRIX:\n");
	 matrix ma1(m1_x,m1_y);
	 ma1.generate();
	// ma1.print();

	 printf("\n");
	// printf("SECOND MATRIX:\n");
	 matrix ma2(m2_x,m2_y);
	 ma2.unit_matrix();
	// ma2.print();

	 printf("\n");
	 matrix ma3(m1_x,m2_y);
	 gettimeofday( &l_time_before, NULL );
	 multiply(ma1,ma2,ma3,0,m1_x);
	// ma3.print();
	 gettimeofday( &l_time_after, NULL );
	 printf( "The search time of NOT USING THREADS: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );


	 printf("\n");
	 matrix ma4(m1_x,m2_y);


	 gettimeofday( &l_time_before, NULL );



	TaskPart *l_tp = (TaskPart*) malloc(sizeof(TaskPart)*n);
	pthread_t l_pt[n];
	for(int i = 0; i < n; i++){
	l_tp[i] =TaskPart(ma1, ma2, ma4,i*m1_x/n,(i+1)*m1_x/n);
	}
	for(int i = 0; i < n; i++){
		pthread_create( &l_pt[i], NULL, my_thread, &l_tp[i] );
	}
	 for(int i = 0; i < n; i++){
		 pthread_join( l_pt[i], NULL );
	 }
	 printf("\n");
	// ma4.print();
	 gettimeofday( &l_time_after, NULL );
	 printf( "The search time of THREADS: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );


	  if(compareMatrix(ma3,ma4)==1){
		  printf("Equal\n");
	  }else{
		  printf("Not equal\n");
	  }


/*	 matrix1 ma3(m1_x,m2_y);
	 multiply(ma1,ma2,ma3,0,m1_x);
	 ma3.print();*/

/*	if(col_1 != row_2){
		printf("Bad Matrix\n");
		return 0;
	}*/






/*	int l_my_length = atoi(t_arg[1]);
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
	}*/

	 // array allocation
/*
	 TYPE *l_my_array = new TYPE [ l_my_length ];
	 if ( !l_my_array )
	 {
		 printf( "Not enough memory for array!\n" );
		 return 1;
	 }
	 srand( ( int ) time( NULL ) );
*/

	 // Initialization of random number generator


/*	 printf( "Random numbers generetion started...\n" );
	 TYPE m1[m1_x][m1_y];

	  for (int i = 0; i < m1_x; i++) {
	    for (int j = 0; j < m1_y; j++) {
	    	m1[i][j] = (float)(((rand() % 20000) - 10000 ) / 100.0);
	    }
	  }
	  for (int i = 0; i < m1_x; i++) {
	    for (int j = 0; j < m1_y; j++) {
	      printf( "%.2f ", m1[i][j] );
	   }
	    printf("\n");
	  }
	  printf("\n");
		 TYPE m2[m2_x][m2_y];

		  for (int i = 0; i < m2_x; i++) {
		    for (int j = 0; j < m2_y; j++) {
		    	m2[i][j] = (float)(((rand() % 20000) - 10000 ) / 100.0);
		    }
		  }
		  for (int i = 0; i < m2_x; i++) {
		    for (int j = 0; j < m2_y; j++) {
		      printf( "%.2f ", m2[i][j] );
		   }
		    printf("\n");
		  }*/
/*		  int m3[10][10];
		   for(int i = 0; i < m1_x; ++i)
		        for(int j = 0; j < m2_y; ++j)
		            for(int k = 0; k < m1_y; ++k)
		            {
		                m3[i][j] += a[i][k] * b[k][j];
		            }


		 if ( !( i % LENGTH_LIMIT ) )
		 {
		  //   printf( "." );
			 fflush( stdout );
		 }
		 //l_my_array[i] = LENGTH_LIMIT-i;
	 }*/

/*	 TYPE *m_data1 = new TYPE [l_my_length];
	 for(int i = 0; i < l_my_length; i++){
		 m_data1[i] = l_my_array[i];
	 }*/

	/* timeval l_time_before, l_time_after;

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
	// for(int m = 1; m < P ; m++){
	//	 mergeArrays(&l_tp[0],&l_tp[m]);
		// l_tp[0].print_Array1();

	  gettimeofday( &l_time_after, NULL );
	  printf( "The merge search time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );


	  gettimeofday( &l_time_before, NULL );
	  TaskPart l_single1(1, 0, l_my_length, m_data1 );
	 // l_single1.bubbleSort_desc();
	  gettimeofday( &l_time_after, NULL );
	  printf( "The search time of COPY: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );



	 gettimeofday( &l_time_before, NULL );
	 TaskPart l_single2(2, 0, l_my_length, m_data1 );
	// l_single2.bubbleSort_desc();
	 gettimeofday( &l_time_after, NULL );
	 printf( "The new desc again search time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );

	 gettimeofday( &l_time_before, NULL );
	 TaskPart l_single3(3, 0, l_my_length, m_data1 );
	// l_single3.bubbleSort();
	 gettimeofday( &l_time_after, NULL );
	 printf( "The new asc search time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );*/
/*pthread_t l_pt1,l_pt2;
TaskPart l_tp1(ma1, ma2, ma4,0*m1_x/n,1*m1_x/n);

pthread_create( &l_pt1, NULL, my_thread, &l_tp1 );

pthread_join( l_pt1, NULL );
ma4.print();
 printf("\n");

 TaskPart l_tp2(ma1, ma2, ma4,1*m1_x/n,2*m1_x/n);

 pthread_create( &l_pt2, NULL, my_thread, &l_tp2 );

 pthread_join( l_pt2, NULL );

 ma4.print();*/
}
