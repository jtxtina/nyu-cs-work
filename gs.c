#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/*** Skeleton for Lab 1 ***/

/***** Globals ******/
float **a; /* The coefficients */
float *x;  /* The unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */
bool error = false; /*whether to stop*/


/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */

/********************************/



/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/* 
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix()
{
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;
  
  for(i = 0; i < num; i++)
  {
    sum = 0;
    aii = fabs(a[i][i]);
    
    for(j = 0; j < num; j++)
       if( j != i)
	 sum += fabs(a[i][j]);
       
    if( aii < sum)
    {
      printf("The matrix will not converge.\n");
      exit(1);
    }
    
    if(aii > sum)
      bigger++;
    
  }
  
  if( !bigger )
  {
     printf("The matrix will not converge\n");
     exit(1);
  }
}


/******************************************************/
/* Read input from file */
/* After this function returns:
 * a[][] will be filled with coefficients and you can access them using a[i][j] for element (i,j)
 * x[] will contain the initial values of x
 * b[] will contain the constants (i.e. the right-hand-side of the equations
 * num will have number of variables
 * err will have the absolute error that you need to reach
 */
void get_input(char filename[])
{
  FILE * fp;
  int i,j;  
 
  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

 fscanf(fp,"%d ",&num);
 fscanf(fp,"%f ",&err);

 /* Now, time to allocate the matrices and vectors */
 a = (float**)malloc(num * sizeof(float*));
 if( !a)
  {
	printf("Cannot allocate a!\n");
	exit(1);
  }

 for(i = 0; i < num; i++) 
  {
    a[i] = (float *)malloc(num * sizeof(float)); 
    if( !a[i])
  	{
		printf("Cannot allocate a[%d]!\n",i);
		exit(1);
  	}
  }
 
 x = (float *) malloc(num * sizeof(float));
 if( !x)
  {
	printf("Cannot allocate x!\n");
	exit(1);
  }


 b = (float *) malloc(num * sizeof(float));
 if( !b)
  {
	printf("Cannot allocate b!\n");
	exit(1);
  }

 /* Now .. Filling the blanks */ 

 /* The initial values of Xs */
 for(i = 0; i < num; i++)
	fscanf(fp,"%f ", &x[i]);
 
 for(i = 0; i < num; i++)
 {
   for(j = 0; j < num; j++)
     fscanf(fp,"%f ",&a[i][j]);
   
   /* reading the b element */
   fscanf(fp,"%f ",&b[i]);
 }
 
 fclose(fp); 

}

/************************************************************/
/* Read matrix a and scatter to processes */
void Read_vector(
	float* linear,      	/*in*/
	float* local,		/*out*/
	int local_n,		/*in*/
	int my_rank,		/*in*/
	MPI_Comm comm       /*in*/) {


	if(my_rank == 0) {
		MPI_Scatter(linear, local_n, MPI_FLOAT, local, local_n,
			MPI_FLOAT, 0, comm);
	} else{
		MPI_Scatter(linear, local_n, MPI_FLOAT, local, local_n,
			MPI_FLOAT, 0, comm);
	}

}

/************************************************************/
/* Calculate Error Rate and judge */
bool Calculate_Error_Rate(
		float* y,
		float* x ) {
	for(int i = 0; i < num; i++) {
		float errorrate = (y[i] - x[i])/y[i];
		if(errorrate > err) {
			return false;
		}
	}
	return true;
}


/************************************************************/


int main(int argc, char *argv[])
{

 int i;
 int nit = 0; /* number of iterations */
 FILE * fp;
 char output[100] ="";
  
 if( argc != 2)
 {
   printf("Usage: ./gsref filename\n");
   exit(1);
 }
  
 /* Read the input file and fill the global data structure above */ 
 get_input(argv[1]);
 
 /* Check for convergence condition */
 /* This function will exit the program if the coffeicient will never converge to 
  * the needed absolute error. 
  * This is not expected to happen for this programming assignment.
  */
 check_matrix();

 /* Parallel program */
 int my_rank, comm_sz, local_n_a, local_n_b;
 float* local_a, local_b, local_y, y, linearA;


 MPI_Init(NULL,NULL);
 MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
 MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);


 local_n_a = num * num / comm_sz;
 local_n_b = num / comm_sz;

 local_a = (float *) malloc(local_n_a * sizeof(float));
 local_b = (float *) malloc(local_n_b * sizeof(float));
 local_y = (float *) malloc(local_n_b * sizeof(float));
 y = (float *) malloc(num * sizeof(float));
 if(my_rank == 0){
 	linearA = (float*) malloc(num * num * sizeof(float));
 	for(int i=0; i<num; i++) {
 		for(int j=0; j<num; j++) {
 			linearA[i*num+j] = a[i][j];
 		}
 		local_a[i] = 0.0;
 		local_b[i] = 0.0;
 		local_y[i] = 0.0;
 		y = 0.0;
 	}
 }

 Read_vector(linearA, local_a, local_n_a, MPI_COMM_WORLD);
 Read_vector(b, local_b, local_n_b, MPI_COMM_WORLD);

 MPI_Bcast(x, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
 while(error == false) {
 	nit++;
 	for(int i = 0; i < local_n_a/num ; i++) {
 		local_y[i] == 0.0;
 		for(int j = 0; j < num; j++) {
 			local_y[i] += local_a[i*num+j]*x[j];
 		}
 		local_y[i] -= local_a[i*num+i]*x[i];
 		local_y[i] = (local_b[i] - local_y[i])/local_a[i*num+i];
 	}
 	MPI_Allgather(local_y, local_n_b, MPI_FLOAT, y, local_n_b, MPI_FLOAT, MPI_COMM_WORLD);
 	if(my_rank == 0) {error = Calculate_Error_Rate(y, x);}
 	for(int i=my_rank*local_n_b; i<(my_rank+1)*local_n_b;i++ ) {
 		x[i] = y[i];
 		y[i] = 0.0;
 	}
 }




 MPI_Finalize();

 /* Writing results to file */
 sprintf(output,"%d.sol",num);
 fp = fopen(output,"w");
 if(!fp)
 {
   printf("Cannot create the file %s\n", output);
   exit(1);
 }
    
 for( i = 0; i < num; i++)
   fprintf(fp,"%f\n",x[i]);
 
 printf("total number of iterations: %d\n", nit);
 
 fclose(fp);


 
 exit(0);

}
