#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>

/******Globals********/
int N, t;
int * genPrimes; //genPrimes[i] = 0/1 to judge. i+1 represents the integer.



void parallel (int thread_count) {
	// primeSize = 0;
	 int phase,i;

#	pragma omp parallel num_threads(thread_count) \ 
		default(none) shared(genPrimes, N) private(i, phase) 
	
	for(phase=0; (phase+2)<=((N+1)/2); phase++) {
		if(genPrimes[phase]==1) {
			i = 2;
#			pragma omp for nowait 
			for(i=2*phase+2; i<=N-2; i=i+phase+2) {
				genPrimes[i] = 0;
			}

		}
	} 

} 


int main(int argc, char * argv[]) {
	if(argc < 3) {
		printf("Usage: ./genprimes N t\n");
		return 0;
	}

	double tstart = 0.0, tend=0.0, ttaken;


	/********Read Input From Command Line and Initialize********/
	N = atoi(argv[1]);
	t = atoi(argv[2]);
	int floor = (N + 1) / 2;
	genPrimes = (int *) malloc((N-1) * sizeof(int));

	for(int i=0; i<N-1; i++) {	  //If number is prime, genPrimes will be 1.
		genPrimes[i] = 1;         //Initialize genPrimes.
	}


	tstart = omp_get_wtime();

		parallel(t);
		int thread_rank = omp_get_thread_num();
		if(thread_rank == 0) {
			ttaken = omp_get_wtime() - tstart;
			printf("Time take for the main part: %f\n", ttaken);
		}

	



	/* Writing results to file */
	char output[100] ="";
	sprintf(output,"%d.txt",N);
	FILE * fp = fopen(output,"w");
	if(!fp)
	{
		printf("Cannot create the file %s\n", output);
		exit(1);
	}

	fprintf(fp,"%d, %d, %d\n",1, 2, 0);
	int lastprime = 2;
	int k = 2;
	for( int i = 1; i < N-1; i++){
		if(genPrimes[i]==1) {
			int prime = i+2;
			fprintf(fp,"%d, %d, %d\n",k, prime, prime-lastprime);
			lastprime = prime;
			k++;
		}
	}
	fclose(fp);


}
