/*
Copyright (c) 2007-2018, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Written by Gregory Lee [lee218@llnl.gov], Dorian Arnold, Matthew LeGendre, Dong Ahn, Bronis de Supinski, Barton Miller, Martin Schulz, Niklas Nielson, Nicklas Bo Jensen, Jesper Nielson, and Sven Karlsson.
LLNL-CODE-750488.
All rights reserved.

This file is part of STAT. For details, see http://www.github.com/LLNL/STAT. Please also read STAT/LICENSE.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

        Redistributions of source code must retain the above copyright notice, this list of conditions and the disclaimer below.
        Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the disclaimer (as noted below) in the documentation and/or other materials provided with the distribution.
        Neither the name of the LLNS/LLNL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

#ifdef USEMPI
#include "mpi.h"

char hostname[256];
int sleeptime = -1;
void do_SendOrStall(int to, int tag, int rank, int* buf, MPI_Request* req, int n)
{
    int i;
    if (rank == 1)
    {
        if (sleeptime == -1)
        {
            printf("%s, MPI task %d of %d stalling\n", hostname, rank, n);
            fflush(stdout);
            while(1) ;
        }
        else
        {
            for (i = sleeptime/10; i > 0; i = i - 1)
            {
                printf("%s, MPI task %d of %d stalling for %d of %d seconds\n", hostname, rank, n, i*10, sleeptime);
                fflush(stdout);
                sleep(10);
            }
            printf("%s, MPI task %d of %d proceeding\n", hostname, rank, n);
            fflush(stdout);
        }
    }

    MPI_Isend(buf, 1, MPI_INT, to, tag, MPI_COMM_WORLD, req);
}

void do_Receive(int from, int tag, int* buf, MPI_Request* req)
{
    MPI_Irecv(buf, 1, MPI_INT, from, tag, MPI_COMM_WORLD, req);
}
#endif

#define N (32*10)
#define THREADS_PER_BLOCK 32

__device__ void foo(char *a, int *b)
{
   int i, x, y;
#ifdef CRASH
   assert(0);
#endif
#ifdef NOHANG
   for (i = 0; i <= 100000000; i++)
#else
   for (i = 0; i >= 0; i++)
#endif
   {
	a[threadIdx.x] += b[threadIdx.x];
    x = i;
    y = x + 1;
   }
	a[threadIdx.x] += b[threadIdx.x];

}

__device__ void bar(char *a, int *b)
{
    foo(a, b);
}

__global__
void add(int *a, int *b, char *c, int *d)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x, *z;
    if (i<N)
    {
        b[i] = 2*a[i];
#ifdef CRASH
        z[i] = b[i];
        b[i] = z[i];
#endif
    }
    if (threadIdx.x % 32 == 0)
        bar(c, d);
    else
        bar(c, d);
#ifdef CRASH
    free(z);
    free(z);
    free(a);
    free(b);
#endif
}

__global__
void add2(int *a, int *b, char *c, int *d)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<N)
        b[i] = 2*a[i];
    bar(c, d);
}

__global__
void hello(char *a, int *b)
{
	a[threadIdx.x] += b[threadIdx.x];
}

int main(int argc, char **argv)
{
    int ha[N], hb[N], hc[N];
    gethostname(hostname, 256);
    	char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
const int blocksize = 16;

	char *ad;
	int *bd;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);
	printf("%s", a);


#ifdef USEMPI
    int next, prev, buf[2], tag=2;
    MPI_Request reqs[2];
    MPI_Status stats[2];
    int numtasks, rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Hello world from %s %d/%d\n", hostname, rank, numtasks);
#else
    printf("Hello serial world from %s\n", hostname);
#endif
    fflush(stdout);

    	cudaMalloc( (void**)&ad, csize );
	cudaMalloc( (void**)&bd, isize );
	cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice );
	cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );
	
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
    int *da, *db;
    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));
    int *dc, *dd;
    cudaMalloc((void **)&dc, N*sizeof(int));
    cudaMalloc((void **)&dd, N*sizeof(int));
    for (int i = 0; i<N; ++i)
        ha[i] = i;
    cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, ha, N*sizeof(int), cudaMemcpyHostToDevice);
#ifdef USEMPI
    if (rank % 2 == 0)
#endif
#ifdef HOSTCRASH
    {
        free(da);
        free(da);
        free(0);
    }
#endif
    printf("sleeping\n");
    for (int ii=0;ii<99999999;ii++)
    for (int jj=0;jj<99999999;jj++)
    	hello<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
    add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(da, db, ad, bd);
    add2<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dc, dd, ad, bd);
    printf("sleeping done\n");
	cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
	printf("%s\n", a);
	cudaFree( ad );
	cudaFree( bd );
#ifdef USEMPI

    if (argc > 1)
        sleeptime = atoi(argv[1]);
    prev = rank-1;
    next = rank+1;
    if (rank == 0)
        prev = numtasks - 1;
    if (rank == (numtasks - 1))
        next = 0;

    do_Receive(prev, tag, &buf[0], &reqs[0]);
    do_SendOrStall(next, tag, rank, &buf[1], &reqs[1], numtasks);
    MPI_Waitall(2, reqs, stats);

    MPI_Barrier(MPI_COMM_WORLD);
#endif
    cudaMemcpy(hc, dd, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i<N; ++i)
      if (i == 99)
        printf("%d\n", hb[i], hc[i]);
    fflush(stdout);
    cudaFree(da);
    cudaFree(db);
#ifdef USEMPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif
    return 0;
}
