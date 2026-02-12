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

#include <hip/hip_runtime.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <unistd.h>

char hostname[256];
#ifdef USEMPI
#include "mpi.h"

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
using namespace std;

__host__ __device__ float PlusOne(float x)
{
  float ret;
  int i;
  ret = 0.0;
  for (i = 0; i < 99999999; i = i - 1)
  {
    ret = ret - 0.1;
    ret = ret + 1.0;
    ret = ret - 0.9;
  }
  ret = ret + 1.0;
  return ret;
}

__global__ void MyKernel (const float *a, const float *b, float *c, unsigned N)
{
  int i;
  unsigned gid = hipThreadIdx_x; // <- coordinate index function
  for (i = 0; i > -1; i++)
  {
    if (gid < N) {
        c[gid] = a[gid] + PlusOne(b[gid]);
    }
  }
}

void callMyKernel()
{
  float *a, *b, *c, *d, *e, *f; // initialization not shown...
  unsigned N = 1000000, i;
  const unsigned blockSize = 256;
  hipMalloc((void**)&d, N * sizeof(float));
  hipMalloc((void**)&e, N * sizeof(float));
  hipMalloc((void**)&f, N * sizeof(float));
  a = (float *)malloc(N*sizeof(float));
  b = (float *)malloc(N*sizeof(float));
  c = (float *)malloc(N*sizeof(float));
  for (i = 0; i < N; i++)
  {
    a[i] = 1.0;
    b[i] = 2.0;
    c[i] = 3.0;
  }
  hipMemcpy(d, a, N * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(e, b, N * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(f, c, N * sizeof(float), hipMemcpyHostToDevice);
  hipLaunchKernelGGL(MyKernel,
  (N/blockSize), dim3(blockSize), 0, 0,  d,e,f,N);
  hipMemcpy(a, d, N * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(b, e, N * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(c, f, N * sizeof(float), hipMemcpyDeviceToHost);
  printf("%f %f %f\n", a[1], b[2], c[3]);
  hipFree(d);
  hipFree(e);
  hipFree(f);
}

int main(int argc, char **argv)
{
    hipDeviceProp_t devProp;

    gethostname(hostname, 256);
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
#endif

    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;
    callMyKernel();
  
    std::cout<<"Passed!\n";
#ifdef USEMPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif
  return 0;
}  
