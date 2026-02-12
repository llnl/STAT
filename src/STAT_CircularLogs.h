/*
Copyright (c) 2007-2020, Lawrence Livermore National Security, LLC.
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


#if !defined(STAT_CircularLogs_)
#define STAT_CircularLogs_

#include <boost/circular_buffer.hpp>

class CircularBuffer
{
    public:
        CircularBuffer(size_t size);
        ~CircularBuffer();

        /* Return a write FILE handle for adding to the buffer */
        FILE *handle();

        /* Return the buffer in two parts. */
        bool getBuffer(char* &buffer1, size_t &buffer1Size, char* &buffer2, size_t &buffer2Size);

        /* Write the buffer to the given file descriptor */
        int flushBufferTo(int fd);

        /* Convert the buffer to a string and return it.  Adds a null character to end of buffer */
        const char *str();

        /* Clear the buffer */
        void reset();

    private:
        typedef boost::circular_buffer<char> Buffer_t;

        static ssize_t writeWrapper(void *cookie, const char *buf, size_t size);
        static int closeWrapper(void *cookie);
        ssize_t cWrite(const char *buf, size_t size);
        int cClose();
        void init();

        Buffer_t *buffer_;
        FILE *fHandle_;
        size_t bufferSize_;
};

#endif
