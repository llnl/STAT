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

#include "STAT_CircularLogs.h"

#if !defined(_GNU_SOURCE)
  #define _GNU_SOURCE
#endif

#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>

CircularBuffer::CircularBuffer(size_t size) :
    buffer_(NULL),
    fHandle_(NULL),
    bufferSize_(size)
{
    cookie_io_functions_t funcs;
    funcs.read = NULL;
    funcs.write = writeWrapper;
    funcs.seek = NULL;
    funcs.close = closeWrapper;

    fHandle_ = fopencookie(this, "w", funcs);
    if (!fHandle_)
        return;
}

CircularBuffer::~CircularBuffer()
{
    if (fHandle_)
        fclose(fHandle_);
    if (buffer_)
        delete buffer_;
    fHandle_ = NULL;
    buffer_ = NULL;
}

void CircularBuffer::init()
{
    if (!buffer_)
    {
        buffer_ = new Buffer_t(bufferSize_);
        if (!buffer_)
            return;
    }

    if (!fHandle_)
    {
        cookie_io_functions_t funcs;
        funcs.read = NULL;
        funcs.write = writeWrapper;
        funcs.seek = NULL;
        funcs.close = closeWrapper;
        fHandle_ = fopencookie(this, "w", funcs);
    }
}

ssize_t CircularBuffer::writeWrapper(void *cookie, const char *buf, size_t size)
{
    CircularBuffer *me = static_cast<CircularBuffer *>(cookie);
    return me->cWrite(buf, size);
}

int CircularBuffer::closeWrapper(void *cookie)
{
    CircularBuffer *me = static_cast<CircularBuffer *>(cookie);
    return me->cClose();
}

ssize_t CircularBuffer::cWrite(const char *buf, size_t size)
{
    for (unsigned i = 0; i < size; i++)
        buffer_->push_back(buf[i]);
    return size;
}

int CircularBuffer::cClose()
{
    reset();
    fHandle_ = NULL;
    return 0;
}

FILE *CircularBuffer::handle()
{
    init();
    return fHandle_;
}

bool CircularBuffer::getBuffer(char* &buffer1, size_t &buffer1Size, char* &buffer2, size_t &buffer2Size)
{
    if (!buffer_)
       return false;

    buffer1 = buffer_->array_one().first;
    buffer1Size = buffer_->array_one().second;
    buffer2 = buffer_->array_two().first;
    buffer2Size = buffer_->array_two().second;

    return true;
}

int CircularBuffer::flushBufferTo(int fd)
{
    int numWritten = 0, result;
    char *buffer1 = NULL, *buffer2 = NULL;
    size_t buffer1Size = 0, buffer2Size = 0;

    getBuffer(buffer1, buffer1Size, buffer2, buffer2Size);
    if (buffer1)
    {
        result = write(fd, buffer1, buffer1Size);
        if (result == -1)
            return -1;
        numWritten += result;
    }

    if (buffer2)
    {
        result = write(fd, buffer2, buffer2Size);
        if (result == -1)
            return -1;
        numWritten += result;
    }

    return numWritten;
}

const char *CircularBuffer::str()
{
    if (!buffer_)
        return "";

    buffer_->push_back(0);
    return buffer_->linearize();
}

void CircularBuffer::reset()
{
    if (!buffer_)
        return;
    buffer_->resize(0);
}
