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

#ifndef MRNETSYMBOLREADER_H
#define MRNETSYMBOLREADER_H

#include "SymReader.h"
#include "mrnet/MRNet.h"
#include "STAT_timer.h"
#include "STAT.h"
#include "Comm/MRNetCommFabric.h"
#include "AsyncFastGlobalFileStat.h"
//#include "MRNetSymbolReader.h"
#include "walker.h"
#include "Symtab.h"

using namespace Dyninst;
using namespace MRN;
using namespace FastGlobalFileStatus;
using namespace FastGlobalFileStatus::MountPointAttribute;
using namespace FastGlobalFileStatus::CommLayer;

extern int CUR_OUTPUT_LEVEL;
#define mrn_dbg(x, y) \
do { \
    if( MRN::CUR_OUTPUT_LEVEL >= x ){           \
        y;                                      \
    }                                           \
} while(0)

extern FILE *gStatOutFp;

class MRNetSymbolReader :public Dyninst::SymReader
{
    public:
        std::string file_;
        const char *buffer_;
        unsigned long size_;
        SymReader *symReaderHandle_;
        int refCount_;

        MRNetSymbolReader(std::string file,
                          const char* buffer,
                          unsigned long size,
                          SymReader* symReaderHandle)
                           : file_(file), buffer_(buffer), size_(size),
                             symReaderHandle_(symReaderHandle) {}
        virtual ~MRNetSymbolReader() {}
        virtual Symbol_t getSymbolByName(std::string symname);
        virtual Symbol_t getContainingSymbol(Dyninst::Offset offset);
        virtual std::string getInterpreterName();
        virtual unsigned getAddressWidth();
#if (SW_MAJOR == 9 && SW_MINOR == 3) || SW_MAJOR >= 10
        virtual bool getABIVersion(int &major, int &minor) const;
        virtual bool isBigEndianDataEncoding() const;
        virtual Architecture getArchitecture() const;
#endif
#ifdef SW_VERSION_8_1_0
        virtual unsigned numSegments();
        virtual bool getSegment(unsigned num, SymSegment &reg);
        virtual Dyninst::Offset getSymbolTOC(const Symbol_t &sym);
#else
        virtual unsigned numRegions();
        virtual bool getRegion(unsigned num, SymRegion &reg);
#endif
        virtual Dyninst::Offset getSymbolOffset(const Symbol_t &sym);
        virtual std::string getSymbolName(const Symbol_t &sym);
        virtual std::string getDemangledName(const Symbol_t &sym);
        virtual unsigned long getSymbolSize(const Symbol_t &sym);
        virtual bool isValidSymbol(const Symbol_t &sym);
        virtual Section_t getSectionByName(std::string name);
        virtual Section_t getSectionByAddress(Dyninst::Address addr);
        virtual Dyninst::Address getSectionAddress(Section_t sec);
        virtual std::string getSectionName(Section_t sec);
        virtual bool isValidSection(Section_t sec);
        virtual void *getElfHandle();
        virtual Dyninst::Offset imageOffset();
        virtual Dyninst::Offset dataOffset();
};

class MRNetSymbolReaderFactory : public Dyninst::SymbolReaderFactory
{
    public:
        MRNetSymbolReaderFactory(SymbolReaderFactory *symbolReaderFactoryHandle,
                            Network *network,
                            Stream *stream)
                             : symbolReaderFactoryHandle_(symbolReaderFactoryHandle),
                               network_(network), stream_(stream) {}
        virtual ~MRNetSymbolReaderFactory() {}
        virtual SymReader *openSymbolReader(std::string pathName);
        virtual SymReader *openSymbolReader(const char *buffer, unsigned long size);
        virtual bool closeSymbolReader(SymReader *sr);

    private:
        SymbolReaderFactory *symbolReaderFactoryHandle_;
        Network *network_;
        Stream *stream_;
        std::map<std::string, MRNetSymbolReader *> openReaders_;
        std::string perfFile_;
        virtual MRNetSymbolReader *openSymReader(const char *bufferm,
                                                 unsigned long size,
                                                 std::string pathName="");
};

#endif
