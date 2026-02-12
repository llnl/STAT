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

#include "MRNetSymbolReader.h"

SymReader *MRNetSymbolReaderFactory::openSymbolReader(std::string pathName)
{
    const char *pathStr = pathName.c_str();
    bool localLib = true;
    int tag = PROT_LIB_REQ, ret;
    long size;
    uint64_t fileContentsLength = 0;
    char *fileName = NULL, *fileContents = NULL;
    FILE *fp;
    PacketPtr packet;
    MRNetSymbolReader *msr;
    std::map<std::string, MRNetSymbolReader* >::iterator iter;

////GLL comment: This is a temp feature to look for a copy in /tmp RAM disk first
//std::string fileBaseName = basename(pathName.c_str());
//std::string tmpFilePath = "/tmp/" + fileBaseName;
//struct stat fileStat;
//if (stat(tmpFilePath.c_str(), &fileStat) == 0)
//{
//fprintf(stderr, "%s %s\n", pathName.c_str(), tmpFilePath.c_str());
//pathName = tmpFilePath;
//}

    mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader", gStatOutFp,
            "Interposed lib functions called openSymbolReader(%s)\n",
            pathName.c_str()));

    iter = openReaders_.find(pathName);
    if (iter == openReaders_.end())
    {
        mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader", gStatOutFp,
                "no existing reader for %s\n", pathStr));

        AsyncGlobalFileStatus myStat(pathStr);
/* TODO: this is a workaround for BlueGene where FGFS is reporting incorrectly */
#ifdef BGL
        if (true)
#else
        if (IS_YES(myStat.isUnique()))
#endif
        {
            localLib = false;

            mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader",
                    gStatOutFp, "requesting contents for %s\n", pathStr));

            if (stream_->send(tag, "%s", pathStr) == -1)
            {
                mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader",
                        gStatOutFp, "BE: stream::send() failure\n"));
                return NULL;
            }
            if (stream_->flush() == -1)
            {
                mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader",
                        gStatOutFp, "BE: stream::flush() failure\n"));
                return NULL;
            }

            //ret = network_->recv(&tag, packet, &stream_);
            ret = stream_->recv(&tag, packet);
            if (ret != 1)
            {
                mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader",
                        gStatOutFp, "BE: network::recv() failure\n"));
                return NULL;
            }

            if (tag == PROT_LIB_REQ_ERR)
            {
                mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader",
                        gStatOutFp, "FE reported error sending contents of %s\n", pathStr));
                localLib = true;
            }
            if (tag != PROT_LIB_REQ_RESP)
            {
                mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader",
                        gStatOutFp, "Unexpected tag %d when trying to receive contents of %s\n", tag, pathStr));
                localLib = true;
            }
            else if (packet->unpack("%Ac %s", &fileContents, &fileContentsLength, &fileName) == -1)
            {
                mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader",
                        gStatOutFp, "Failed to unpack contents of %s, length %d\n", pathStr, fileContentsLength));
                localLib = true;
            }
            free(fileName);
            fileName = NULL;
        }

        if (localLib == true)
        {
            mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader",
                    gStatOutFp, "Trying to read %s locally\n", pathStr));
            fp = fopen(pathStr, "r");
            if (fp == NULL)
            {
                mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader",
                        gStatOutFp, "File %s does not exist on BE\n", pathStr));
                return NULL;
            }

            fseek(fp, 0, SEEK_END);
            size = ftell(fp);
            if (size == -1)
            {
                mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader",
                        gStatOutFp, "%s: File %s ftell returned -1\n", strerror(errno), pathStr));
                fclose(fp);
                return NULL;
            }
            fseek(fp, 0, SEEK_SET);
            fileContents = (char *)malloc(size * sizeof(char));
            if (fileContents == NULL)
            {
                mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader",
                        gStatOutFp, "Malloc returned NULL for %s\n", pathStr));
                fclose(fp);
                return NULL;
            }
            fread(fileContents, size, 1, fp);
            fclose(fp);

            msr = openSymReader((char *)fileContents, size, pathName);
            if (msr == NULL)
            {
                mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader",
                        gStatOutFp, "openSymReader returned NULL for %s %d\n",
                        pathName.c_str(), size));
                free(fileContents);
                return NULL;
            }
        }
        else
        {
            mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader",
                    gStatOutFp, "reading contents of %s\n", pathStr));
            msr = openSymReader((char *)fileContents, fileContentsLength,
                                pathName);
            if (msr == NULL)
            {
                mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader",
                        gStatOutFp, "openSymReader returned NULL for %s %d\n",
                        pathName.c_str(), fileContentsLength));
                return NULL;
            }
        }
        openReaders_[pathName] = msr;
        msr->refCount_ = 1;
    }
    else
    {
        mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader", gStatOutFp,
                "Found existing reader for %s\n", pathStr));
        msr = iter->second;
        msr->refCount_++;
    }
    mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymbolReader", gStatOutFp, "done\n"));

    return static_cast<SymReader *>(msr);
}

MRNetSymbolReader *
MRNetSymbolReaderFactory::openSymReader(const char *buffer,
                                        unsigned long size,
                                        std::string file)
{
    SymReader *handle = symbolReaderFactoryHandle_->openSymbolReader(buffer,
                                                                     size);
    if (handle == NULL)
    {
        mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymReader", gStatOutFp,
                "openSymbolReader returned NULL for %s %d\n", file.c_str(),
                size));
        return NULL;
    }
    MRNetSymbolReader *msr = new MRNetSymbolReader(file, buffer, size, handle);
    return msr;
}

SymReader *
MRNetSymbolReaderFactory::openSymbolReader(const char *buffer, unsigned long size)
{
    MRNetSymbolReader *msr = openSymReader(buffer,size);
    if (msr == NULL)
    {
        mrn_dbg(2, mrn_printf(__FILE__, __LINE__, "openSymReader", gStatOutFp,
                "openSymbolReader returned NULL for %d\n", size));
        return NULL;
    }
    msr->refCount_ = 1;
    return static_cast<SymReader *>(msr);
}

bool MRNetSymbolReaderFactory::closeSymbolReader(SymReader *sr)
{
    MRNetSymbolReader *msr = static_cast<MRNetSymbolReader *>(sr);
    msr->refCount_--;
    std::map<std::string, MRNetSymbolReader *>::iterator iter = openReaders_.find(msr->file_);
    if ((iter != openReaders_.end()) && (msr->refCount_ == 0))
        openReaders_.erase(iter);

    delete msr;
    return true;
}

inline Symbol_t MRNetSymbolReader::getSymbolByName(std::string symname)
{
    return symReaderHandle_->getSymbolByName(symname);
}

inline Symbol_t MRNetSymbolReader::getContainingSymbol(Dyninst::Offset offset)
{
    return symReaderHandle_->getContainingSymbol(offset);
}

inline std::string  MRNetSymbolReader::getInterpreterName()
{
    return symReaderHandle_->getInterpreterName();
}

inline unsigned MRNetSymbolReader::getAddressWidth()
{
    return symReaderHandle_->getAddressWidth();
}

#if (SW_MAJOR == 9 && SW_MINOR == 3) || SW_MAJOR >= 10
inline bool MRNetSymbolReader::getABIVersion(int &major, int &minor) const
{
    return symReaderHandle_->getABIVersion(major, minor);
}

inline bool MRNetSymbolReader::isBigEndianDataEncoding() const
{
    return symReaderHandle_->isBigEndianDataEncoding();
}

inline Architecture MRNetSymbolReader::getArchitecture() const
{
    return symReaderHandle_->getArchitecture();
}
#endif


#ifdef SW_VERSION_8_1_0
inline bool MRNetSymbolReader::getSegment(unsigned num, SymSegment &reg)
{
    return symReaderHandle_->getSegment(num, reg);
}

inline unsigned int MRNetSymbolReader::numSegments()
{
    return symReaderHandle_->numSegments();
}

inline Dyninst::Offset MRNetSymbolReader::getSymbolTOC(const Symbol_t &sym)
{
    return symReaderHandle_->getSymbolTOC(sym);
}

#else
inline unsigned MRNetSymbolReader::numRegions()
{
    return symReaderHandle_->numRegions();
}

inline bool MRNetSymbolReader::getRegion(unsigned num, SymRegion &reg)
{
    return symReaderHandle_->getRegion(num, reg);
}
#endif /* SW_VERSION_8_1_0 */

inline Dyninst::Offset MRNetSymbolReader::getSymbolOffset(const Symbol_t &sym)
{
    return symReaderHandle_->getSymbolOffset(sym);
}

inline std::string MRNetSymbolReader::getSymbolName(const Symbol_t &sym)
{
    return symReaderHandle_->getSymbolName(sym);
}

inline std::string MRNetSymbolReader::getDemangledName(const Symbol_t &sym)
{
    return symReaderHandle_->getDemangledName(sym);
}

inline  unsigned long MRNetSymbolReader::getSymbolSize(const Symbol_t &sym)
{
    return symReaderHandle_->getSymbolSize(sym);
}

inline bool MRNetSymbolReader::isValidSymbol(const Symbol_t &sym)
{
    return symReaderHandle_->isValidSymbol(sym);
}

inline Section_t MRNetSymbolReader::getSectionByName(std::string name)
{
    return symReaderHandle_->getSectionByName(name);
}

inline Section_t MRNetSymbolReader::getSectionByAddress(Dyninst::Address addr)
{
    return symReaderHandle_->getSectionByAddress(addr);
}

inline Dyninst::Address MRNetSymbolReader::getSectionAddress(Section_t sec)
{
    return symReaderHandle_->getSectionAddress(sec);
}

inline std::string MRNetSymbolReader::getSectionName(Section_t sec)
{
    return symReaderHandle_->getSectionName(sec);
}

inline bool MRNetSymbolReader::isValidSection(Section_t sec)
{
    return symReaderHandle_->isValidSection(sec);
}

inline void *MRNetSymbolReader::getElfHandle()
{
    return symReaderHandle_->getElfHandle();
}

inline Dyninst::Offset MRNetSymbolReader::imageOffset()
{
    return symReaderHandle_->imageOffset();
}

inline Dyninst::Offset MRNetSymbolReader::dataOffset()
{
    return symReaderHandle_->dataOffset();
}

