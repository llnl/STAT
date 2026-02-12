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


/**
 * Add some complexity to get rid of the warnings
 * of PACKAGE_* redefinitions when MRNet defines
 * conflicting names with STAT.
 **/

/* Include config.h and back-up its PACKAGE defines */
#include "config.h"
#if defined(PACKAGE_BUGREPORT)
  #define STAT_PACKAGE_BUGREPORT PACKAGE_BUGREPORT
  #undef PACKAGE_BUGREPORT
#endif
#if defined(PACKAGE_NAME)
  #define STAT_PACKAGE_NAME PACKAGE_NAME
  #undef PACKAGE_NAME
#endif
#if defined(PACKAGE_STRING)
  #define STAT_PACKAGE_STRING PACKAGE_STRING
  #undef PACKAGE_STRING
#endif
#if defined(PACKAGE_TARNAME)
  #define STAT_PACKAGE_TARNAME PACKAGE_TARNAME
  #undef PACKAGE_TARNAME
#endif
#if defined(PACKAGE_VERSION)
  #define STAT_PACKAGE_VERSION PACKAGE_VERSION
  #undef PACKAGE_VERSION
#endif

/* Include MRNet */
#include "mrnet/MRNet.h"
#include "mrnet/Types.h"
#include "mrnet/Packet.h"

/* Drop MRNet package includes and include xplat */
#if defined(PACKAGE_NAME)
  #undef PACKAGE_NAME
#endif
#if defined(PACKAGE_STRING)
  #undef PACKAGE_STRING
#endif
#if defined(PACKAGE_TARNAME)
  #undef PACKAGE_TARNAME
#endif
#if defined(PACKAGE_VERSION)
  #undef PACKAGE_VERSION
#endif
#include "xplat/NetUtils.h"

/* Remove MRNet's Package defines */
#if defined(PACKAGE_BUGREPORT)
  #undef PACKAGE_BUGREPORT
#endif
#if defined(PACKAGE_NAME)
  #undef PACKAGE_NAME
#endif
#if defined(PACKAGE_STRING)
  #undef PACKAGE_STRING
#endif
#if defined(PACKAGE_TARNAME)
  #undef PACKAGE_TARNAME
#endif
#if defined(PACKAGE_VERSION)
  #undef PACKAGE_VERSION
#endif

/* Restore STAT's package defines*/
#if defined(STAT_PACKAGE_BUGREPORT)
  #define PACKAGE_BUGREPORT STAT_PACKAGE_BUGREPORT
#endif
#if defined(STAT_PACKAGE_NAME)
  #define PACKAGE_NAME STAT_PACKAGE_NAME
#endif
#if defined(STAT_PACKAGE_STRING)
  #define PACKAGE_STRING STAT_PACKAGE_STRING
#endif
#if defined(STAT_PACKAGE_TARNAME)
  #define PACKAGE_TARNAME STAT_PACKAGE_TARNAME
#endif
#if defined(STAT_PACKAGE_VERSION)
  #define PACKAGE_VERSION STAT_PACKAGE_VERSION
#endif

