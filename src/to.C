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


#include <string>
#include "TO_std.h"
#include "TO_Program.h"
#include "TO_LexicographicalOrdering.h"
#include "to.h"

extern "C"
{

vector<string> search_paths;
vector<to_tuple> tuples;
map<int, string> line_to_lex;

char *get_lex_string(to_tuple input)
{
    //fprintf(stderr, "found %s\n", (char *)line_to_lex[input.line].c_str());
    return (char *)line_to_lex[input.line].c_str();
}

int run_to()
{
    lNoLexGraphicalStringTuple tup;
    ProgramPoints pt;
    TemporalOrderingAPI ord;
    int i;

    if (tuples.size() < 1)
        return 0;

    pt.set_basePath(tuples[0].basepath);
    pt.set_incSearchPaths(search_paths);
    pt.set_srcFileName(tuples[0].filename);

    for (i = 0; i < tuples.size(); i++)
    {
        tup.lineNumber = tuples[i].line;
    //fprintf(stderr, "run %s/%s:%d\n", tuples[i].basepath, tuples[i].filename, tuples[i].line);
        pt.get_execPoints().push_back(tup);
    }

    ord.set_pPoints(pt);
    ord.run(false);
    ord.orderProgramPoints();

    for (i = 0; i < tuples.size(); i++)
    {
        line_to_lex[ord.get_pPoints().get_execPoints()[i].lineNumber] = ord.get_pPoints().get_execPoints()[i].lexicoGraphicalStr;
    }

    return 0;
}

int add_program_point(to_tuple input)
{
    //fprintf(stderr, "add %s/%s:%d\n", input.basepath, input.filename, input.line);
    tuples.push_back(input);
    return 0;
}
int clear_program_points()
{
    tuples.clear();
    return 0;
}
char *get_temporal_string(to_tuple input)
{
    int i, j, rc;
    bool found = false;
    string ret;
    char buf[256];
    ProgramPoints pt;
    TemporalOrderingAPI ord;
    vector<lNoLexGraphicalStringTuple>::iterator iter;
    lNoLexGraphicalStringTuple *cur = NULL;

    for (i = 0; i < tuples.size(); i++)
    {
        lNoLexGraphicalStringTuple tup;
        tup.lineNumber = tuples[i].line;
        tup.lexicoGraphicalStr = tuples[i].basepath;
        pt.get_execPoints().push_back(tup);
    }
    ord.set_pPoints(pt);
    ord.orderProgramPoints();
    i = 0;
    pt = ord.get_pPoints();
    for (iter = pt.get_execPoints().begin();
        iter != pt.get_execPoints().end(); iter++)
    {
        if (cur != NULL)
        {
            rc = ord.partialOrdering(cur->lexicoGraphicalStr, (*iter).lexicoGraphicalStr);
            if (found == true)
            {
                /* See if we're "equal" to the previously found point */
                if (rc == 0)
                    ret += ".?";
                break;
            }
            if (rc != 0)
                i++;
            if ((*iter).lexicoGraphicalStr == input.basepath)
            {
//fprintf(stderr, "%s = %s\n", (*iter).lexicoGraphicalStr.c_str(), input.basepath);
                sprintf(buf, "%d", i);
                ret = buf;
                if (rc == 0)
                {
                    ret += ".?";
                    break;
                }
                found = true;
            }
        }
        else
        {
            i++;
            if ((*iter).lexicoGraphicalStr == input.basepath)
            {
//fprintf(stderr, "%s = %s\n", (*iter).lexicoGraphicalStr.c_str(), input.basepath);
                sprintf(buf, "%d", i);
                ret = buf;
                found = true;
            }
        }
        cur = &(*iter);
    }
    return (char *)(ret.c_str());
}

int add_include_path(char *path)
{
    search_paths.push_back(path);
    return 0;
}

}
