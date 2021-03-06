//-----------------------------------------------------------------------bl-
//--------------------------------------------------------------------------
// 
// datasort - an IO/data distribution utility for large data sorts.
//
// Copyright (C) 2013 Karl W. Schulz
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the Version 2.1 GNU Lesser General
// Public License as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc. 51 Franklin Street, Fifth Floor, 
// Boston, MA  02110-1301  USA
//
//-----------------------------------------------------------------------el-

#include "sortio.h"

// --------------------------------------------------------------------
// IO_Tasks_Work(): work manager for IO tasks
// --------------------------------------------------------------------

void sortio_Class::IO_Tasks_Work()
{
  assert(initialized_);

  int thread_id = omp_get_thread_num();
  grvy_printf(INFO,"[sortio]:IO[%i]: thread id for Master thread = %i\n",ioRank_,thread_id);

  ReadFiles();

  grvy_printf(INFO,"[sortio][IO/Read][%.4i]: ALL DONE with Read\n",ioRank_);

  return;
}
