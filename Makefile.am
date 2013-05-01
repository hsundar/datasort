AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS  = -I m4

bin_PROGRAMS     = testnew
testnew_SOURCES     = sortio.h
testnew_SOURCES    += test.cpp sortio.cpp read_data.cpp io_read_tasks.cpp io_xfer_tasks.cpp
testnew_SOURCES    += xfer_recv_tasks.cpp sort_tasks.cpp 
testnew_SOURCES    += $(PARSORT_PREFIX)/src/par/parUtils.C $(PARSORT_PREFIX)/src/binOps/binUtils.C
testnew_SOURCES    += $(PARSORT_PREFIX)/src/par/sort_profiler.C

AM_CPPFLAGS = $(GRVY_CFLAGS) $(OPENMP_CXXFLAGS) $(BOOST_CPPFLAGS)
#INCLUDES = 
LIBS     = $(GRVY_LIBS) $(OPENMP_CXXFLAGS)
