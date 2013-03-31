AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS  = -I m4

bin_PROGRAMS     = test
test_SOURCES     = sortio.h
test_SOURCES    += test.cpp sortio.cpp read_data.cpp io_read_tasks.cpp io_xfer_tasks.cpp
test_SOURCES    += xfer_recv_tasks.cpp

INCLUDES = $(GRVY_CFLAGS) $(OPENMP_CXXFLAGS)
LIBS     = $(GRVY_LIBS) $(OPENMP_CXXFLAGS)
