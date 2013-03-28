AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS  = -I m4

bin_PROGRAMS     = test
test_SOURCES     = test.cpp sortio.cpp read_data.cpp sortio.h 

INCLUDES = $(GRVY_CFLAGS) $(OPENMP_CXXFLAGS)
LIBS     = $(GRVY_LIBS) $(OPENMP_CXXFLAGS)
