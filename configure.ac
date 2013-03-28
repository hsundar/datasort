#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.63])
AC_INIT([sort-io], [0.1], [karl@tacc.utexas.edu])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([test.cpp])
AM_INIT_AUTOMAKE
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AC_LANG([C])
ACX_MPI([CC="$MPICC"], AC_MSG_ERROR([Could not find MPI C compiler support]))
AC_LANG([C++])
ACX_MPI([HAVE_MPI=1 CXX="$MPICXX" CC="$MPICC" AC_DEFINE(HAVE_MPI,1,[Define if MPI is available])],
		    AC_MSG_ERROR([Could not find MPI C++ compiler support]))

AX_OPENMP([], AC_MSG_ERROR([Could not detect required OpenMP support]))

# Checks for libraries.

AX_PATH_GRVY([0.30],[yes])

# Checks for header files.
AC_CHECK_HEADERS([fcntl.h stdlib.h sys/ioctl.h unistd.h])

AC_OUTPUT(Makefile)
