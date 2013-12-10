#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.63])
AC_INIT([sort-io], [0.20.0], [karl@tacc.utexas.edu])
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

AX_PATH_GRVY([0.30],[yes])

# Check for parsort location 

AC_ARG_VAR(PARSORT_DIR,[root directory of PARSORT installation])

AC_ARG_WITH(parsort,
  [AS_HELP_STRING([--with-parsort[=DIR]],[root directory of PARSORT installation (default = PARSORT_DIR)])],
  [with_parsort=$withval
if test "${with_parsort}" != yes; then
    PARSORT_PREFIX=$withval
fi
],[
with_parsort=yes
if test "x${PARSORT_DIR}" != "x"; then
   PARSORT_PREFIX=${PARSORT_DIR}
fi
])

echo "parsort_prefix = = $PARSORT_PREFIX"
ac_PARSORT_save_CPPFLAGS="$CPPFLAGS"
CPPFLAGS="-I${PARSORT_PREFIX}/include -I${PARSORT_PREFIX}/include/par -I${PARSORT_PREFIX}/include/binOps -I${PARSORT_PREFIX}/include/seq -I${PARSORT_PREFIX}/include/omp_par"
AC_LANG_PUSH([C++])

succeeded=no
AC_CHECK_HEADER([parUtils.h],[succeeded=yes],[succeeded=no])

if test "$succeeded" = no; then
    AC_MSG_ERROR([parSort headers not found.  Try either --with-parsort or setting PARSORT_DIR.])
fi
AC_SUBST(PARSORT_PREFIX)


# Checks for header files.
AC_LANG_PUSH([C])
AC_CHECK_HEADERS([fcntl.h stdlib.h sys/ioctl.h unistd.h])
BOOST_REQUIRE([1.46])



AC_OUTPUT(Makefile)
