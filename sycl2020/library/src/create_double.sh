#!/bin/bash

if [ -f libmumax3cl_d.hpp ]; then
  rm libmumax3cl_d.hpp
fi
cp libmumax3cl_f.hpp libmumax3cl_d.hpp
sed -i 's/float/double/' libmumax3cl_d.hpp
if [ -f libmumax3cl_d.cpp ]; then
  rm libmumax3cl_d.cpp
fi
cp libmumax3cl_f.cpp libmumax3cl_d.cpp
sed -i 's/libmumax3cl_f/libmumax3cl_d/' libmumax3cl_d.cpp
