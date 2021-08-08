#!/bin/bash

if [ -f libmumax3cl_d.hpp ]; then
  rm libmumax3cl_d.hpp
fi
cp libmumax3cl_f.hpp libmumax3cl_d.hpp
sed -i 's/float/double/' libmumax3cl_d.hpp
