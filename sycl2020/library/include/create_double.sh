#!/bin/bash

if [ -f libmumax3cl_d.h ]; then
  rm libmumax3cl_d.h
fi
cp libmumax3cl_f.h libmumax3cl_d.h
sed -i 's/float/double/' libmumax3cl_d.h
