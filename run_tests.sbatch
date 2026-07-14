#!/bin/bash
#SBATCH -A csc607
#SBATCH -J rocshmem_test_suite
#SBATCH -N 2
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o test_output_%j.log

/ccs/home/angeloej/eze-p/tests.sh