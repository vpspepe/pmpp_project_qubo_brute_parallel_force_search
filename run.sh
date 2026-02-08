#!/bin/bash

# SLURM parameters
#SBATCH -n 4
#SBATCH -t 5
#SBATCH --mem-per-cpu 3800
#SBATCH --gpus-per-task=1

# Special parameters. DO NOT CHANGE THESE!
#SBATCH -A kurs00091
#SBATCH -p kurs00091
#SBATCH --reservation=kurs00091

# Redirect stdout and stderr
#SBATCH -o dense.out
#SBATCH -e dense.err

module purge
module load cuda/12.5 gcc/13.1.0

# Profiling
mkdir -p out
rm -f out/*

ncu --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section Occupancy --section PmSampling --section PmSampling_WarpStates --section SchedulerStats --section SourceCounters --section ComputeWorkloadAnalysis --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --section WorkloadDistribution -o out/profiling --import-source on -f build/QUBOBruteForcing

tar -czvf out.tar.gz out

./build/QUBOBruteForcing
cat *.out *.err > out.txt

