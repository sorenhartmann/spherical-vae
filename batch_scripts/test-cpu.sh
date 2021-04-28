 #!/bin/sh
 #BSUB -q hpc
 #BSUB -J hparam-search-cpu-svae-mocap
 #BSUB -n 1
 #BSUB -W 2:00
 #BSUB -B
 #BSUB -N
 #BSUB -R span[hosts=1]
 #BSUB -R "rusage[mem=4GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err

module load python3/3.7.7

python3 src/models/convolutional.py