#!/bin/bash
#SBATCH --job-name=d2d_test_samplers
#SBATCH --qos=broadwl
#SBATCH --output=slurm_logs/test_samplers.out
#SBATCH --error=slurm_logs/test_samplers.err
#SBATCH --account=pi-rkessler
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000
#SBATCH --time=0-01:00:00

# Environment set-up
module load mpich
source /project2/rkessler/PRODUCTS/miniconda/bin/activate
conda deactivate
conda activate
conda activate DEBASS

# Relax locked memory limit for Infiniband communication
ulimit -l unlimited

# OpenMP thread configuration
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# NumPy / BLAS thread configuration
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

CONFIG=$DEBASS_USERS/mariaace/DR1/d2d_upgrades/DES_CONFIG.yml

# Print job information
echo "Job started at $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Config: $CONFIG"
echo ""

MPIRUN="mpirun -np $SLURM_NTASKS --map-by socket --bind-to socket"

# -----------------------------------------------------------------------
# 1. emcee -- DEBUG mode (3 steps, MPI)
# -----------------------------------------------------------------------
echo "============================================================"
echo "TEST 1/3: emcee (--DEBUG, MPI)"
echo "============================================================"
$MPIRUN dust2dusty --CONFIG $CONFIG --SAMPLER emcee --USE_MPI --DEBUG
echo "emcee debug run exit code: $?"
echo ""

# -----------------------------------------------------------------------
# 2. zeus -- DEBUG mode (3 steps, MPI)
# -----------------------------------------------------------------------
echo "============================================================"
echo "TEST 2/3: zeus (--DEBUG, MPI)"
echo "============================================================"
$MPIRUN dust2dusty --CONFIG $CONFIG --SAMPLER zeus --USE_MPI --DEBUG
echo "zeus debug run exit code: $?"
echo ""

# -----------------------------------------------------------------------
# 3. nautilus -- DEBUG mode (3 likelihood calls, MPIPoolExecutor)
# -----------------------------------------------------------------------
echo "============================================================"
echo "TEST 3/3: nautilus (--DEBUG, MPI)"
echo "============================================================"
$MPIRUN dust2dusty --CONFIG $CONFIG --SAMPLER nautilus --USE_MPI --DEBUG
echo "nautilus debug run exit code: $?"
echo ""

echo "============================================================"
echo "All sampler tests complete at $(date)"
echo "============================================================"
