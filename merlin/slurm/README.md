# Work with slurm

## Variables

### Job variables (`slurm_job.sh`)

🚧: requires configuration

+ `--account=GOVxxxxxx`: project ID 🚧
+ `--nodes=2`: total number of nodes 🚧
+ `--mem=8G`: main memory 🚧
+ `--ntasks-per-node=1`: fixed to 1, one torchrun per node https://stackoverflow.com/a/65897194
+ `--cpus-per-task=2`: same as gpu per node 🚧
+ `--gres=gpu:2`: number of allocated gpus per node 🚧
+ `--mail-type=END,BEGIN`: Send the mail when the job starts and finishes.
+ `--mail-user=xxx@xxx.com`: your email 🚧
+ `--time=00:00:20`: total run time limit (HH:MM:SS) 🚧

### Environment variables

+ `SLURM_PROCID`: task ID, which is also the node ID since our ntasks-per-node is set to 1, set by slurm
+ `LOCAL_RANK`: local processes/gpu id, set by torchrun
+ `WORLD_SIZE`: total number of processes/gpus (nnodes * nproc-per-node), set by torchrun
+ `RANK`: global processes/gpu id, set by torchrun

## Build image (local)

Build the image in our local machine from the definition (`paslab_llm.def`)

```bash
$ sudo singularity build paslab_llm.sif paslab_llm.def
$ singularity shell --nv paslab_llm.sif # test in a interactive shell
Apptainer> pip list | grep "mistral"
mistral_common            1.5.1
Apptainer> exit
```

Upload the built image (`paslab_llm.sif` 6.3G) to the remote slurm server

```bash
$ sftp <location>
sftp> put paslab_llm.sif
```

## Submit job (in the login node)

```bash
$ ls
model.py paslab_llm.sif slurm_job.sh
$ sbatch -o "R_$(date +%Y%m%d%H%M%S)_%j.log" slurm_job.sh
```

# References

+ [HackMD](https://hackmd.io/@aben20807/HyKAHCfg0/%2F%40aben20807%2FHySiPauLyg)
