# Environment Setup (last updated on 1/3/2025)
## 1. Install Dependencies (repeat this on every node)
* clone [this repo](https://github.com/muchi674/ntu_paslab_llm.git) and switch to branch named `merlin`
    ```shell
    git checkout merlin --
    ```
* create an conda environment named `merlin`
    ```shell
    conda create --name merlin python=3.10
    conda activate merlin
    ```

* install required packages from pypi channel
    ```shell
    pip install -r requirements.txt
    ```

## 2. generate ssh key pair
* on master node, generate ssh key pair using `ssh-keygen`
    ```shell
    ssh-keygen -t rsa -b 4096 -N "" -f $HOME/.ssh/id_merlin
    ```
* on master node, copy the public key to slave nodes using `ssh-copy-id` (repeat this for every slave node)
    ```shell
    ssh-copy-id -i $HOME/.ssh/id_merlin -p <slave node port> <username>@<slave node ip>
    ```
    * example: `ssh-copy-id -i $HOME/.ssh/id_merlin -p xxxx OOO@10.10.10.1` will add `id_merlin.pub` to 51 server's authorized_keys under `OOO`'s `.ssh` directory

## 3. modify ~/.bashrc (repeat this on every node)
comment out these lines
```shell
# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac
```

# Run the Script
1. modify `merlin/launch/launch_config.json`, make sure to set `username` to your account name
2. under `merlin/launch/`, run:
    ```shell
    python launch_nodes.py --launch-config ./launch_config.json
    ```
3. to terminate the process, run:
    ```shell
    python launch_nodes.py --launch-config ./launch_config.json --terminate
    ```
> the result is now located in 51 server, use `tmux a -t merlin` on 51 server to view them.
# References
* [pytorch x.x.x+cu121](https://pytorch.org/)
*  [xformers](https://github.com/facebookresearch/xformers)
*  [mistral_common](https://github.com/mistralai/mistral-common)