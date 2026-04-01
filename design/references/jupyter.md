## Jupyter Notebooks

Prerequisites

To work through these guides you must

- Be a member of a project in the [BriCS portal](https://portal.isambard.ac.uk/)
- Have followed the instructions in [Getting Started](https://docs.isambard.ac.uk/user-documentation/getting_started/) to set up SSH login and create a valid SSH certificate for authenticating to BriCS facilities (using the `clifton` command line tool)
- Have followed the [instructions to install Conda](https://docs.isambard.ac.uk/user-documentation/guides/python/#conda-installing-and-using-miniforge) in your user storage space

JupyterHub on Isambard-AI Phase 2

Jupyter notebook sessions on Isambard-AI Phase 2 can be started directly through a web browser using [JupyterHub](https://jupyterhub.readthedocs.io/). Instructions are available in the the [JupyterHub guide](https://docs.isambard.ac.uk/user-documentation/guides/jupyterhub/).

To start a Jupyter notebook session on other BriCS compute services, please use the instructions on this page.

## Start an interactive JupyterLab session on a compute node

This guide outlines a procedure for starting a [JupyterLab](https://jupyterlab.readthedocs.io/) server on a compute node and connecting to this from a web browser on your computer using an SSH tunnel.

Connect to a login node over SSH, following the instructions in the [guide on logging in](https://docs.isambard.ac.uk/user-documentation/guides/login/).

On the login node, activate the Conda base environment, e.g. for Conda installed using [the instructions to install Conda](https://docs.isambard.ac.uk/user-documentation/guides/python/#conda-installing-and-using-miniforge)

```js
source ~/miniforge3/bin/activate
```

Create a Conda environment with JupyterLab installed, e.g. using the Conda environment file [jupyter-user-environment.yml](https://docs.isambard.ac.uk/user-documentation/guides/example-data/jupyter/jupyter-user-environment.yml)

```js
jupyter-user-environment.ymlname: jupyter-user-env
channels:
  - conda-forge
  - nodefaults
dependencies:
  # JupyterLab user interface
  - jupyterlab >=4.2,<5.0

  # Jupyter Notebook classic interface v7, built on modern Jupyter Lab & Jupyter Server
  - notebook >=7.2,<8.0
```

and the `conda env create` command:

```js
conda env create --file jupyter-user-environment.yml
```

From the login node we can submit a job which starts the JupyterLab server on a compute node.

The following example submission script [submit\_jupyter\_user\_session\_i-aip1.sh](https://docs.isambard.ac.uk/user-documentation/guides/example-data/jupyter/submit_jupyter_user_session_i-aip1.sh) launches a JupyterLab server on a compute node listening on a Slingshot High Speed Network (HSN) address. The job requests exclusive access to [1 GH200 superchip](https://docs.isambard.ac.uk/specs/#system-specifications-isambard-ai-phase-1) of the 4 per compute node.

```js
submit_jupyter_user_session_i-aip1.sh#!/bin/bash
#SBATCH --job-name=jupyter_user_session
#SBATCH --gpus=1  # this also allocates 72 CPU cores and 115GB memory
#SBATCH --time=01:00:00

source ~/miniforge3/bin/activate jupyter-user-env

# Add pre-installed kernelspecs to the Jupyter data search path
export JUPYTER_PATH="/tools/brics/jupyter/jupyter_data${JUPYTER_PATH:+:}${JUPYTER_PATH:-}"

HSN_FQDN="$(hostname).hsn.ai-p1.isambard.ac.uk"
LISTEN_IP=$(dig "${HSN_FQDN}" A +short | tail -n 1)
LISTEN_PORT=8888

set -o xtrace
jupyter lab --no-browser --ip="${LISTEN_IP}" --port="${LISTEN_PORT}"
```

Create a copy of the submission script on the login node, then submit this as a job using the `sbatch` command:

```js
$ sbatch submit_jupyter_user_session_i-aip1.sh
Submitted batch job 12345
```

The following example submission script [submit\_jupyter\_user\_session\_i-aip2.sh](https://docs.isambard.ac.uk/user-documentation/guides/example-data/jupyter/submit_jupyter_user_session_i-aip2.sh) launches a JupyterLab server on a compute node listening on a Slingshot High Speed Network (HSN) address. The job requests exclusive access to [1 GH200 superchip](https://docs.isambard.ac.uk/specs/#system-specifications-isambard-ai-phase-2) of the 4 per compute node.

```js
submit_jupyter_user_session_i-aip2.sh#!/bin/bash
#SBATCH --job-name=jupyter_user_session
#SBATCH --gpus=1  # this also allocates 72 CPU cores and 115GB memory
#SBATCH --time=01:00:00

source ~/miniforge3/bin/activate jupyter-user-env

# Add pre-installed kernelspecs to the Jupyter data search path
export JUPYTER_PATH="/tools/brics/jupyter/jupyter_data${JUPYTER_PATH:+:}${JUPYTER_PATH:-}"

HSN_FQDN="$(hostname).hsn.ai-p2.isambard.ac.uk"
LISTEN_IP=$(dig "${HSN_FQDN}" A +short | tail -n 1)
LISTEN_PORT=8888

set -o xtrace
jupyter lab --no-browser --ip="${LISTEN_IP}" --port="${LISTEN_PORT}"
```

Create a copy of the submission script on the login node, then submit this as a job using the `sbatch` command:

```js
$ sbatch submit_jupyter_user_session_i-aip2.sh
Submitted batch job 12345
```

The following example submission script [submit\_jupyter\_user\_session\_i3.sh](https://docs.isambard.ac.uk/user-documentation/guides/example-data/jupyter/submit_jupyter_user_session_i3.sh) launches a JupyterLab server on a compute node listening on a Slingshot High Speed Network (HSN) address. The job requests exclusive access to [a full Grace CPU superchip compute node](https://docs.isambard.ac.uk/specs/#system-specifications-isambard-3-grace).

```js
submit_jupyter_user_session_i3.sh#!/bin/bash
#SBATCH --job-name=jupyter_user_session
#SBATCH --nodes=1    # request a single compute node
#SBATCH --ntasks=1
#SBATCH --mem=0      # request all memory on the node
#SBATCH --exclusive  # request exclusive access to the node
#SBATCH --time=01:00:00

source ~/miniforge3/bin/activate jupyter-user-env

# Add pre-installed kernelspecs to the Jupyter data search path
export JUPYTER_PATH="/tools/brics/jupyter/jupyter_data${JUPYTER_PATH:+:}${JUPYTER_PATH:-}"

HSN_FQDN="$(hostname).hsn.cm.i3.isambard.ac.uk"
LISTEN_IP=$(dig "${HSN_FQDN}" A +short | tail -n 1)
LISTEN_PORT=8888

set -o xtrace
jupyter lab --no-browser --ip="${LISTEN_IP}" --port="${LISTEN_PORT}"
```

Create a copy of the submission script on the login node, then submit this as a job using the `sbatch` command:

```js
$ sbatch submit_jupyter_user_session_i3.sh
Submitted batch job 12345
```

Note the job ID returned by `sbatch` for future reference (12345 in the above example).

Request resources for the interactive session in the submission script

The resources requested in the job script (via `#SBATCH` directives) control the resources available for the Jupyter session and may need to be adjusted for your requirements. For example, the `--time` option sets the maximum runtime requested for the job and should be set to accommodate the length of time you expect to be working interactively. See the [Slurm documentation](https://docs.isambard.ac.uk/user-documentation/guides/slurm/) and [sbatch man page](https://slurm.schedmd.com/sbatch.html) for further information on specifying resources.

Once the job starts running, examine the output file (by default, named `slurm-<JOB_ID>.out`), e.g.

```js
$ cat slurm-12345.out
+ jupyter lab --no-browser --ip=10.253.8.114 --port=8888
[I 2024-06-28 15:08:55.458 ServerApp] jupyter_lsp | extension was successfully linked.
[I 2024-06-28 15:08:55.463 ServerApp] jupyter_server_terminals | extension was successfully linked.
[I 2024-06-28 15:08:55.468 ServerApp] jupyterlab | extension was successfully linked.
[I 2024-06-28 15:08:55.472 ServerApp] notebook | extension was successfully linked.
```

Near the start of the JupyterLab output there should be some URLs for accessing the server, e.g.

```js
To access the server, open this file in a browser:
        file:///lus/lfs1aip1/home/username/.local/share/jupyter/runtime/jpserver-264618-open.html
    Or copy and paste one of these URLs:
        http://10.253.8.242:8888/lab?token=c5185e5adb2574941d4f5136a7df23e3a441b56452bf9db0
        http://127.0.0.1:8888/lab?token=c5185e5adb2574941d4f5136a7df23e3a441b56452bf9db0
```

In the above output there are 2 URLs of the form

```js
http://<IP_ADDRESS>:<PORT>/lab?token=<TOKEN>
```

The following information will be required to set up a connection to the Jupyter server running on the compute node from a web browser on your computer:

**`10.253.x.y:8888` IP address and port**

The Jupyter server is listening on this IP address and port on the compute node. The IP address is the address of the compute node on the Slingshot High-Speed Network (HSN). This `<IP_ADDRESS>:<PORT>` combination will be used to set up an SSH tunnel from your computer to the Jupyter server.

**`10.242.x.y:8888` IP address and port**

The Jupyter server is listening on this IP address and port on the compute node. The IP address is the address of the compute node on the Slingshot High-Speed Network (HSN). This `<IP_ADDRESS>:<PORT>` combination will be used to set up an SSH tunnel from your computer to the Jupyter server.

**`10.243.x.y:8888` IP address and port**

The Jupyter server is listening on this IP address and port on the compute node. The IP address is the address of the compute node on the Slingshot High-Speed Network (HSN). This `<IP_ADDRESS>:<PORT>` combination will be used to set up an SSH tunnel from your computer to the Jupyter server.

**Authentication token (`<TOKEN>` in the [URL query string](https://en.wikipedia.org/wiki/Query_string))**

The token is used to authenticate to the Jupyter server when connecting from a web browser on your computer. The token can be copied from either of the URLs in the output.

Now the Jupyter server is running on a compute node and listening on a HSN IP address, an SSH tunnel can be used to allow a browser running on your computer to connect to the server on the compute node and control a Jupyter session.

Using the SSH certificate and the SSH configuration settings generated by the `clifton` tool (see the [guide to logging in](https://docs.isambard.ac.uk/user-documentation/guides/login/)), we can establish an SSH tunnel that forwards a local port on your computer to the compute node HSN IP address and port that JupyterLab is listening on.

To forward connections to `http://localhost:8888` in a web browser on your computer to the HSN IP address and port from JupyterLab's output using a project-specific host configuration generated by `clifton ssh-config`, run the following command in a terminal on your computer

```js
ssh -T -L localhost:8888:<IP_ADDRESS>:<PORT> <PROJECT>.aip1.isambard
```

```js
ssh -T -L localhost:8888:<IP_ADDRESS>:<PORT> <PROJECT>.aip2.isambard
```

```js
ssh -T -L localhost:8888:<IP_ADDRESS>:<PORT> <PROJECT>.3.isambard
```

substituting the HSN IP address and port from JupyterLab's output for `<IP_ADDRESS>` and `<PORT>`, and your project's short name for `<PROJECT>`.

This SSH command establishes a tunnel such that connections to `localhost:8888` on your computer will be forwarded to the compute node IP address and port where JupyterLab is listening, allowing applications running on your computer (such as your web browser) to communicate with the remote Jupyter server. The command will not output anything if it connects correctly and should be left running in the background while accessing the remote Jupyter server.

With the SSH tunnel established, it is now possible to connect to the JupyterLab session from a web browser.

Open a web browser on your computer, and go to the address

```js
http://localhost:8888/lab?token=<TOKEN>
```

substituting `<TOKEN>` for the authentication token in JupyterLab's output. For example, using the token from the above output, the address would be `http://localhost:8888/lab?token=c5185e5adb2574941d4f5136a7df23e3a441b56452bf9db0`.

This should connect the browser through to the JupyterLab server running on a compute node, providing access to the compute node via interactive notebook, console, and terminal interfaces:

![Screenshot of JupyterLab interface in a browser window served from http://localhost:8888](https://docs.isambard.ac.uk/user-documentation/guides/images/jupyter/jupyterlab_browser_localhost.png)

When you have finished working in JupyterLab, shut down the server using the File > Shut Down option from the JupyterLab menu

![Screenshot of JupyterLab interface in a browser window showing File menu with Shut Down option highlighted](https://docs.isambard.ac.uk/user-documentation/guides/images/jupyter/jupyterlab_browser_localhost_shutdown.png)

The job should be cancelled automatically once the server has shut down.

If there is an issue with shutting down JupyterLab from the browser interface, connect to the login node via SSH and use the `scancel` command to manually end the job

```js
scancel <JOB_ID>
```

substituting `<JOB_ID>` for the job ID output by the `sbatch` command used to the start the job.

Once the job is cancelled, you should also stop the `ssh -T -L` command running on your computer. This can be done by closing the terminal window or using the keyboard shortcut Ctrl + C with the terminal window focused.

## Use a custom Jupyter kernel in a JupyterLab session

This guide outlines a procedure for creating a [Jupyter kernel](https://docs.jupyter.org/en/latest/install/kernels.html) based on a custom Conda environment or Python virtual environment and using this in a JupyterLab session.

Creating a [Jupyter kernel](https://docs.jupyter.org/en/latest/install/kernels.html) for a Conda environment or Python virtual environment enables software installed in that environment (e.g. for machine learning, data analysis, or other workloads) to be used interactively in a Jupyter notebook or console.

### Creating a kernel spec

Jupyter kernels can be registered for use in JupyterLab by creating a [kernel spec](https://jupyter-client.readthedocs.io/en/latest/kernels.html#kernel-specs).

A [kernel spec](https://jupyter-client.readthedocs.io/en/latest/kernels.html#kernel-specs) for an IPython kernel with access to (Python) software installed in a Conda environment or Python virtual environment can be created using the [ipykernel](https://github.com/ipython/ipykernel) package.

See below for how to create a kernel spec from a Conda environment and a Python virtual environment.

#### 1\. Initial setup

Connect to a login node over SSH, following the instructions in the [guide on logging in](https://docs.isambard.ac.uk/user-documentation/guides/login/).

On the login node, activate the Conda base environment, e.g. for Conda installed using [the instructions to install Conda](https://docs.isambard.ac.uk/user-documentation/guides/python/#conda-installing-and-using-miniforge)

```js
source ~/miniforge3/bin/activate
```

Connect to a login node over SSH, following the instructions in the [guide on logging in](https://docs.isambard.ac.uk/user-documentation/guides/login/).

If necessary, load a suitable version of Python into the shell environment, e.g. by loading the `cray-python` module

```js
module load cray-python
```

#### 2\. Create the environment

Create a Conda environment containing the software packages to be used in a Jupyter session and the [ipykernel](https://github.com/ipython/ipykernel) package, e.g. using this [example Conda environment definition file](https://docs.isambard.ac.uk/user-documentation/guides/example-data/jupyter/data-analysis-example-environment.yml)

```js
data-analysis-example-environment.ymlname: data-analysis-env
channels:
  - conda-forge
  - nodefaults
dependencies:
  - python >= 3.10
  - ipykernel  # to create Jupyter kernels
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
```

and the `conda env create` command:

```js
conda env create --file data-analysis-example-environment.yml
```

This will create an environment containing the `ipykernel` package and a number of Python packages for data analysis and visualisation.

Using an existing Conda environment

To enable a kernel spec to be created for an existing Conda environment, install the [ipykernel](https://github.com/ipython/ipykernel) package into that environment using `conda install`, e.g.

```js
conda install -n <ENV_NAME> ipykernel
```

substituting `<ENV_NAME>` for the name of the already created Conda environment.

Create a Python virtual environment and install the [ipykernel](https://github.com/ipython/ipykernel) package alongside any other packages needed in this environment e.g. using this [example requirements file](https://docs.isambard.ac.uk/user-documentation/guides/example-data/jupyter/data-analysis-example-requirements.txt)

```js
data-analysis-example-requirements.txtipykernel  # to create Jupyter kernels
pandas
numpy
scikit-learn
matplotlib
seaborn
```
```js
python -m venv --upgrade-deps ~/venvs/data-analysis-venv
~/venvs/data-analysis-venv/bin/python -m pip install -r data-analysis-example-requirements.txt
```

This will create an environment containing the `ipykernel` package and a number of Python packages for data analysis and visualisation.

Using an existing Python virtual environment

To enable a kernel spec to be created for an existing Python virtual environment, install the [ipykernel](https://github.com/ipython/ipykernel) package into that environment using `python -m pip install`, e.g.

```js
<VENV_PATH>/bin/python -m pip install ipykernel
```

substituting `<VENV_PATH>` for the path where the virtual environment was created.

Don't forget the `ipykernel`!

The `ipykernel` package must be installed in the environment you wish to create a Jupyter kernel for. This is used to create and install the [kernel spec](https://jupyter-client.readthedocs.io/en/latest/kernels.html#kernel-specs) needed by Jupyter to make use of the kernel.

#### 3\. Install the kernel spec

Activate the environment, e.g. using the example environment

```js
conda activate data-analysis-env
```

Now run the `python -m ipykernel install` command to install a [kernel spec](https://jupyter-client.readthedocs.io/en/latest/kernels.html#kernel-specs) for the activated environment (in the default user location searched by Jupyter)

```js
python -m ipykernel install --user --name "data-analysis-env" --display-name "Python (data-analysis-env)"
```

This will install the kernel spec for the activated environment in the default user location searched by Jupyter. On Linux this is `~/.local/share/jupyter/kernels`. The value of the `--name` argument should be a unique name for the kernel spec, to be used internally by Jupyter as an identifier for the kernel. The value of `--display-name` is displayed in the Jupyter web interface.

Use an environment variable to name your kernel spec

You can use the `CONDA_DEFAULT_ENV` environment variable to get the name of the currently activated Conda environment, so the following general command will create a kernel spec for an activated environment with `--name` and `--display-name` based on the environment name

```js
python -m ipykernel install --user --name "${CONDA_DEFAULT_ENV}" --display-name "Python (${CONDA_DEFAULT_ENV})"
```

The `python -m ipykernel install` command should produce output stating the location of the kernel spec, e.g.

```js
Installed kernelspec data-analysis-env in /lus/lfs1aip1/home/username/.local/share/jupyter/kernels/data-analysis-env
```

Activate the virtual environment, e.g. using the example environment

```js
source ~/venvs/data-analysis-venv/bin/activate
```

Now run the `python -m ipykernel install` command to install a [kernel spec](https://jupyter-client.readthedocs.io/en/latest/kernels.html#kernel-specs) for the activated environment (in the default user location searched by Jupyter)

```js
python -m ipykernel install --user --name "data-analysis-venv" --display-name "Python (data-analysis-venv)"
```

This will install the kernel spec for the activated environment in the default user location searched by Jupyter. On Linux this is `~/.local/share/jupyter/kernels`. The value of the `--name` argument should be a unique name for the kernel spec, to be used internally by Jupyter as an identifier for the kernel. The value of `--display-name` is displayed in the Jupyter web interface.

Use an environment variable to name your kernel spec

You can use the `VIRTUAL_ENV` environment variable to get the name of the currently activated virtual environment, so the following general command will create a kernel spec for an activated virtual environment with `--name` and `--display-name` based on the environment name

```js
python -m ipykernel install --user --name "$(basename ${VIRTUAL_ENV})" --display-name "Python ($(basename ${VIRTUAL_ENV}))"
```

The `python -m ipykernel install` command should produce output stating the location of the kernel spec, e.g.

```js
Installed kernelspec data-analysis-venv in /lus/lfs1aip1/home/username/.local/share/jupyter/kernels/data-analysis-venv
```

#### Deactivate the environment

Deactivate the environment, e.g. by closing the terminal or running

```js
conda deactivate
```

Deactivate the environment, e.g. by closing the terminal or running

```js
deactivate
```

### Launching a custom kernel in a Jupyter session

Once created using the above instructions, the custom Jupyter kernel should be available to launch from a JupyterLab/Jupyter Notebook session submitted as a job to compute node (as described in the [guide on starting interactive JupyterLab sessions on a compute node](https://docs.isambard.ac.uk/user-documentation/guides/jupyter/#start-an-interactive-jupyterlab-session-on-a-compute-node)), e.g.

![Screenshot of JupyterLab interface in a browser window with the option to launch a custom kernel for Conda environment "data_analysis-env"](https://docs.isambard.ac.uk/user-documentation/guides/images/jupyter/jupyterlab_browser_localhost_custom_kernel.png)

In this example, selecting the "Python (data-analysis-env)" button under "Notebook" or "Console" in the Launcher tab will start an interactive notebook or console session using the kernel associated with the `data-analysis-env` environment.

Launching a notebook or console using a custom kernel for an environment will allow the packages installed in that environment to be used in the notebook, e.g. using Pandas installed in the [example data analysis environment](https://docs.isambard.ac.uk/user-documentation/guides/example-data/jupyter/data-analysis-example-environment.yml) in an notebook

![Screenshot of JupyterLab interface in a browser window with a notebook launched from a custom kernel running Pandas code to display a plot](https://docs.isambard.ac.uk/user-documentation/guides/images/jupyter/jupyterlab_browser_localhost_pandas_example_nb.png)

See the [JupyterLab User Guide](https://jupyterlab.readthedocs.io/en/stable/user/index.html) for further information on working with kernels in JupyterLab

### Managing kernel specs

Installed kernel specs can be managed using `jupyter kernelspec` subcommands. To list the kernels available to Jupyter use

```js
jupyter kernelspec list
```

Running `jupyter kernelspec`

The `jupyter kernelspec` command will need to be run in a context where the `jupyter` command is available (e.g. after activating the `jupyter-user-env` Conda environment created in the [guide on starting interactive JupyterLab sessions on a compute node](https://docs.isambard.ac.uk/user-documentation/guides/jupyter/#start-an-interactive-jupyterlab-session-on-a-compute-node)).

The output of `jupyter kernelspec list` should include the kernel spec installed previously, e.g.

```js
$ jupyter kernelspec list
Available kernels:
  python3              /lus/lfs1aip1/home/username/miniforge3/envs/jupyter-user-env/share/jupyter/kernels/python3
  data-analysis-env    /lus/lfs1aip1/home/username/.local/share/jupyter/kernels/data-analysis-env
```

To remove a kernel spec, use

```js
jupyter kernelspec remove <KERNEL_NAME>
```

replacing `<KERNEL_NAME>` with the name of the kernel in the `jupyter kernelspec list` output. See `jupyter kernelspec --help` for further management subcommands.

### Further information on Jupyter kernels

For further details on creating IPython/Jupyter kernels see [IPython documentation](https://ipython.readthedocs.io/en/latest/install/kernel_install.html).

For information on the search path for kernel specs and their internal structure, see the [Jupyter Client documentation](https://jupyter-client.readthedocs.io/en/latest/kernels.html#kernel-specs)
