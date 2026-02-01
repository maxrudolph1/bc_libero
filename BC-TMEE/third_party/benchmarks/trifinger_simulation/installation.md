First, install dependencies.

- **Install Claire's fork of the `trifinger_simulation` package**
    - This is the official pyBullet simulation of the TriFinger robot
    - To install, first clone [my fork of the package](https://github.com/ClaireLC/trifinger_simulation)`git clone <https://github.com/ClaireLC/trifinger_simulation.git>`
    - Then, follow the installation [instructions in their documentation](https://open-dynamic-robot-initiative.github.io/trifinger_simulation/getting_started/installation.html)
        - On Linux, there should be no issues following the instructions as is; all the pip packages in `requirements.txt` should install with no issues.
        - On my M1 mac, I had to first create a conda env with python 3.8 and install each of the packages in `requirements.txt` one-by-one (except the`pin` package)
            - Using pip to install the `pin` package only works on Linux. For Mac, use conda to install `pinocchio`: `conda install pinocchio -c conda-forge`. 
                - After installing `pinocchio`, check that you can run `import pinocchio`. If this results in a segfault, try installing `eigenpy=2.7.10` with `conda install -c conda-forge eigenpy=2.7.10`. 
                - Note (September 21, 2022): We found that `eigenpy=2.7.13` results in a segfault upon running `import pinocchio`. For some reason, when installing `pinocchio` with conda, the package manager automatically installs `eigenpy=2.7.13`, so we needed to downgrade the version to `eigenpy=2.7.10` to get `pinocchio` to import.
            - If you install `pinocchio` with conda, you'll need to comment out the lines in `trifinger_simulation/setup.py` and `trifinger_simulation/requirements.txt` that look for `pin` (Line 5 in `requirements.txt` and Line 47 in `setup.py`).
- **`imitation_learning` package**
    - Follow installation instructions in the `eai-foundations/eval/trifinger/imitation_learning` directory.
- **`torchrl` package**
    - For Linux and non-M1/M2 MacOS computers, `pip install torchl` should work.
    - For M1/M2 Macs, need to install from [source](https://github.com/facebookresearch/rl):
        
        ```
        git clone <https://github.com/facebookresearch/rl>
        cd /path/to/torchrl/
        python setup.py develop
        ```


Then, install this package as an editable package in your conda env of choice.
