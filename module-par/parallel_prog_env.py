# ipython / Jupyter environment file for ODU HPC (Wahab & Turing)
# NOTE: This module should mirror what "parallel-prog-env" provides.


def load_parallel_prog_env(module):
    """Initializes module environment for parallel programming.

    Unfortunately, the symbol named `module` unfortunately has to be
    imported from somewhere else.
    """
    global _CLUSTER
    with open("/etc/cluster", "r") as F:
        _CLUSTER = F.read().strip()

    if _CLUSTER == "turing":
        module("load", "DeapSECURE")
        module("load", "icc/19")
        module("load", "impi/19")
        module("load", "python/3.7")
        module("load", "mpi4py")
    elif _CLUSTER == "wahab":
        module("load", "DeapSECURE")
        module("load", "openmpi/3.1.4") # gcc-based
        module("load", "ucx/1.9.0")
        # Note: At present, mpi4py is provided in DeapSECURE shared library
    else:
        raise OSError("Unspported cluster: " + _CLUSTER)
