"""This is a sample code on doing multi-party computation using Paillier HE.

This is part 2 of three separate scripts that compose three operations
done by three different parties.
This script will read the encrypted numbers and do some math,
then writes back to a disk file called "M2_list.json".
"""

import os
import numpy as np

#import the paillier library
from phe import paillier

# our own support library:
from paillier_tools import *

# load the encrypted messages
public_key, M1_list = envec_load_json(read_file("M1_list.json"))

# operate
M2_list = [ M * 2.0 for M in M1_list ]

# save to disk:
M2_json = envec_dump_json(public_key, M2_list, indent=2)
write_file("M2_list.json", M2_json)
os.chmod("M2_list.json", 0o644)
