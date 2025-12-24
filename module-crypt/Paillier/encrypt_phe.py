"""This is a sample code on doing multi-party computation using Paillier HE.

This is part 1 of three separate scripts that compose three operations
done by three different parties:

* script 1 (this one) generates the plaintext numbers and encrypts them
  and writes them to disk file called "M1_list.json"

* script 2 (math1_phe.py) will read the encrypted numbers and do some math,
  then writes back to a disk file called "M2_list.json"

* script 3 (decrypt_phe.py) will read "M2_list.json" and decrypt the numbers.
"""

import os
import numpy as np

#import the paillier library
from phe import paillier

# our own support library:
from paillier_tools import *

# load a previously generated key:
public_key = pubkey_load_jwk(read_file("phe_key.pub"))

# the original messages
m1_list = [3.141592653, 300, -4.6e-12]

# the encrypted messages
M1_list = [public_key.encrypt(x) for x in m1_list]

# save to disk:
M1_json = envec_dump_json(public_key, M1_list, indent=2)
write_file("M1_list.json", M1_json)
os.chmod("M1_list.json", 0o644)
