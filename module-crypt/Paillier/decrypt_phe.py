"""This is a sample code on doing multi-party computation using Paillier HE.

This is part 3 of three separate scripts that compose three operations
done by three different parties.
This script will read "M2_list.json" and decrypt the numbers.
"""

import numpy as np

#import the paillier library
from phe import paillier

# our own support library:
from paillier_tools import *

# we have to load the private key:
public_key, private_key = keypair_load_jwk(read_file("phe_key.pub"),
                                           read_file("phe_key.priv"))

# load the encrypted messages
pubkey_Unused, M2_list = envec_load_json(read_file("M2_list.json"))

# decrypt
m2_list = [ private_key.decrypt(M) for M in M2_list ]

# save to disk:
print("Decrypted numbers:")
print(m2_list)
