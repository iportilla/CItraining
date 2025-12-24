"""A sample code to do summation of encrypted numbers using
python-paillier library.
"""
import time
import numpy as np

# import the paillier library
from phe import paillier

pubkey, privkey = paillier.generate_paillier_keypair(n_length=2048)

x_center = 0.75
x_random = np.random.random(100)

x_plain = x_center + x_random
print("Plaintext x:")
print("  ", str(x_plain))

print("Encrypting numbers...")
t0 = time.time()
X_enc = [ pubkey.encrypt(x1) for x1 in x_plain ]
t1 = time.time()
print("time: encryption = {}".format(t1-t0))


t0 = time.time()
X_enc_sum = np.sum(X_enc)
t1 = time.time()
print("time: summation = {}".format(t1-t0))

# We only need to decrypt the sum:
x_sum = privkey.decrypt(X_enc_sum)
print("Sum of x = ", x_sum)


