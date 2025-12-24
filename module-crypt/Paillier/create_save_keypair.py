"""Demo for creating & saving keypair
"""

# import additional stuff needed later on:
import json
import os
import stat

# import Paillier library
import phe
from phe import paillier

# our own I/O library
from paillier_tools import *

pubkey, privkey = paillier.generate_paillier_keypair(n_length=2048)

pub_jwk, priv_jwk = keypair_dump_jwk(pubkey, privkey)

with open("phe_key.pub", "w") as F:
    F.write(pub_jwk + "\n")
    print("Written public key to {}".format(F.name))
    print("n={}".format(pubkey.n))

# The private key file is more critical and has to have rw flags only for the owner,
# so we have to do this to always create a new file with 0600 permission to begin with
# Ref: https://stackoverflow.com/a/15015748/655885
fname = "phe_key.priv"
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL  # Refer to "man 2 open".
mode = stat.S_IRUSR | stat.S_IWUSR  # This is 0o600.
umask = 0o777 ^ mode  # Prevents always downgrading umask to 0.

# For security, remove file with potentially elevated mode
try:
    os.remove(fname)
except OSError:
    pass

# Open file descriptor and set the file perm from the outset!
umask_original = os.umask(umask)
try:
    fdesc = os.open(fname, flags, mode)
finally:
    os.umask(umask_original)

# Open file handle and write to file
with os.fdopen(fdesc, "w") as F:
    F.write(priv_jwk + "\n")


#os.system("lsof -p %s" % os.getpid())
#os.close(fdesc) -- don't! It's already closed

