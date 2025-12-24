"""
Paillier Tools - Utility functions for Paillier cryptosystem
Provides key serialization and file I/O helpers for the phe library.
"""

import json
from phe import paillier


def keypair_dump_jwk(public_key, private_key):
    """
    Serialize Paillier keypair to JWK (JSON Web Key) format.
    
    Args:
        public_key: PaillierPublicKey object
        private_key: PaillierPrivateKey object
        
    Returns:
        tuple: (public_key_jwk, private_key_jwk) as JSON strings
    """
    # Serialize public key
    public_jwk = {
        'kty': 'DAJ',  # Key type: Damgård-Jurik (Paillier is a special case)
        'alg': 'PAI-GN1',  # Algorithm: Paillier
        'n': str(public_key.n),  # Modulus
        'max_int': str(public_key.max_int)
    }
    
    # Serialize private key (includes public key info)
    private_jwk = {
        'kty': 'DAJ',
        'alg': 'PAI-GN1',
        'n': str(public_key.n),
        'p': str(private_key.p),
        'q': str(private_key.q)
    }
    
    return json.dumps(public_jwk), json.dumps(private_jwk)


def keypair_load_jwk(public_key_jwk, private_key_jwk):
    """
    Deserialize Paillier keypair from JWK format.
    
    Args:
        public_key_jwk: JSON string of public key
        private_key_jwk: JSON string of private key
        
    Returns:
        tuple: (PaillierPublicKey, PaillierPrivateKey)
    """
    # Parse JSON
    public_data = json.loads(public_key_jwk)
    private_data = json.loads(private_key_jwk)
    
    # Reconstruct keys
    n = int(private_data['n'])
    p = int(private_data['p'])
    q = int(private_data['q'])
    
    # Create public and private keys
    public_key = paillier.PaillierPublicKey(n=n)
    private_key = paillier.PaillierPrivateKey(public_key, p, q)
    
    return public_key, private_key


def write_file(filename, content):
    """
    Write content to a file.
    
    Args:
        filename: Path to file
        content: String content to write
    """
    with open(filename, 'w') as f:
        f.write(content)


def read_file(filename):
    """
    Read content from a file.
    
    Args:
        filename: Path to file
        
    Returns:
        str: File content
    """
    with open(filename, 'r') as f:
        return f.read().strip()
