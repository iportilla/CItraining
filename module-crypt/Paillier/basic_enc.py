"""This is a sample code on basic operation of encrypted numbers
with python-paillier library: https://github.com/n1analytics/python-paillier/tree/master/phe
"""
import numpy as np

#import the paillier library
from phe import paillier

#generate keys
public_key, private_key = paillier.generate_paillier_keypair()

#the original message
secret_number_list = [3.141592653, 300, -4.6e-12]

#encrypt the message
encrypted_number_list = [public_key.encrypt(x) for x in secret_number_list]

#get the encrypted number
encrypted_number = [k.ciphertext(be_secure=False) for k in encrypted_number_list]

#decrypt the message
decrypted_number_list = [private_key.decrypt(x) for x in encrypted_number_list]

#homomorphic addition
enc_add = np.add(np.array(encrypted_number_list),np.array([1,1, 1]))

#homomorphic multiplication
enc_mult = np.multiply(np.array(encrypted_number_list),np.array([2,2,2]))

#decrypt the homomorphic addtion
decrypted_add = [private_key.decrypt(x) for x in enc_add]

#decrypt the homormorphic multiplication
decrypted_mult = [private_key.decrypt(x) for x in enc_mult]

#print the message
print('Original data: ',secret_number_list)
print ('Encrypted data: ',encrypted_number)
print ('Decrypted data: ',decrypted_number_list)
print ('Decrypted addition: ',decrypted_add)
print ('Decrypted multiplication: ',decrypted_mult)



