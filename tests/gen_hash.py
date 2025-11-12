# gen_hash.py
from passlib.hash import bcrypt
print(bcrypt.hash("Test123!")) 