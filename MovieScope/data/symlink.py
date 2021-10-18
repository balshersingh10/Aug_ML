# Python program to explain os.symlink() method

# importing os module
import os


# Source file path
src = './current_genres.txt'

# Destination file path
dst = './current_genres(symlink).txt'

# Create a symbolic link
# pointing to src named dst
# using os.symlink() method
os.symlink(src, dst)

print("Symbolic link created successfully")
