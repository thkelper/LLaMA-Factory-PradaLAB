import hashlib
from fire import Fire

def main(path):
    with open(path, 'rb') as f:
        file_hash = hashlib.sha1()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    print(file_hash.hexdigest())

if __name__ == '__main__':
    Fire(main)