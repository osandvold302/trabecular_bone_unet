import os
import hashlib
import random
import paramiko
import json
import glob

study_dir = '/d1/hip/DL/SPGR_AF2_242_242_1500_um/'
scan_files = glob.glob(study_dir + '*/data/dataFile.mat')

shuffled = random.shuffle(scan_files)
hashed_files = { 100+i : scan_files[i] for i in range(len(scan_files)) }

with open('mapped_files.json', 'w') as fp:
    json.dump(hashed_files, fp)

username = 'sandvolo'
hostname = 'HOSTNAME'
password = 'PASSWORD'

transport = paramiko.Transport((hostname))

transport.connect(username = username, password = password)

sftp = paramiko.SFTPClient.from_transport(transport)

target_dir = './SPGR_AF2_242_242_1500_um/'

for key, localpath in hashed_files.items():
    sftp.put(localpath, target_dir + str(key) + '_dataFile.mat')

sftp.close()
transport.close()
print ('Upload done.')

