import os

for j in range(5):
    for i in range(5):
        os.system(
            f"cd /home/jonfrey/git/model-uncertainty-for-adaptation && /home/jonfrey/miniconda3/envs/stego/bin/python3 /home/jonfrey/git/model-uncertainty-for-adaptation/do_segm.py --city=Rome --no-src-data --freeze-classifier  --unc-noise  --lambda-ce=1 --lambda-ent=1 --save=/home/jonfrey/git/model-uncertainty-for-adaptation/results --lambda-ssl=0.1 --scene=scene000{i} --prefix=run{j}"
        )
