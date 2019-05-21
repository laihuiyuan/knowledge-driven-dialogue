# -*- coding: UTF-8 -*-

import numpy as np

f_tr = open('./data/train.txt', 'w')
f_te = open('./data/dev.txt', 'w')

with open('./data/resource/new.txt', 'r') as f:
    nums = np.random.permutation(21858)
    for i, line in enumerate(f):
        if i in nums[:1500]:
            f_te.write(line)
        else:
            f_tr.write(line)
