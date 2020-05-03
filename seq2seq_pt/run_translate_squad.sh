#!/bin/bash
python3 translate.py \
       -model /search/odin/zll/NQG/data/models/NQG_plus/result_0429_125215/model_e28.pt \
       -src  /search/odin/zll/NQG/data/test/dev.txt.shuffle.test.source.txt \
       -feats /search/odin/zll/NQG/data/test/dev.txt.shuffle.test.pos /search/odin/zll/NQG/data/test/dev.txt.shuffle.test.ner /search/odin/zll/NQG/data/test/dev.txt.shuffle.test.case \
       -tgt  /search/odin/zll/NQG/data/test/dev.txt.shuffle.test.target.txt \