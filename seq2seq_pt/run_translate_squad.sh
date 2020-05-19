#!/bin/bash
python3 translate.py \
       -model /search/odin/zll/NQG/data/models/NQG_plus/result_20200516-222400/model_dev_metric_0.0853_e7.pt \
       -src  /search/odin/zll/NQG/data/test/dev.txt.shuffle.test.source.txt \
       -feats /search/odin/zll/NQG/data/test/dev.txt.shuffle.test.pos /search/odin/zll/NQG/data/test/dev.txt.shuffle.test.ner /search/odin/zll/NQG/data/test/dev.txt.shuffle.test.case \
       -tgt  /search/odin/zll/NQG/data/test/dev.txt.shuffle.test.target.txt \