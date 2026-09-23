# Released DDC reproduction

Dataset:
tmall-buy-merged

Backbone:
LightGCN

Config:
use_epre=True
epre_sort_mode=y_uio
epre_select_mode=top
epre_topk=0.3
loss_combination=b_a

e_pop:
official json
key=0

Result:

MRR@10:
0.07078

NDCG@10:
0.05882

AvgPop@10:
1729.18
