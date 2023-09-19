# SA-MESH

[[paper]][0]

This repo implements the experiments in Unlocking Slot Attention by Changing Optimal Transport Costs.

To run the experiments, first prepare the data with the instructions provided in ``data_scripts/README.MD`` and install the necessary packages.

## Experiments

mdsprites:

```
python run_object_discovery.py data=mdsprites model=image model/slot_attention=mesh
```

ClevrTex:

```
python run_object_discovery.py data=clevrtex64 model=image model/slot_attention=mesh
```

CLEVRER-S:

```
python run_object_discovery.py data=clevrer_s model=video model/slot_attention=mesh
```

CLEVRER-L:

```
python run_object_discovery.py data=clevrer_l model=video model/slot_attention=mesh
```


[0]: https://arxiv.org/abs/2301.13197
