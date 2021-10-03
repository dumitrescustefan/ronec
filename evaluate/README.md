## Evaluation script 

Use the ``eval_ronec_v2.py`` script to evaluate the performance of a Romanian transformer token classification accuracy (NER task) on RONEC v2. 

The following results were obtained using the command ``python eval_ronec_v2.py --batch_size=8 --accumulate_grad_batches=4 --experiment_itterations 5``:

```json
coming soon!
```

Default params use a batch_size of 8 and no accumulation. The results above were obtained running on an 11GB 2080Ti GPU. The default transformer model is ``dumitrescustefan/bert-base-romanian-cased-v1``. PyTorch version was 1.9.1, pytorch_lightning at 1.4.9.

To run the script yourself, clone the repo, change the current working dir to ``evaluate``, install requirements with ``pip install -r requirements.txt`` (preferably in a virtual env) and run the command given above.
Use ``python eval_ronec_v2.py --help`` for the full available parameters.  





