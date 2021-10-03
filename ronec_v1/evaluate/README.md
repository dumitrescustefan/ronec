## Evaluation script 

Use the ``eval_ronec.py`` script to evaluate RONEC performance of a transformer model using HuggingFace's awesome lib. 

The following results were obtained using the command ``python eval_bert.py --batch_size=16 --accumulate_grad_batches=4``:

```
{'valid_loss': 0.35410113722989056, 'valid_ent_type': 0.8919540229885057, 'valid_partial': 0.8862953138815208, 'valid_strict': 0.8247568523430592, 'valid_exact': 0.8463306808134393, 'test_loss': 0.3475743992762132, 'test_ent_type': 0.8885135135135135, 'test_partial': 0.8977102102102102, 'test_strict': 0.8303303303303303, 'test_exact': 0.8633633633633634}
```

Default params use a batch_size of 32 and no accumulation, but that requires 16GB of GPU RAM. The results above were obtained  running on a 11GB GPU. The default transformer model is ``dumitrescustefan/bert-base-romanian-cased-v1``.

Use ``python eval_bert.py --help`` for the full available parameters. 





