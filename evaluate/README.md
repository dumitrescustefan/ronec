## Evaluation script 

Use the ``eval_ronec_v2.py`` script to evaluate the performance of a Romanian transformer token classification accuracy (NER task) on RONEC v2. 

The following results were obtained using the command ``python eval_ronec_v2.py --batch_size=8 --accumulate_grad_batches=2 --experiment_itterations 5 --model_name <model_name>``:

|                      Model                     	| test_loss 	| test_ent_type 	| test_partial 	| test_exact 	| test_strict 	|
|:----------------------------------------------:	|:---------:	|:-------------:	|:------------:	|:----------:	|:-----------:	|
| racai/distilbert-base-romanian-cased           	|   0.285   	|     0.888     	|     0.929    	|    0.919   	|    0.878    	|
| dumitrescustefan/bert-base-romanian-cased-v1   	|   0.261   	|     0.920     	|     0.940    	|    0.892   	|    0.919    	|
| dumitrescustefan/bert-base-romanian-uncased-v1 	|   0.319   	|     0.957     	|     0.987    	|    0.982   	|    0.952    	|
| readerbench/RoGPT2-medium                      	|   0.386   	|     0.925     	|     0.991    	|    0.989   	|    0.924    	|
| readerbench/RoGPT2-base                        	|   0.493   	|     0.819     	|     0.856    	|    0.804   	|    0.745    	|


The results above were obtained using a batch size of 16 (batch_size=8 and accumulate_grad_batches=2) with a default lr = 2e-05. For explanations of the metrics obtained using [nervaluate](https://pypi.org/project/nervaluate/), please see [this post](https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/). Training is stopped by optimizing the ``strict`` metric on the dev set. 

To run the script yourself, clone the repo, change the current working dir to ``evaluate``, install requirements with ``pip install -r requirements.txt`` (preferably in a virtual env) and run the command given above.
Use ``python eval_ronec_v2.py --help`` for the full available parameters.  





