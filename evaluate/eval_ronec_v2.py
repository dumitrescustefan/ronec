import logging, os, sys, json, torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss, CrossEntropyLoss
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, Trainer, TrainingArguments
from pytorch_lightning.callbacks import EarlyStopping
from nervaluate import Evaluator
import numpy as np

class TransformerModel(pl.LightningModule):
    def __init__(self, model_name="dumitrescustefan/bert-base-romanian-cased-v1", tokenizer_name=None, lr=2e-05,
                 model_max_length=512, bio2tag_list=[], tag_list=[]):
        super().__init__()

        if tokenizer_name is None or tokenizer_name == "":
            tokenizer_name = model_name

        print("Loading AutoModel [{}] ...".format(model_name))
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, strip_accents=False)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(bio2tag_list), from_flax=False)
        self.dropout = nn.Dropout(0.2)

        self.lr = lr
        self.model_max_length = model_max_length
        self.bio2tag_list = bio2tag_list
        self.tag_list = tag_list
        self.num_labels = len(bio2tag_list)

        self.train_loss = []
        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []
        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []

        # check cls, sep and pad tokens
        if self.tokenizer.cls_token_id is None:
            print(f"*** Warning, tokenizer {tokenizer_name} has no defined CLS token: sequences will not be marked with special chars! ***")
        if self.tokenizer.sep_token_id is None:
            print(f"*** Warning, tokenizer {tokenizer_name} has no defined SEP token: sequences will not be marked with special chars! ***")
       
        # add pad token
        self.validate_pad_token()
    
    def validate_pad_token(self):
        if self.tokenizer.pad_token is not None:
            return
        if self.tokenizer.sep_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the SEP token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.sep_token
            return
        if self.tokenizer.eos_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the EOS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            return
        if self.tokenizer.bos_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the BOS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.bos_token
            return
        if self.tokenizer.cls_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the CLS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.cls_token
            return
        raise Exception("Could not detect SEP/EOS/BOS/CLS tokens, and thus could not assign a PAD token which is required.")
        

    def forward(self, input_ids, attention_mask, labels):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return output["loss"], output["logits"]

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, logits = self(input_ids, attention_mask, labels)
        self.train_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_idx = batch["token_idx"]

        loss, logits = self(input_ids, attention_mask, labels)  # logits is [batch_size, seq_len, num_classes]

        batch_size = logits.size()[0]
        batch_pred = torch.argmax(logits.detach().cpu(), dim=-1).tolist()  # reduce to [batch_size, seq_len] as list
        batch_gold = labels.detach().cpu().tolist()  # [batch_size, seq_len] as list
        batch_token_idx = token_idx.detach().cpu().tolist()

        for batch_idx in range(batch_size):
            pred, gold, idx = batch_pred[batch_idx], batch_gold[batch_idx], batch_token_idx[batch_idx]
            y_hat, y = [], []
            for i in range(0, max(idx) + 1): # for each sentence
                pos = idx.index(i)  # find next token index and get pred and gold
                y_hat.append(pred[pos])
                y.append(gold[pos])
            self.valid_y_hat.append(y_hat)
            self.valid_y.append(y)

        self.valid_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        print()
        mean_val_loss = sum(self.valid_loss) / len(self.valid_loss)
        gold, pred = [], []
        for y, y_hat in zip(self.valid_y, self.valid_y_hat):
            gold.append([self.bio2tag_list[token_id] for token_id in y])
            pred.append([self.bio2tag_list[token_id] for token_id in y_hat])

        evaluator = Evaluator(gold, pred, tags=self.tag_list, loader="list")

        results, results_by_tag = evaluator.evaluate()
        self.log("valid/avg_loss", mean_val_loss, prog_bar=True)
        self.log("valid/ent_type", results["ent_type"]["f1"])
        self.log("valid/partial", results["partial"]["f1"])
        self.log("valid/strict", results["strict"]["f1"])
        self.log("valid/exact", results["exact"]["f1"])

        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_idx = batch["token_idx"]

        loss, logits = self(input_ids, attention_mask, labels)  # logits is [batch_size, seq_len, num_classes]

        batch_size = logits.size()[0]
        batch_pred = torch.argmax(logits.detach().cpu(), dim=-1).tolist()  # reduce to [batch_size, seq_len] as list
        batch_gold = labels.detach().cpu().tolist()  # [batch_size, seq_len] as list
        batch_token_idx = token_idx.detach().cpu().tolist()

        for batch_idx in range(batch_size):
            pred, gold, idx = batch_pred[batch_idx], batch_gold[batch_idx], batch_token_idx[batch_idx]
            y_hat, y = [], []
            for i in range(0, max(idx) + 1):  # for each sentence
                pos = idx.index(i)  # find next token index and get pred and gold
                y_hat.append(pred[pos])
                y.append(gold[pos])
            self.test_y_hat.append(y_hat)
            self.test_y.append(y)

        self.test_loss.append(loss.detach().cpu().numpy())

    def test_epoch_end(self, outputs):
        mean_val_loss = sum(self.test_loss) / len(self.test_loss)
        gold, pred = [], []
        for y, y_hat in zip(self.test_y, self.test_y_hat):
            gold.append([self.bio2tag_list[token_id] for token_id in y])
            pred.append([self.bio2tag_list[token_id] for token_id in y_hat])

        evaluator = Evaluator(gold, pred, tags=self.tag_list, loader="list")

        results, results_by_tag = evaluator.evaluate()
        self.log("test/avg_loss", mean_val_loss, prog_bar=True)
        self.log("test/ent_type", results["ent_type"]["f1"])
        self.log("test/partial", results["partial"]["f1"])
        self.log("test/strict", results["strict"]["f1"])
        self.log("test/exact", results["exact"]["f1"])

        import pprint
        print("_" * 120)
        print("\n\n Test results: \n")
        pprint.pprint(results["strict"])
        print("\n Per class Strict-F1 values:")
        for cls in self.tag_list:
            print(f'\t {cls} : \t{results_by_tag[cls]["strict"]["f1"]:.3f}')

        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)

    def predict(self, input_string):
        input_ids = self.tokenizer.encode(input_string, add_special_tokens=False)
        attention_mask = [1] * len(input_ids)

        # convert to tensors

        # run the model
        output = self.model(input_ids=torch.LongTensor(input_ids), return_dict=True)
        logits = output["logits"]

        # extract results
        indices = torch.argmax(logits.detach().cpu(), dim=-1).squeeze(dim=0).tolist()  # reduce to [batch_size, seq_len] as list

        for id, ind in zip(input_ids, indices):
            print(f"\t[{self.tokenizer.decode(id)}] -> {ind}")


class MyDataset(Dataset):
    def __init__(self, instances):
        self.instances = []

        # run check
        for instance in instances:
            ok = True
            if len(instance["ner_ids"]) != len(instance["tokens"]):
                print("Different length ner_tags found")
                ok = False
            else:
                for tag, token in zip(instance["ner_ids"], instance["tokens"]):
                    if token.strip() == "":
                        ok = False
                        print("Empty token found")
            if ok:
                self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i]


class MyCollator(object):
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        self.validate_pad_token()
        
    def validate_pad_token(self):
        if self.tokenizer.pad_token is not None:
            return
        if self.tokenizer.sep_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the SEP token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.sep_token
            return
        if self.tokenizer.eos_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the EOS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            return
        if self.tokenizer.bos_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the BOS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.bos_token
            return
        if self.tokenizer.cls_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the CLS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.cls_token
            return
        raise Exception("Could not detect SEP/EOS/BOS/CLS tokens, and thus could not assign a PAD token which is required.")
            

    def __call__(self, input_batch):
        batch_input_ids, batch_labels, batch_attention, batch_token_idx = [], [], [], []
        max_len = 0

        for instance in input_batch:
            instance_ids, instance_labels, instance_attention, instance_token_idx = [], [], [], []

            for i in range(len(instance["tokens"])):
                subids = self.tokenizer.encode(instance["tokens"][i], add_special_tokens=False)
                sublabels = [instance["ner_ids"][i]]

                if len(subids) > 1:  # we have a word split in more than 1 subids, fill appropriately
                    filler_sublabel = sublabels[0] if sublabels[0] % 2 == 0 else sublabels[0] + 1
                    sublabels.extend([filler_sublabel] * (len(subids) - 1))

                instance_ids.extend(subids)  # extend with the number of subids
                instance_labels.extend(sublabels)  # extend with the number of subtags
                instance_token_idx.extend([i] * len(subids))  # extend with the id of the token

                assert len(subids) == len(sublabels) # check for possible errors in the dataset

            if len(instance_ids) != len(instance_labels):
                print(len(instance_ids))
                print(len(instance_labels))
                print(instance_ids)
                print(instance_labels)
            assert len(instance_ids) == len(instance_labels)

            # cut to max sequence length, if needed
            if len(instance_ids) > self.max_seq_len - 2:
                instance_ids = instance_ids[:self.max_seq_len - 2]
                instance_labels = instance_labels[:self.max_seq_len - 2]
                instance_token_idx = instance_token_idx[:self.max_seq_len - 2]

            # prepend and append special tokens, if needed
            #print()
            #print(instance_ids)
            if self.tokenizer.cls_token_id and self.tokenizer.sep_token_id:
                instance_ids = [self.tokenizer.cls_token_id] + instance_ids + [self.tokenizer.sep_token_id]
                instance_labels = [0] + instance_labels + [0]
                instance_token_idx = [-1] + instance_token_idx  # no need to pad the last, will do so automatically at return
            #print(instance_ids)
            instance_attention = [1] * len(instance_ids)


            # update max_len for later padding
            max_len = max(max_len, len(instance_ids))

            # add to batch
            batch_input_ids.append(torch.LongTensor(instance_ids))
            batch_labels.append(torch.LongTensor(instance_labels))
            batch_attention.append(torch.LongTensor(instance_attention))
            batch_token_idx.append(torch.LongTensor(instance_token_idx))

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True,
                                                         padding_value=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(batch_attention, batch_first=True, padding_value=0),
            "labels": torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=0),
            "token_idx": torch.nn.utils.rnn.pad_sequence(batch_token_idx, batch_first=True, padding_value=-1)
        }


def run_evaluation(
        automodel_name: str,
        tokenizer_name: str,

        train_file: str = None,
        validation_file: str = None,
        test_file: str = None,
        dataset_name: str = None,

        gpus: int = 1,
        batch_size: int = 8,
        accumulate_grad_batches: int = 1,
        lr: float = 3e-5,
        model_max_length: int = 512,

        experiment_iterations: int = 1,
        results_file: str = "results_ronec_v2.json"
):
    print(f"Running {experiment_iterations} experiment(s) with model / tokenizer {automodel_name} / {tokenizer_name}")
    if dataset_name != "":
        print(f"\t with dataset {dataset_name}")
    if train_file != "":
        print(f"\t with training file {train_file}")
    if validation_file != "":
        print(f"\t with validation file {validation_file}")
    if test_file != "":
        print(f"\t with test file {test_file}")

    if dataset_name == "" and (train_file == "" or validation_file == "" or test_file == ""):
        print("\n Either a dataset or train/validation/test files must be given.")
        return

    print("\t batch size is {}, accumulate grad batches is {}, final batch_size is {}\n".format(
        batch_size,
        accumulate_grad_batches,
        batch_size * accumulate_grad_batches)
    )

    # load data
    if dataset_name == "":
        import random
        with open(train_file, "r", encoding="utf8") as f:
            train_data = json.load(f)#[:100]
        with open(validation_file, "r", encoding="utf8") as f:
            validation_data = json.load(f)
        with open(test_file, "r", encoding="utf8") as f:
            test_data = json.load(f)
    else:
        from datasets import load_dataset
        dataset = load_dataset(dataset_name)
        print(dataset)
        sys.exit(0)

    # deduce bio2 tag mapping and simple tag list, required by nervaluate
    tags = []  # tags without the B- or I- prefix
    bio2tags = set() # tags with the B- and I- prefix, all tags are here

    for instance in train_data + validation_data + test_data:
        for tag in instance["ner_tags"]:
            bio2tags.add(tag)

    print(f"Dataset contains {len(bio2tags)} BIO2 classes: {bio2tags}.")
    tags = sorted(list(set([tag[2:] if len(tag)>2 else tag for tag in bio2tags]))) # skip B- and I-
    print(f"\nThere are {len(tags)} classes: {tags}")

    # init tokenizer and start loading data
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, strip_accents=False)

    print("Loading data...")
    train_dataset = MyDataset(train_data)
    val_dataset = MyDataset(validation_data)
    test_dataset = MyDataset(test_data)

    my_collator = MyCollator(tokenizer=tokenizer, max_seq_len=model_max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True,
                                  collate_fn=my_collator, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, shuffle=False,
                                collate_fn=my_collator, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False,
                                 collate_fn=my_collator, pin_memory=True)

    print("Train dataset has {} instances.".format(len(train_dataset)))
    print("Valid dataset has {} instances.".format(len(val_dataset)))
    print("Test dataset has {} instances.\n".format(len(test_dataset)))

    itt = 0

    valid_loss = []
    valid_ent_type = []
    valid_partial = []
    valid_strict = []
    valid_exact = []
    test_loss = []
    test_ent_type = []
    test_partial = []
    test_strict = []
    test_exact = []

    while itt < experiment_iterations:
        print("Running experiment {}/{}".format(itt + 1, experiment_iterations))

        model = TransformerModel(
            model_name=automodel_name,
            lr=lr,
            model_max_length=model_max_length,
            bio2tag_list=list(bio2tags),
            tag_list=tags
        )

        early_stop = EarlyStopping(
            monitor='valid/strict',
            min_delta=0.001,
            patience=5,
            verbose=True,
            mode='max'
        )

        trainer = pl.Trainer(
            gpus=gpus,
            callbacks=[early_stop],
            # limit_train_batches=10,
            # limit_val_batches=2,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=1.0,
            enable_checkpointing=False
        )
        trainer.fit(model, train_dataloader, val_dataloader)

        print("\nEvaluating model on the VALIDATION dataset:")
        result_valid = trainer.test(model, val_dataloader)
        print("\nEvaluating model on the TEST dataset:")
        result_test = trainer.test(model, test_dataloader)

        with open("results_ronec_{}_of_{}.json".format(itt + 1, experiment_iterations), "w") as f:
            json.dump(result_test[0], f, indent=4, sort_keys=True)

        valid_loss.append(result_valid[0]['test/avg_loss'])
        valid_ent_type.append(result_valid[0]['test/ent_type'])
        valid_partial.append(result_valid[0]['test/partial'])
        valid_strict.append(result_valid[0]['test/strict'])
        valid_exact.append(result_valid[0]['test/exact'])
        test_loss.append(result_test[0]['test/avg_loss'])
        test_ent_type.append(result_test[0]['test/ent_type'])
        test_partial.append(result_test[0]['test/partial'])
        test_strict.append(result_test[0]['test/strict'])
        test_exact.append(result_test[0]['test/exact'])

        itt += 1

    print("Done, writing results...\n")

    result = {
        "valid_loss": sum(valid_loss) / experiment_iterations,
        "valid_ent_type": sum(valid_ent_type) / experiment_iterations,
        "valid_partial": sum(valid_partial) / experiment_iterations,
        "valid_strict": sum(valid_strict) / experiment_iterations,
        "valid_exact": sum(valid_exact) / experiment_iterations,
        "test_loss": sum(test_loss) / experiment_iterations,
        "test_ent_type": sum(test_ent_type) / experiment_iterations,
        "test_partial": sum(test_partial) / experiment_iterations,
        "test_strict": sum(test_strict) / experiment_iterations,
        "test_exact": sum(test_exact) / experiment_iterations
    }

    with open(results_file, "w") as f:
        json.dump(result, f, indent=4, sort_keys=True)

    print("\nFinal averaged results on TEST data: ")
    from pprint import pprint
    pprint(result)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser() # todo redo defaults
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--model_name', type=str,
                        default="dumitrescustefan/bert-base-romanian-cased-v1")
    parser.add_argument('--tokenizer_name', type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--train_file", type=str, default="../data/train.json")
    parser.add_argument("--validation_file", type=str, default="../data/valid.json")
    parser.add_argument("--test_file", type=str, default="../data/test.json")
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--experiment_iterations', type=int, default=1)
    parser.add_argument('--results_file', type=str, default="ronec_v2_results.json")

    args = parser.parse_args()

    if args.tokenizer_name == "":
        args.tokenizer_name = args.model_name

    run_evaluation(
        automodel_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        train_file=args.train_file,
        validation_file=args.validation_file,
        test_file=args.test_file,
        dataset_name=args.dataset_name,
        gpus=args.gpus,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        lr=args.lr,
        experiment_iterations=args.experiment_iterations,
        results_file=args.results_file
    )
