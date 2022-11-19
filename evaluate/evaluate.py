import os, json, torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForTokenClassification, set_seed
from pytorch_lightning.callbacks import EarlyStopping
from nervaluate import Evaluator
import pandas as pd

class TransformerModel(pl.LightningModule):
    def __init__(self, model_name, tokenizer, lr, lr_factor, lr_patience, model_max_length, bio2tags, tag_list):
        super().__init__()

        print("Loading AutoModel [{}] ...".format(model_name))
        self.tokenizer = tokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(bio2tags), from_flax=False)

        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.model_max_length = model_max_length
        self.bio2tags = bio2tags
        self.tag_list = tag_list
        self.num_labels = len(bio2tags)

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
        raise Exception(
            "Could not detect SEP/EOS/BOS/CLS tokens, and thus could not assign a PAD token which is required.")

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

        loss, _ = self(input_ids, attention_mask, labels)
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

        return {
            "loss": loss, 
            "y": y, 
            "y_hat": y_hat
        }

    def validation_epoch_end(self, outputs):
        odf = pd.DataFrame(outputs)

        mean_val_loss = odf["loss"].mean()
        gold, pred = [], []
        for _, row in odf.iterrows():
            gold.append([self.bio2tags[token_id] for token_id in row["y"]])
            pred.append([self.bio2tags[token_id] for token_id in row["y_hat"]])

        evaluator = Evaluator(gold, pred, tags=self.tag_list, loader="list")

        results, _ = evaluator.evaluate()
        self.log("valid/avg_loss", mean_val_loss, prog_bar=True)
        self.log("valid/ent_type", float(results["ent_type"]["f1"]))
        self.log("valid/partial", float(results["partial"]["f1"]))
        self.log("valid/strict", float(results["strict"]["f1"]), prog_bar=True)
        self.log("valid/exact", float(results["exact"]["f1"]))

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

        return {
            "loss": loss, 
            "y": y,
            "y_hat": y_hat
        }

    def test_epoch_end(self, outputs):
        odf = pd.DataFrame(outputs)
        
        mean_val_loss = odf["loss"].mean()
        gold, pred = [], []
        for _, row in odf.iterrows():
            gold.append([self.bio2tags[token_id] for token_id in row["y"]])
            pred.append([self.bio2tags[token_id] for token_id in row["y_hat"]])

        evaluator = Evaluator(gold, pred, tags=self.tag_list, loader="list")

        results, _ = evaluator.evaluate()
        self.log("test/avg_loss", mean_val_loss, prog_bar=True)
        self.log("test/ent_type", results["ent_type"]["f1"])
        self.log("test/partial", results["partial"]["f1"])
        self.log("test/strict", results["strict"]["f1"])
        self.log("test/exact", results["exact"]["f1"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    factor=self.lr_factor, 
                    patience=self.lr_patience, 
                    mode='max'
                ),
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'valid/strict',
                'strict': True,
                'name': 'learning_rate',
            },
        }

    def predict(self, input_string):
        input_ids = self.tokenizer.encode(input_string, add_special_tokens=False)

        # run the model
        output = self.model(input_ids=torch.unsqueeze(torch.LongTensor(input_ids), 0), return_dict=True)
        logits = output["logits"]

        # extract results
        indices = torch.argmax(logits.detach().cpu(), dim=-1).squeeze(dim=0).tolist()  # reduce to [batch_size, seq_len] as list

        output_ids = []

        for id in input_ids:
            output_ids.append(self.tokenizer.decode(id))
        
        output_classes = []
        for i in indices:
            output_classes.append(self.bio2tags[i])
        
        return output_ids, output_classes


class RoNecDataset(Dataset):
    def __init__(self, instances):
        self.instances = []

        # run check
        for instance in instances:
            ok = True
            if len(instance["ner_ids"]) != len(instance["tokens"]):
                print("Different length ner_tags found")
                ok = False
            else:
                for _, token in zip(instance["ner_ids"], instance["tokens"]):
                    if token.strip() == "":
                        ok = False
                        print("Empty token found")
            if ok:
                self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i]


class Collator(object):
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
            if self.tokenizer.cls_token_id and self.tokenizer.sep_token_id:
                instance_ids = [self.tokenizer.cls_token_id] + instance_ids + [self.tokenizer.sep_token_id]
                instance_labels = [0] + instance_labels + [0]
                instance_token_idx = [-1] + instance_token_idx  # no need to pad the last, will do so automatically at return
            instance_attention = [1] * len(instance_ids)

            # update max_len for later padding
            max_len = max(max_len, len(instance_ids))

            # add to batch
            batch_input_ids.append(torch.LongTensor(instance_ids))
            batch_labels.append(torch.LongTensor(instance_labels))
            batch_attention.append(torch.LongTensor(instance_attention))
            batch_token_idx.append(torch.LongTensor(instance_token_idx))

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(batch_attention, batch_first=True, padding_value=0),
            "labels": torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=0),
            "token_idx": torch.nn.utils.rnn.pad_sequence(batch_token_idx, batch_first=True, padding_value=-1)
        }


def run_evaluation(args):
    print("Loading data...")
    with open(args.train_file, "r", encoding="utf8") as f:
        train_data = json.load(f)
    with open(args.validation_file, "r", encoding="utf8") as f:
        validation_data = json.load(f)
    with open(args.test_file, "r", encoding="utf8") as f:
        test_data = json.load(f)

    # deduce bio2 tag mapping and simple tag list, required by nervaluate
    # deduce bio2 tag mapping and simple tag list, required by nervaluate
    tags = ["O"] * 16  # tags without the B- or I- prefix
    bio2tags = ["O"] * 31  # tags with the B- and I- prefix, all tags are here

    for instance in train_data:
        for tag, tag_index in zip(instance["ner_tags"], instance["ner_ids"]):
            bio2tags[tag_index] = tag  # put the bio2 tag in its correct position
            if tag_index % 2 == 0 and tag_index > 0:
                tags[int(tag_index / 2)] = tag[2:]

    print(f"\tDataset contains {len(bio2tags)} BIO2 classes: {bio2tags}.")
    print(f"\tThere are {len(tags)} classes: {tags}\n")

    # init tokenizer and start loading data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, strip_accents=False)
    train_dataset = RoNecDataset(train_data)
    val_dataset = RoNecDataset(validation_data)
    test_dataset = RoNecDataset(test_data)

    collator = Collator(tokenizer=tokenizer, max_seq_len=args.model_max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                  collate_fn=collator, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False,
                                collate_fn=collator, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False,
                                 collate_fn=collator, pin_memory=True)

    print("\tTrain dataset has {} instances.".format(len(train_dataset)))
    print("\tValid dataset has {} instances.".format(len(val_dataset)))
    print("\tTest dataset has {} instances.\n".format(len(test_dataset)))

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
    while itt < args.experiment_iterations:
        print("Running experiment {}/{}".format(itt + 1, args.experiment_iterations))

        model = TransformerModel(
            model_name=args.model_name,
            lr=args.lr,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            model_max_length=args.model_max_length,
            bio2tags=bio2tags,
            tokenizer=tokenizer,
            tag_list=tags,
        )

        early_stop = EarlyStopping(
            monitor='valid/strict',
            min_delta=0.0001,
            patience=5,
            verbose=True,
            mode='max'
        )

        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=args.max_epochs,
            callbacks=[lr_monitor, early_stop],
            accumulate_grad_batches=args.accumulate_grad_batches,
            gradient_clip_val=1.0,
            #limit_train_batches=50,
            #limit_val_batches=50,
        )
        trainer.fit(model, train_dataloader, val_dataloader)

        print("\nEvaluating model on the VALIDATION dataset:")
        result_valid = trainer.test(model, val_dataloader)
        print("\nEvaluating model on the TEST dataset:")
        result_test = trainer.test(model, test_dataloader)

        with open("results_ronec_{}_of_{}.json".format(itt + 1, args.experiment_iterations), "w") as f:
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
        "valid_loss": sum(valid_loss) / args.experiment_iterations,
        "valid_ent_type": sum(valid_ent_type) / args.experiment_iterations,
        "valid_partial": sum(valid_partial) / args.experiment_iterations,
        "valid_strict": sum(valid_strict) / args.experiment_iterations,
        "valid_exact": sum(valid_exact) / args.experiment_iterations,
        "test_loss": sum(test_loss) / args.experiment_iterations,
        "test_ent_type": sum(test_ent_type) / args.experiment_iterations,
        "test_partial": sum(test_partial) / args.experiment_iterations,
        "test_strict": sum(test_strict) / args.experiment_iterations,
        "test_exact": sum(test_exact) / args.experiment_iterations
    }

    with open("results_{}.json".format(args.model_name.replace("/", "_")), "w") as f:
        json.dump(result, f, indent=4, sort_keys=True)

    print("\nFinal averaged results on TEST data: ")
    from pprint import pprint
    pprint(result)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    parser.add_argument('--model_name', type=str, default="dumitrescustefan/bert-base-romanian-uncased-v1")
    parser.add_argument("--train_file", type=str, default="../data/train.json")
    parser.add_argument("--validation_file", type=str, default="../data/valid.json")
    parser.add_argument("--test_file", type=str, default="../data/test.json")
    parser.add_argument('--lr', type=float, default=2e-05)
    parser.add_argument('--lr_factor', type=float, default=2/3)
    parser.add_argument('--lr_patience', type=float, default=5)
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--experiment_iterations', type=int, default=1)

    args = parser.parse_args()

    if args.seed >= 0:
        pl.seed_everything(args.seed, workers=True)
        set_seed(args.seed)
        os.environ['PYTHONHASHSEED']=str(args.seed)
    else:
        print("Using a random seed.")

    run_evaluation(args)
