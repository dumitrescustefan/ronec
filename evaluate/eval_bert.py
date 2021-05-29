import logging, os, sys, json, torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss, CrossEntropyLoss
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, Trainer, TrainingArguments
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from nervaluate import Evaluator
import numpy as np


class TransformerModel(pl.LightningModule):
    def __init__(self, model_name="dumitrescustefan/bert-base-romanian-cased-v1", num_labels=34, lr=2e-05,
                 model_max_length=512, label_encoder=None, tags=np.array(["O"])):
        super().__init__()
        print("Loading AutoModel [{}]...".format(model_name))
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = nn.Dropout(0.2)

        self.label_encoder = label_encoder

        self.lr = lr
        self.model_max_length = model_max_length

        self.tags = tags.tolist()
        self.tags.remove(args.pad_label)
        self.tags.remove("O")

        self.tags = list(set([tag.split("-")[1] for tag in self.tags]))

        self.train_y_hat = []
        self.train_y = []
        self.train_loss = []
        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []
        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []

        self.cnt = 0

    def forward(self, inputs_ids, attention_mask, labels):
        output = self.model(
            inputs_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return output["loss"], output["logits"]

    def training_step(self, batch, batch_idx):
        inputs_ids, attention_mask, labels, token_idx = batch

        loss, logits = self(inputs_ids, attention_mask, labels)

        self.train_y_hat.extend(pred[idx].tolist() for pred, idx in zip(torch.argmax(logits, -1).detach().cpu(), token_idx))
        self.train_y.extend(gold[idx].tolist() for gold, idx in zip(labels.detach().cpu(), token_idx))
        self.train_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        mean_train_loss = sum(self.train_loss) / len(self.train_loss)
        gold = [self.label_encoder.inverse_transform(labels) for labels in self.train_y]
        pred = [self.label_encoder.inverse_transform(labels) for labels in self.train_y_hat]

        evaluator = Evaluator(gold, pred, tags=self.tags, loader="list")

        results, results_by_tag = evaluator.evaluate()
        self.log("train/avg_loss", mean_train_loss, prog_bar=True)
        self.log("train/ent_type", results["ent_type"]["f1"])
        self.log("train/partial", results["partial"]["f1"])
        self.log("train/strict", results["strict"]["f1"])
        self.log("train/exact", results["exact"]["f1"], prog_bar=True)

        self.train_y_hat = []
        self.train_y = []
        self.train_loss = []

    def validation_step(self, batch, batch_idx):
        inputs_ids, attention_mask, labels, token_idx = batch

        loss, logits = self(inputs_ids, attention_mask, labels)

        self.valid_y_hat.extend(pred[idx].tolist() for pred, idx in zip(torch.argmax(logits, -1).detach().cpu(), token_idx))
        self.valid_y.extend(gold[idx].tolist() for gold, idx in zip(labels.detach().cpu(), token_idx))
        self.valid_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        mean_val_loss = sum(self.valid_loss) / len(self.valid_loss)
        gold = [self.label_encoder.inverse_transform(labels) for labels in self.valid_y]
        pred = [self.label_encoder.inverse_transform(labels) for labels in self.valid_y_hat]

        evaluator = Evaluator(gold, pred, tags=self.tags, loader="list")

        results, results_by_tag = evaluator.evaluate()
        self.log("valid/avg_loss", mean_val_loss, prog_bar=True)
        self.log("valid/ent_type", results["ent_type"]["f1"])
        self.log("valid/partial", results["partial"]["f1"])
        self.log("valid/strict", results["strict"]["f1"])
        self.log("valid/exact", results["exact"]["f1"], prog_bar=True)

        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []

    def test_step(self, batch, batch_idx):
        inputs_ids, attention_mask, labels, token_idx = batch

        loss, logits = self(inputs_ids, attention_mask, labels)

        self.test_y_hat.extend(pred[idx].tolist() for pred, idx in zip(torch.argmax(logits, -1).detach().cpu(), token_idx))
        self.test_y.extend(gold[idx].tolist() for gold, idx in zip(labels.detach().cpu(), token_idx))
        self.test_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def test_epoch_end(self, outputs):
        mean_test_loss = sum(self.test_loss) / len(self.test_loss)
        gold = [self.label_encoder.inverse_transform(labels) for labels in self.test_y]
        pred = [self.label_encoder.inverse_transform(labels) for labels in self.test_y_hat]

        evaluator = Evaluator(gold, pred, tags=self.tags, loader="list")

        results, results_by_tag = evaluator.evaluate()
        self.log("test/avg_loss", mean_test_loss, prog_bar=True)
        self.log("test/ent_type", results["ent_type"]["f1"])
        self.log("test/partial", results["partial"]["f1"])
        self.log("test/strict", results["strict"]["f1"])
        self.log("test/exact", results["exact"]["f1"])

        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)


class MyDataset(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.instances = []
        print("Reading corpus: {}".format(file_path))

        assert os.path.isfile(file_path)

        with open(file_path, "r", encoding="utf8") as f:
            list_tokens = []
            list_labels = []

            for line in f:
                if not line.startswith("#"):
                    if line != "\n":
                        tokens = line.split("\t")

                        token = tokens[1]
                        label = tokens[-1].replace("\n", "")

                        list_tokens.append(token)
                        list_labels.append(label)
                    else:
                        assert len(list_tokens) == len(list_labels) and list_tokens != 0

                        instance = {
                            "tokens": list_tokens,
                            "labels": list_labels
                        }

                        self.instances.append(instance)

                        list_tokens = []
                        list_labels = []

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i]


def load_label_encoder(file_path):
    print("\nLoading labels encoder...")
    labels = set([args.pad_label])

    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            if not line.startswith("#"):
                if line != "\n":
                    tokens = line.split("\t")

                    label = tokens[-1].replace("\n", "")
                    labels.add(label)

    encoder = LabelEncoder()
    encoder.fit(list(labels))

    assert len(encoder.classes_) == 34
    print("Encoder trained with the following classes: \n{}\n".format(encoder.classes_))

    return encoder


class MyCollator(object):
    def __init__(self, tokenizer, label_encoder, pad_label):
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.pad_label = pad_label
        self.pad_label_id = self.label_encoder.transform([self.pad_label])[0]

    def __call__(self, batch):
        batch_inputs_ids = []
        batch_attention = []
        batch_labels = []
        batch_token_idx = []

        for instance in batch:
            list_tokens = []
            list_labels = []
            list_token_idx = [1]

            for token, label in zip(instance["tokens"], instance["labels"]):
                subtokens = self.tokenizer.encode(token, add_special_tokens=False)
                sublabels = self.label_encoder.transform([label]).tolist() * len(subtokens)

                list_tokens += subtokens
                list_labels += sublabels
                list_token_idx += [list_token_idx[-1] + len(subtokens)]

            batch_inputs_ids.append(torch.tensor([self.tokenizer.cls_token_id] + list_tokens[:510] + [self.tokenizer.sep_token_id]))
            batch_attention.append(torch.tensor([1] * (len(list_tokens) + 2)))
            batch_labels.append(torch.tensor([self.pad_label_id] + list_labels[:510] + [self.pad_label_id]))
            batch_token_idx.append(list_token_idx[:-1])

            assert len(batch_inputs_ids[-1]) == len(batch_attention[-1]) == len(batch_labels[-1])

        input_ids = torch.nn.utils.rnn.pad_sequence(batch_inputs_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(batch_attention, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=self.pad_label_id)

        return input_ids, attention_mask, labels, batch_token_idx


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--model_name', type=str,
                        default="dumitrescustefan/bert-base-romanian-cased-v1")  # xlm-roberta-base
    parser.add_argument("--data_path", type=str, default="../ronec/conllup/raw")
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--pad_label', type=str, default="<PAD>")
    parser.add_argument('--experiment_iterations', type=int, default=1)
    args = parser.parse_args()

    print("Batch size is {}, accumulate grad batches is {}, final batch_size is {}\n".format(
        args.batch_size,
        args.accumulate_grad_batches,
        args.batch_size * args.accumulate_grad_batches)
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    label_encoder = load_label_encoder(os.path.join(args.data_path, "train.conllu"))

    print("Loading data...")
    train_dataset = MyDataset(os.path.join(args.data_path, "train.conllu"))
    val_dataset = MyDataset(os.path.join(args.data_path, "dev.conllu"))
    test_dataset = MyDataset(os.path.join(args.data_path, "test.conllu"))

    my_collator = MyCollator(tokenizer=tokenizer, label_encoder=label_encoder, pad_label=args.pad_label)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                  collate_fn=my_collator, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False,
                                collate_fn=my_collator, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False,
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

    while itt < args.experiment_iterations:
        print("Running experiment {}/{}".format(itt + 1, args.experiment_iterations))

        model = TransformerModel(
            model_name=args.model_name,
            num_labels=len(label_encoder.classes_),
            lr=args.lr,
            model_max_length=args.model_max_length,
            label_encoder=label_encoder,
            tags=label_encoder.classes_
        )

        early_stop = EarlyStopping(
            monitor='valid/exact',
            patience=5,
            verbose=True,
            mode='max'
        )

        trainer = pl.Trainer(
            gpus=args.gpus,
            callbacks=[early_stop],
            # limit_train_batches=5,
            # limit_val_batches=2,
            accumulate_grad_batches=args.accumulate_grad_batches,
            gradient_clip_val=1.0,
            checkpoint_callback=False
        )
        trainer.fit(model, train_dataloader, val_dataloader)

        result_valid = trainer.test(model, val_dataloader)
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

    print("Done, writing results...")

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

    with open("results_ronec_of_{}.json".format(args.model_name.replace("/", "_")), "w") as f:
        json.dump(result, f, indent=4, sort_keys=True)

    print(result)
