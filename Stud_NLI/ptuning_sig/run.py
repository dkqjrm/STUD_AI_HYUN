import torch
from model import Classifier
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
import numpy as np
import random
import argparse


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--pre_seq_len', type=int, default=16)
    parser.add_argument('--prefix_projection', type=bool, default=True)
    parser.add_argument('--prefix_hidden_size', type=int, default=512)
    parser.add_argument('--num_hidden_layers', type=int, default=12)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument("--prompt", default='이 학생은 자퇴할 것이다.', type=str)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument('--accumulate', default=1, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument("--epoch", default=20, type=int)

    parser.add_argument('--devices', nargs='+', type=int, default=[3], help='list of device ids')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)

    args = vars(parser.parse_args())

    fix_seed(args["seed"])


    model = Classifier(config=args)

    logger = TensorBoardLogger(save_dir="./Studs", name="lr{}_batch{}_epoch{}_pat{}_ac{}_prelen{}".format(args["learning_rate"], args["batch_size"], args['epoch'], args['patience'], args["accumulate"], args['pre_seq_len']))

    logger.log_hyperparams(args)

    early_stopping = EarlyStopping("val_epoch_f1", patience=args["patience"],mode='max')


    checkpoint = ModelCheckpoint(dirpath="./output",
                                 filename="{epoch}_{val_epoch_f1:.2f}_" + "lr{}_batch{}_prelen{}".format(args["learning_rate"], args["batch_size"], args['pre_seq_len']),
                                 monitor="val_epoch_f1",
                                 mode="max")

    trainer = Trainer(accelerator="gpu",
                      devices=args["devices"],
                      max_epochs=args["epoch"],
                      accumulate_grad_batches=args['accumulate'],
                      logger=logger,
                      callbacks=[early_stopping, checkpoint],
                      log_every_n_steps=1,
                      precision='16-mixed')

    trainer.fit(model)

    loaded_model = Classifier.load_from_checkpoint(checkpoint.best_model_path)
    trainer.test(loaded_model)