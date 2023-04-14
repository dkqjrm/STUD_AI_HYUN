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

    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument('--accumulate', default=1, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument("--epoch", default=20, type=int)

    parser.add_argument('--devices', nargs='+', type=int, default=[0], help='list of device ids')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)

    args = vars(parser.parse_args())

    fix_seed(args["seed"])

    model = Classifier(config=args)

    logger = TensorBoardLogger(save_dir="./Studs", name="lr{}_batch{}_ac{}_epoch{}_pat{}".format(args["learning_rate"], args["batch_size"], args["accumulate"], args["epoch"], args["patience"]))

    logger.log_hyperparams(args)

    early_stopping = EarlyStopping("val_epoch_f1", patience=args["patience"], mode='max')

    checkpoint = ModelCheckpoint(dirpath="./output",
                                 filename="{epoch}_{val_epoch_f1:.2f}_" + "lr{}_batch{}".format(args["learning_rate"],
                                                                                                args["batch_size"]),
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
