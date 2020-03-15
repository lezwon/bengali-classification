import pretrainedmodels
import logging
import torch.nn.functional as F
from torch import nn
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

class ShutdownInstanceCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        print('Shutting down instance')
        # client.stop_notebook_instance(NotebookInstanceName = instance_name)

class Resnet34(pl.LightningModule):
    def __init__(self, pretrained=True, **kwargs):
        super(Resnet34, self).__init__()
        self.__dict__.update(kwargs)
        
        self.setup_logging()

        model_name = 'resnet34'
        if pretrained is True:
            self.model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__[model_name](pretrained=None)
            
        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)
        
    def setup_logging(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename='train.log',level=logging.INFO)
        
    def forward(self, x):
        bs, c, h, w = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2


    def prepare_data(self):
        self.train_dataset = BengaliDataset(
            folds=self.TRAINING_FOLDS,
            img_height=self.IMG_HEIGHT,
            img_width=self.IMG_WIDTH,
            mean=self.MODEL_MEAN,
            std=self.MODEL_STD
        )

        self.validation_dataset = BengaliDataset(
            folds=self.VALIDATION_FOLDS,
            img_height=self.IMG_HEIGHT,
            img_width=self.IMG_WIDTH,
            mean=self.MODEL_MEAN,
            std=self.MODEL_STD
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.3, verbose=True)
        return [optimizer], [scheduler]
    
    
    def loss_fn(self, outputs, targets):
        o1, o2, o3 = outputs
        t1, t2, t3 = targets

        l1 = nn.CrossEntropyLoss()(o1, t1)
        l2 = nn.CrossEntropyLoss()(o2, t2)
        l3 = nn.CrossEntropyLoss()(o3, t3)
        return (l1 + l2 + l3) / 3

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )

    def training_step(self, batch, batch_idx):
        logging.info(f'TRAINING BATCH: {batch_idx}')
        image = batch['image']
        grapheme_root = batch['grapheme_root']
        vowel_diacritic = batch['vowel_diacritic']
        consonant_diacritic = batch['consonant_diacritic']

        outputs = self.forward(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = self.loss_fn(outputs, targets)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.TEST_BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )

    def validation_step(self, batch, batch_idx):
        logging.info(f'VALIDATION BATCH: {batch_idx}')
        image = batch['image']
        grapheme_root = batch['grapheme_root']
        vowel_diacritic = batch['vowel_diacritic']
        consonant_diacritic = batch['consonant_diacritic']

        outputs = self.forward(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        
        loss = self.loss_fn(outputs, targets)
        logs = {'loss': loss}
        return {'val_loss': loss, 'log': logs}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        logging.info(f'VAL_LOSS:{val_loss_mean}')
        return {'val_loss': val_loss_mean}


model = Resnet34(
    pretrained=True,
    DEVICE = "cuda",
    CUDA_VISIBILE_DEVICES=0,
    IMG_HEIGHT=137,
    IMG_WIDTH=236,
    EPOCHS=20,
    TRAIN_BATCH_SIZE=256,
    TEST_BATCH_SIZE=64,
    MODEL_MEAN=(0.485, 0.456, 0.406),
    MODEL_STD=(0.229, 0.224, 0.225),
    TRAINING_FOLDS=(0, 1, 2, 3),
    VALIDATION_FOLDS=(4, ),
    BASE_MODEL='bengali'
)

trainer = pl.Trainer(
    early_stop_callback=True, 
    gpus=1,
    )

trainer.fit(model)