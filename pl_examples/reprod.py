# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
# USE THIS MODEL TO REPRODUCE A BUG YOU REPORT
# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
import glob
import os
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import Dataset

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):

    def __init__(self):
        """
        Testing PL Module

        Use as follows:
        - subclass
        - modify the behavior for what you want

        class TestModel(BaseTestModel):
            def training_step(...):
                # do your own thing

        or:

        model = BaseTestModel()
        model.training_epoch_end = None

        """
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self.layer(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.log('loss', loss)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def validation_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.log('x', loss)

    def test_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.log('y', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


def run_test():
    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('x', loss)

    # fake data
    train_data = torch.utils.data.DataLoader(RandomDataset(32, 64))
    val_data = torch.utils.data.DataLoader(RandomDataset(32, 64))

    # model
    model = TestModel()
    tmp_dir = 'temp/'
    if os.path.exists(tmp_dir):
        os.rmdir(tmp_dir)

    trainer = Trainer(
        default_root_dir=os.getcwd(),
        max_epochs=2,
        accelerator='ddp',
        gpus=2,
        checkpoint_callback=ModelCheckpoint(
            dirpath=tmp_dir,
            monitor='x',
            mode='min',
            save_top_k=1
        )
    )
    trainer.fit(model, train_data, val_data)
    checkpoints = list(sorted(glob.glob(os.path.join(tmp_dir, "*.ckpt"), recursive=True)))
    print("checkpoints", checkpoints)
    print(trainer.checkpoint_callback.best_model_path)
    assert os.path.exists(
        trainer.checkpoint_callback.best_model_path), f'Could not find checkpoint at rank {trainer.global_rank}'


if __name__ == '__main__':
    run_test()