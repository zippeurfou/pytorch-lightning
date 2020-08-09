import torch
from torch import nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.lightning import LightningModule
from torch_geometric.data import DataLoader
import torch_geometric.nn as gnn
import torch.nn.functional as F
seed_everything(0)


class BasicGNNClassifyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.nb_data = 0
        self.total_curr = 0

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        batch_targets = batch.y
        y_hat = self(batch)
        loss = self.loss(y_hat, batch_targets)
        cur_pre = self.current_nums(y_hat, batch_targets)
        self.total_curr += cur_pre
        # epoch_test_mae += MAE(batch_scores, batch_targets)
        self.nb_data += batch_targets.size(0)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        acc = self.total_curr / self.nb_data
        self.total_curr = 0
        self.nb_data = 0
        acc_log = {"acc":acc}
        results = {'progress_bar': acc_log}
        return results

    def current_nums(self,scores, targets):
        scores = scores.detach().argmax(dim=1)
        acc = (scores == targets).float().sum().item()
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def loss(self,y_pre,y_true):
        return F.cross_entropy(y_pre,y_true)


class GCN(BasicGNNClassifyModel):
    '''
        “Semi-supervised Classification with Graph Convolutional Networks”
        https://arxiv.org/abs/1609.02907

        In paper, two layer
    '''

    def __init__(self,hidden_dim,out_dim,dropout,graph_norm,batch_norm,
                 residual,in_channels,hidden_nums,readout):
        '''
          in_channels : num of node features
          out_channels: num of class
      '''
        super().__init__()
        # residual = False
        self.hidden_dim = hidden_dim
        self.out_channels = out_dim
        self.dropout_rate = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        in_channels = in_channels
        self.hidden_nums = hidden_nums
        # self.device = net_params['device']
        self.readout = readout


        self.layers = nn.ModuleList([gnn.GCNConv(in_channels, self.hidden_dim)])
        self.layers.extend([gnn.GCNConv(self.hidden_dim, self.hidden_dim)
                            for _ in range(self.hidden_nums)])
        self.layers.append(gnn.GCNConv(self.hidden_dim, self.out_channels))
        if self.batch_norm:
            self.ba_norms = nn.ModuleList([gnn.BatchNorm(self.hidden_dim)
                                           for _ in range(self.hidden_nums + 1)])
            self.ba_norms.append(
                gnn.BatchNorm(self.out_channels))

    def forward(self, data):
        x, edge_index , batch = data.x, data.edge_index,data.batch
        x = torch.cat([x,data.pos],dim=1)

        for idx in range(len(self.layers)):
            h = x
            gc_model = self.layers[idx]
            x = gc_model(x,edge_index)
            x = F.dropout(x, p=self.dropout_rate)
            if self.graph_norm:
                x = gnn.GraphSizeNorm()(x)
            if self.batch_norm:
                x = self.ba_norms[idx](x)
            x = F.relu(x)
            if self.residual:
                if len(h[0]) == len(x[0]) :
                    x = h + x
        if self.readout == "mean" :
            x = gnn.global_mean_pool(x,batch)
        out = F.log_softmax(x, dim=-1)
        return out

    def loss(self,y_pre,y_true):
        return F.cross_entropy(y_pre,y_true)
        # return F.nll_loss(y_pre,y_true)



def main():
    from torch_geometric.datasets import GNNBenchmarkDataset
    dataset = GNNBenchmarkDataset(root="AllDataSet/", name="MNIST", split="train")
    print(dataset[0])
    out_dim = dataset.num_classes
    in_dim = dataset.num_node_features + 2

    train_loader = DataLoader(dataset, batch_size=128,pin_memory=True,num_workers=8)
    #print(train_loader.collate_fn)

    model = GCN(20,out_dim,0.0,True,True,True,in_dim,2,"mean")

    trainer = Trainer(gpus=2,max_epochs=100,distributed_backend="ddp", replace_sampler_ddp=True)
    #trainer = Trainer(gpus=1,max_epochs=5)
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()