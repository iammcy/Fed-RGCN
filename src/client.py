from dgl._deprecate.graph import DGLGraph
import torch
from torch import nn, optim
from tqdm import tqdm
from dgl import DGLGraph
import copy


class Client:
    def __init__(self, client_id, data, model, device, epoch, lr, l2norm, model_path, logging, writer) -> None:
        # log setting
        self.model_path = model_path
        self.logging = logging
        self.writer = writer

        # client setting
        self.client_id = client_id
        self.device = device
        self.data = data
        self.model = model.to(self.device)
        self.epoch = epoch

        self.Epoch = -1                    # record the FL round by self
        self.round = None                  # record the mask round of FL this round from the server
        self.val_acc = None
        self.model_param = None
        
        # training setting
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2norm)

        # data setting
        # node data process
        self.labels = torch.from_numpy(data.labels).view(-1).to(self.device)
        train_idx = data.train_idx
        self.val_idx = train_idx[ : len(train_idx) // 5]
        self.train_idx = train_idx[len(train_idx) // 5 : ]
        self.test_idx = data.test_idx
        # edges data process
        edge_type = torch.from_numpy(data.edge_type)
        edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1)
        # create graph
        self.graph = DGLGraph().to(self.device)
        self.graph.add_nodes(data.num_nodes)
        self.graph.add_edges(data.edge_src, data.edge_dst)
        self.graph.edata.update({'rel_type': edge_type.to(self.device), 'norm': edge_norm.to(self.device)})


    def train(self):

        pbar = tqdm(range(self.epoch))
        self.val_acc = 0
        self.Epoch += 1
        for _ in pbar:
            # Compute prediction error
            logits = self.model(self.graph)
            loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_acc = (logits[self.train_idx].argmax(1) == self.labels[self.train_idx]).float().sum()
            train_acc = train_acc / len(self.train_idx)
            val_acc =  (logits[self.val_idx].argmax(1) == self.labels[self.val_idx]).float().sum()
            val_acc = val_acc / len(self.val_idx)

            self.val_acc = self.val_acc + val_acc

            pbar.set_description("Client {:>2} Training: Train Loss: {:.4f} | Train Acc: {:.4f} | Val Acc: {:.4f}".format(
                                self.client_id, loss.item(), train_acc, val_acc))
            self.writer.add_scalar(f"training/loss/{self.client_id}", loss.item(), self.Epoch * self.epoch + _)
            self.writer.add_scalar(f"training/acc/{self.client_id}", train_acc, self.Epoch * self.epoch + _)
            self.writer.add_scalar(f"val/acc/{self.client_id}", val_acc, self.Epoch * self.epoch + _)

        self.writer.add_embedding(logits[self.val_idx], self.labels[self.val_idx], global_step=self.Epoch, tag="clent"+str(self.client_id))

        self.model_param = self.model.state_dict()
        self.val_acc = self.val_acc / self.epoch

    def test(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.graph)
            test_loss = self.loss_fn(logits[self.test_idx], self.labels[self.test_idx])
            test_acc =  (logits[self.test_idx].argmax(1) == self.labels[self.test_idx]).float().sum()
            test_acc = test_acc / len(self.test_idx)

        self.logging.info("Client {:>2} Test: Test Loss: {:.4f} | Test Acc: {:.4f}".format(
            self.client_id, test_loss.item(), test_acc
        ))
        # save to disk
        torch.save(self.model_param, self.model_path + "client" + str(self.client_id) + '_model.ckpt')

        return test_acc
    
    # get the global model's parameters from parameter server
    def getParame(self, round, param):
        self.round = round
        if self.model_param is not None:
            for layer_name in self.model_param:
                if ("layers.0" in layer_name) or layer_name.endswith("w_comp"): 
                    param[layer_name] = copy.deepcopy(self.model_param[layer_name])
        self.model.load_state_dict(param)


    # upload the local model's parameters to parameter server
    def uploadParame(self):
        param = {}
        for layer_name in self.model_param:
            if not layer_name.endswith("w_comp") and ( "layers.0" not in layer_name):
                param[layer_name] = self.model_param[layer_name]
        return self.round, param, self.val_acc