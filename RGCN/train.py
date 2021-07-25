from dgl.contrib.data import load_data
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from model import Model

# configurations
n_hidden = 16 # number of hidden units
n_bases = 3 # use number of relations as number of bases
n_hidden_layers = 2 # use 1 input layer, 1 output layer, no hidden layer
n_epochs = 25 # epochs to train
lr = 0.01 # learning rate
l2norm = 0 # L2 norm coefficient


data = load_data(dataset='aifb')
num_nodes = data.num_nodes
num_rels = data.num_rels
num_classes = data.num_classes
labels = data.labels
train_idx = data.train_idx
val_idx = train_idx[:len(train_idx) // 5]
train_idx = train_idx[len(train_idx) // 5:]

edge_type = torch.from_numpy(data.edge_type)
edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1)

labels = torch.from_numpy(labels).view(-1)


# create graph
g = DGLGraph((data.edge_src, data.edge_dst))
g.edata.update({'rel_type': edge_type, 'norm': edge_norm})

# create model
model = Model(num_nodes, n_hidden, num_classes, num_rels, n_bases, n_hidden_layers)
print(model.state_dict().keys())

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

print("start training...")
model.train()
for epoch in range(n_epochs):
    logits = model(g)
    loss = F.cross_entropy(logits[train_idx], labels[train_idx])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc = (logits[train_idx].argmax(1) == labels[train_idx]).float().sum()
    train_acc = train_acc / len(train_idx)
    val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
    val_acc =  (logits[val_idx].argmax(1) == labels[val_idx]).float().sum()
    val_acc = val_acc / len(val_idx)

    print("Epoch {:05d} | ".format(epoch) +
          "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
              train_acc, loss.item()) +
          "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
              val_acc, val_loss.item()))
