import sys
sys.path.append("../")
import copy
import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from dgl.contrib.data import load_data
import numpy as np
from RGCN.model import Model
from server import Server
from client import Client
import argparse
import json
import logging
import os
import random


# init directories
def init_dir(args):
    # logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    # state
    if not os.path.exists(args.state_dir):
        os.makedirs(args.state_dir)
    if not os.path.exists(args.state_dir + args.run_mode + '/'):
        os.makedirs(args.state_dir + args.run_mode + '/')

    # tensorboard log
    if not os.path.exists(args.tb_log_dir):
        os.makedirs(args.tb_log_dir)


# init logger
def init_logger(args):
    log_file = os.path.join(args.log_dir, args.run_mode + '.log')

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode='a+'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# init FL setting
def init_fed(args):
    # load data
    data = load_data(dataset=args.dataset)
    # keep the same number of node and relation to each client 
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    num_classes = data.num_classes

    # create sub graph for each client
    # A. Extract the subgraph
    # client_edge_num = len(data.edge_type) // args.num_client
    # edge_num_list = [client_edge_num] * args.num_client
    # if client_edge_num * args.num_client != len(data.edge_type):
    #     edge_num_list[0] = client_edge_num + len(data.edge_type) - client_edge_num * args.num_client
    # edge_perms = random_split(range(len(data.edge_type)), edge_num_list)

    # B. Extract the subgraph
    edge_perms = []
    for _ in range(args.num_client):
        perm = np.random.permutation(len(data.edge_type))[:int(len(data.edge_type)*args.labeled_rate)]
        edge_perms.append(perm)

    datasets = []
    for perm in edge_perms:
        client_data = copy.deepcopy(data)

        train_idx = client_data.train_idx
        np.random.shuffle(train_idx)
        client_data.train_idx = train_idx[:int(len(train_idx)*args.labeled_rate)]

        client_data.edge_src = client_data.edge_src[perm]
        client_data.edge_dst = client_data.edge_dst[perm]
        client_data.edge_type = client_data.edge_type[perm]
        client_data.edge_norm = client_data.edge_norm[perm]

        norm_matrix = np.zeros((num_nodes, num_rels))
        for i in range(len(perm)):
            norm_matrix[client_data.edge_dst[i]][client_data.edge_type[i]] += 1
        
        for i in range(len(perm)):
            client_data.edge_norm[i] = 1 / norm_matrix[client_data.edge_dst[i]][client_data.edge_type[i]]

        datasets.append(client_data)

    # init the roles of FL
    # 1.init server
    ser_model = Model(num_nodes, args.n_hidden, num_classes, num_rels, args.n_bases, args.n_hidden_layers, args.gpu)
    server = Server(ser_model, args.gpu, logging, args.writer)
    # 2.init clients
    clients = []
    for i in range(args.num_client):
        model = Model(num_nodes, args.n_hidden, num_classes, num_rels, args.n_bases, args.n_hidden_layers, args.gpu)
        client = Client(i, datasets[i], model, args.gpu, args.local_epoch, args.lr, args.l2norm, args.state_dir + args.run_mode + "/", logging, args.writer)
        clients.append(client)
    
    return server, clients


# FL process
def FedRunning(args, server, clients):

    for t in range(args.round):

        logging.info(f"---------------------Round {t}---------------------")

        # The 0 step
        perm = list(range(args.num_client))
        random.shuffle(perm)
        perm = np.array(perm[:int(args.num_client * args.fraction)])

        # The 1 step
        for client in np.array(clients)[perm]:
            client.getParame(*server.sendParame())
            client.train()
            server.getParame(*client.uploadParame())
        
        # The 2 step
        server.aggregate()
        # server.test(test_dataloader)

    
    logging.info(f"--------------------Finally Test--------------------")
    test_acc = 0
    for client in np.array(clients):
        client.getParame(*server.sendParame())
        test_acc += client.test()
    test_acc /= len(clients)
    logging.info("Clients Test Avg Acc: {:>8f}".format(test_acc))


# Single client run on local device
def SingleRunning(args, server, clients):

    for client in np.array(clients):
        client.getParame(*server.sendParame())

    for t in range(args.round):

        logging.info(f"---------------------Round {t}---------------------")
        for client in np.array(clients):
            client.train()
    
    logging.info(f"--------------------Finally Test--------------------")
    test_acc = 0
    for client in np.array(clients):
        test_acc += client.test()
    test_acc /= len(clients)
    logging.info("Clients Test Avg Acc: {:>8f}".format(test_acc))


# Entire data will be run together
def EntireRunning(args):

    args.labeled_rate = 1.0
    args.num_client = 1

    # init all data on one client
    server, clients = init_fed(args)
    for t in range(args.round):

        logging.info(f"---------------------Round {t}---------------------")
        clients[0].train()
    
    logging.info(f"--------------------Finally Test--------------------")
    clients[0].test()
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data and path setting
    parser.add_argument('--dataset', default='aifb', type=str)

    parser.add_argument('--state_dir', '-state_dir', default='../log/state/', type=str)
    parser.add_argument('--log_dir', '-log_dir', default='../log/', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='../log/tb_log/', type=str)
    parser.add_argument('--labeled_rate', default=0.7, type=float)

    # one task hyperparam
    parser.add_argument('--lr', default=1e-2, type=int)
    parser.add_argument('--l2norm', default=0, type=float, help="L2 norm coefficient")
    parser.add_argument('--gpu', default='0', type=str, help="running on this device")
    parser.add_argument('--num_cpu', default=1, type=int, help="number of cpu")
    parser.add_argument('--run_mode', default='Fed', choices=['Fed',
                                                                    'Single',
                                                                    'Entire'], type=str)

    # for Fed-RGCN
    parser.add_argument('--num_client', default=10, type=int, help="number of clients")
    parser.add_argument('--fraction', default=1, type=float, help="fractional clients")
    parser.add_argument('--round', default=10, type=int, help="federated learning rounds")
    parser.add_argument('--local_epoch', default=5, help="local epochs to train")
    parser.add_argument('--n_hidden', default=16, type=float, help="number of hidden units")
    parser.add_argument('--n_bases', default=50, type=int, help="use number of relations as number of bases")
    parser.add_argument('--n_hidden_layers', default=2, type=int, help="use 1 input layer, 1 output layer, n hidden layers")

    # for random 
    parser.add_argument('--seed', default=12345, type=int)

    args = parser.parse_args()
    args_str = json.dumps(vars(args))

    args.gpu = torch.device('cuda:' + args.gpu)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # create directories
    init_dir(args)

    # init writer
    writer = SummaryWriter(args.tb_log_dir + args.run_mode + "/")
    args.writer = writer

    # init logger
    init_logger(args)
    logging.info(args_str) 

    if args.run_mode == 'Fed':
        # init FL setting
        server, clients = init_fed(args)
        # running
        FedRunning(args, server, clients)
    elif args.run_mode == 'Single':
        # init FL setting
        server, clients = init_fed(args)
        # running
        SingleRunning(args, server, clients)
    elif args.run_mode == 'Entire':
        EntireRunning(args)
