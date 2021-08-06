# Fed-RGCN
Pytorch-based and DGL-based implementation of Relational Graph Convolutional Networks via federated learning for Node Classification

## Dependencies
- Python 3.6+
- [PyTorch](http://pytorch.org/) 1.0+
- [DGL](https://www.dgl.ai/) 0.6+

## Results
### The first experiment
#### Dataset Settings
- Ratio of the number of labeled nodes on each client to the total number of labeled nodes: 70%
- Each client's local subgraph does not intersect with other clients
#### The experimental results
The results of **Entire-RGCN**, **Single-RGCN** and the baseline FL **Fed-RGCN** on **AIFB** are as follows.
| | Test Acc |
|:----------:|:----------:|
| Entire | 0.8611 |
| Single | 0.3583 |
| Fed    | 0.4166 |

### The second experiment
#### Dataset Settings
- Ratio of the number of labeled nodes on each client to the total number of labeled nodes: 70%
- The ratio of the local subgraph to the full graph for each client: 70%
#### The experimental results
The results of **Entire-RGCN**, **Single-RGCN** and the baseline FL **Fed-RGCN** on **AIFB** are as follows.
| | Test Acc |
|:----------:|:----------:|
| Entire | 0.9176 |
| Single | 0.8444 |
| Fed    | 0.8833 |

### Conclusion
Entire-RGCN > Fed-RGCN > Single-RGCN

## Running the code 
### Usage
- running Entire-RGCN
```shell
cd ./src
python main.py --run_mode=Entire
```
- running Single-RGCN
```shell
cd ./src
python main.py --run_mode=Single
```
- running Fed-RGCN
```shell
cd ./src
python main.py --run_mode=Fed
```
