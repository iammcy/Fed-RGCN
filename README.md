# Fed-RGCN
Pytorch-based and DGL-based implementation of Relational Graph Convolutional Networks via federated learning for Node Classification

## Dependencies
- Python 3.6+
- [PyTorch](http://pytorch.org/) 1.0+
- [DGL](https://www.dgl.ai/) 0.6+

## Results
The results of **Entire-RGCN**, **Single-RGCN** and the baseline FL **Fed-RGCN** on **AIFB** are as follows.
| | Test Acc |
|:----------:|:----------:|
| Entire | 0.8611 |
| Single | 0.3583 |
| Fed    | 0.4166 |
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
