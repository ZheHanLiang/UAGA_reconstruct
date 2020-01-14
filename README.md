# UAGA_reconstruct
## Introduction
This is the **reconstructed** code for *[Unsupervised Adversarial Graph Alignment with Graph Embedding](https://arxiv.org/pdf/1907.00544.pdf)*.  
This new code optimized and simplified the [original code](https://github.com/ZheHanLiang/UAGA).
Part of iUAGA's code will be updated later.

## Major improvements
1. Providing more optional parameters and you can adjust some common parameters easily in the cmd now;
2. Providing execute.sh so you can run a series of commands automatically, which will reduce the number of operations;
3. Providing a variety of similarity calculation methods;
4. Providing complete data, which had processed;
5. The calculation process of cosine similarity is optimized, which makes the code run more efficiently;
6. Added a lot of Chinese remarks.

## Execute
* *initial_data_processing.py* : Transforming the raw data into a suitable format for code to run.
* *deepwalk* : Embedding the nodes of graph in an unsupervised fashion.
* *main.py* : The main function to run the code.

You can also run the *execute.sh* in cmd to launch the program, including the three steps above:
```
sh execute.sh
```
For more details, please refer to the [execute.sh](https://github.com/ZheHanLiang/UAGA_reconstruct/blob/master/UAGA/execute.sh).

## Dependencies
* Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](https://www.scipy.org/)
* [PyTorch](http://pytorch.org/)

## Datasets
We used three public datasets in the paper:
* [Last.fm](http://lfs.aminer.cn/lab-datasets/multi-sns/lastfm.tar.gz)
* [Flickr](http://lfs.aminer.cn/lab-datasets/multi-sns/livejournal.tar.gz)
* [MySpace](http://lfs.aminer.cn/lab-datasets/multi-sns/myspace.tar.gz)

You can also get the dataset in *--graph data*.  
For more details, please refer to [here](https://www.aminer.cn/cosnet).

### Embedding format
We utilized *DeepWalk* to learn the source and target graph embeddings in this work, so the format of input data followed the output of deepwalk. You can learn more about the detail from [here](https://github.com/phanein/deepwalk).
