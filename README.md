# Heterogeneous Graph Contrastive Learning with Meta-path Contexts and Adaptively Weighted Negative Samples 

## MEOW & AdaMEOW
This is the source code of two papers:

- $\textbf{MEOW}:$ ["Heterogeneous Graph Contrastive Learning with Meta-path Contexts and Weighted Negative Samples (SDM 2023)"](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch5)

- $\textbf{AdaMEOW}:$  ["Heterogeneous Graph Contrastive Learning with Meta-path Contexts and Adaptively Weighted Negative Samples (TKDE 2024)"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10487103).

![The proposed framework](./MEOW.png)

## Environment Settings
> python==3.9.0 \
> scipy==1.8.1 \
> torch==1.12.0 \
> numpy==1.23.0 \
> scikit_learn==1.1.1\
> faiss-gpu==1.7.2

## Usage
Go into ./code_meow, and then you can use the following commend to run our model **MEOW**;  
or go into ./code_adameow, and then you can use the following commend to run our model **AdaMEOW**: 
> python main.py acm --gpu=0

Here, "acm" can be replaced by "dblp", "aminer","imdb".

Some files in the './data' could not be uploaded because they were over 25MB. 
All the data files we store in 
url：https://pan.baidu.com/s/1vlBrC4S7EZgowGyHGF8apg 
pwd：n84e

## Citation
```
@article{yu2024heterogeneous,
  title={Heterogeneous Graph Contrastive Learning With Meta-Path Contexts and Adaptively Weighted Negative Samples},
  author={Yu, Jianxiang and Ge, Qingqing and Li, Xiang and Zhou, Aoying},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  publisher={IEEE}
}
```
