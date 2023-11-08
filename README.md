# HyperS2V
Codes for HyperS2V.

# Paper:
https://arxiv.org/abs/2311.04149

# Usage:
python src/main.py --input graph/star_hyper.edgelist --num-walks 100 --walk-length 80 --window-size 5 --dimensions 2 --until-layer 5 --workers 10 --suffix star --OPT1 --OPT3 --output star_hyper

# Data format:

edgeID nodeID1 nodeID2 ...

See example: graph/star_hyper.edgelist.

# Output:

NumberOfNodes NumberOfDimension

nodeID1 dim1 dim2 ...

nodeID2 dim1 dim2 ...

...

See example: emb/star_hyper.emb.

# Citation
@misc{liu2023hypers2v,\b
      title={HyperS2V: A Framework for Structural Representation of Nodes in Hyper Networks}, 
      author={Shu Liu and Cameron Lai and Fujio Toriumi},
      year={2023},
      eprint={2311.04149},
      archivePrefix={arXiv},
      primaryClass={cs.SI}
}
