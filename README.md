# HyperS2V
Codes for HyperS2V.

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

# Paper:
