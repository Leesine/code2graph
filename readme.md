# [DSHGT](https://arxiv.org/pdf/2306.01376.pdf)

# Requirements

```
beautifulsoup4==4.12.2
gensim==4.0.1
networkx==2.5
numpy==1.23.4
omegaconf==2.2.3
pydot==1.4.2
requests==2.28.1
scikit_learn==1.3.2
torch==2.0.0+cu117
torch_geometric==2.1.0.post1
tqdm==4.64.1
wordninja==2.0.0
```

# Program Code Description
- config/: Configuration files including data locations, model parameters, and operational parameters.
- data_process/: Data processing classes that generate a CPG for the entire file using joern, then split it into function-level CPGs, normalize and vectorize, etc.
- dataset/: Custom dataset-related classes, including CPGDataset.
- model/: Implementation class for the HGT model.
- get_metadata.py: Retrieves all node and edge types as input for the heterogeneous graph.
- metrics.py: Used to calculate various metrics.
- run.py: Entry point for program training startup.
- train.py: Initiates model training.
- utils.py: utility classes.

# Code Execution Process
1. Obtain the Source Code CPG File
<br>Convert the source code into cpg.bin using joern_parse, then export the CPG to export.dot using joern_export.
```shell
cd data_process
python c_preprocess_source_code.py
```
2. Get the CPG File for Each Method
<br> Generate a dot format CPG file for each method from each source file's export.dot using a forward-backward traversal algorithm.
```shell
cd data_process
python split_export_dot.py
```
3. Set NetworkX Attributes for Each Method
<br/>Read each method's CPG file and set related NetworkX graph attributes.
```shell
cd data_process
python data_generator.py
```
4. Symbolizer and Dataset Split
<br/>Normalize the source code using the symbolizer and divide the data into training, validation, and test sets.
```shell
cd data_process
python dataset_generator.py
```
5. Obtain the Source Code Word Embedding, Initialize Graph
<br/> Obtain the embedding of the source code to initialize the graph vectors.
```shell
cd data_process
python word_embedding.py
```
6. Program Execution
<br/> Start training the HGT model, and identify code vulnerabilities through graph-level classification in the Readout layer.
```shell
python run.py
```
Please note that in the context of this translation, "CPG" stands for "Code Property Graph," which is a representation of the program structure that integrates control flow, data flow, and other semantic information. The term "Symbolizer" is also taken in its literal sense, assuming it's a tool or process used to normalize the source code.


If you find DSHGT helpful, please consider citing our work:
```
@article{10.1145/3674729,
author = {Zhang, Tiehua and Xu, Rui and Zhang, Jianping and Liu, Yuze and Chen, Xin and Yin, Jun and Zheng, Xi},
title = {DSHGT: Dual-Supervisors Heterogeneous Graph Transformer—A Pioneer Study of Using Heterogeneous Graph Learning for Detecting Software Vulnerabilities},
year = {2024},
issue_date = {November 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {33},
number = {8},
issn = {1049-331X},
url = {https://doi.org/10.1145/3674729},
doi = {10.1145/3674729},
abstract = {Vulnerability detection is a critical problem in software security and attracts growing attention both from academia and industry. Traditionally, software security is safeguarded by designated rule-based detectors that heavily rely on empirical expertise, requiring tremendous effort from software experts to generate rule repositories for large code corpus. Recent advances in deep learning, especially Graph Neural Networks (GNN), have uncovered the feasibility of automatic detection of a wide range of software vulnerabilities. However, prior learning-based works only break programs down into a sequence of word tokens for extracting contextual features of codes, or apply GNN largely on homogeneous graph representation (e.g., AST) without discerning complex types of underlying program entities (e.g., methods, variables). In this work, we are one of the first to explore heterogeneous graph representation in the form of Code Property Graph and adapt a well-known heterogeneous graph network with a dual-supervisor structure for the corresponding graph learning task. Using the prototype built, we have conducted extensive experiments on both synthetic datasets and real-world projects. Compared with the state-of-the-art baselines, the results demonstrate superior performance in vulnerability detection (average F1 improvements over 10\% in real-world projects) and language-agnostic transferability from C/C ({+}{+}) to other programming languages (average F1 improvements over 11\%).},
journal = {ACM Trans. Softw. Eng. Methodol.},
month = nov,
articleno = {202},
numpages = {31},
keywords = {Vulnerability detection, heterogeneous graph learning, code property graph (CPG)}
}
```
