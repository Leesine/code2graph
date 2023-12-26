# [本项目代码是DSHGT论文的实现](https://arxiv.org/pdf/2306.01376.pdf)

# 安装依赖

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

# 程序代码说明

- config/:  配置文件，数据位置和模型参数、运行参数
- data_process/:  数据处理类，通过joern整个文件生成cpg，然后拆分成function级别的cpg，归一化和向量化等。
- dataset/: 自定义数据集相关类，包括CPGDataset
- model/: HGT模型实现类
- get_metadata.py: 获取所有节点和边类型，作为异构图输入
- metrics.py：用于计算各类指标
- run.py 程序训练启动入口
- tran.py: 启动模型训练
- utils.py：各种工具类

# 代码运行过程

1. 运行preprocess_source_code：通过joern_parse将源代码转化为cpg.bin，然后joern_export将cpg导出到export.dot
2. 运行split_export_dot: 将每个源文件的export.dot文件，通过前向后向遍历算法得到每个方法的dot格式的cpg文件
3. 运行data_generator: 读取每个方法的cpg文件，设置networkx图相关属性
4. 运行dataset_generator：通过symbolizer将源代码归一化，并且划分训练集、验证集、测试集
5. 运行word_embedding：获取源代码的embedding作为图初始化的向量
6. 运行run：程序运行