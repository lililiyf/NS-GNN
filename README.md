# N&S-GNN

基于节点表示和结构表示建立了一个图神经网络预训练模型

A  graph neural network pre-training model based on "node representation" and "structure representation"

Author: LiYafen

由于个人时间原因，本项目本应投稿期刊但被搁置，因而公开在github上，供大家交流。

# 模型介绍
在图神经网络领域，预训练可以解决图数据标记稀缺的问题，对图神经网络的应用有很大促进作用。然而目前图神经网络预训练相关研究较少，现有的预训练模型无法很好的抓住图中通用特征。在本文中，我探究了图中通用结构特征的表示，提出了“结构表示”的概念和量化方法，旨在通过“结构表示”抓住图中较为通用的结构特征。经实验验证，在预训练模型中增加“结构表示”，在一定程度上会增益模型性能。以“节点表示+结构表示”的结构为基础，我提出了一种新的图神经网络预训练模型N&S-GNN，面向图分类任务开展。我为模型实现了掩盖属性预测和图簇距离预测两个预训练任务，基于此，为节点表示和结构表示设计了4种预训练任务的组合，通过实验，确定出使N&S-GNN模型性能表现最好的预训练任务组合为“Cluster Distance + Cluster Distance”组合。最后，我对各种GNN模型的预训练进行了系统的实证研究。实验结果表明，N&S-GNN模型在众多传统的预训练模型中表现优异，一定程度上抓住了图中较为通用的结构特征。

![image](https://user-images.githubusercontent.com/85494471/187849019-5f8c4021-059b-484e-afa8-f8b4d9dbbca6.png)

# 模型评估
第一轮对比，重点对不同预训练任务组合下的模型性能进行评估。本文在生物领域的数据集上完成了对表3-1所示的四种预训练任务组合下的N&S-GNN模型的评估，结果如表4-2所示。预训练任务组合“Cluster Distance + Cluster Distance”使N&S-GNN模型性能表现最好，在下游任务上准确率最高。这说明了为节点表示和结构表示都设计Cluster Distance预训练任务会使得模型学习到更通用的节点表示和结构表示信息。

表1 预训练任务设计组合

<img width="571" alt="image" src="https://user-images.githubusercontent.com/85494471/187849918-f895c4ba-4f93-4461-829a-efaa34cb1ab5.png">


第二轮对比，重点对“节点表示+结构表示”的结构进行了性能评估。第二轮对比在生物领域的数据集上完成。实际上，N&S-GNN模型经过了两个预训练任务，分别训练节点表示和结构表示，而传统模型只进行了一个预训练任务，训练节点表示。为了排除预训练任务对模型结构性能评估的影响，本文设置了表4-3所示的对比实验，让传统模型和当前模型都经过相同的预训练任务，然后评估两个模型的准确率。经实验验证，本文发现当前模型的准确率比传统模型的准确率都高，在Masking + Cluster Distance组合上提升效果较为明显，高出3%，在Cluster Distance + Cluster Distance上略有提高，高出0.3%。这在一定程度上说明了，以“节点表示+结构表示”为结构的模型性能优于以“节点表示”为结构的模型。

表2 生物领域有无“结构表示”模型（预训练）结果比对

<img width="530" alt="image" src="https://user-images.githubusercontent.com/85494471/187849915-9fa3cc01-063f-454b-b5aa-797b0aee8481.png">


表3 生物领域有无“结构表示”模型（无预训练）结果比对

<img width="543" alt="image" src="https://user-images.githubusercontent.com/85494471/187849916-674724b0-084e-4f03-a4ae-179001e87eaa.png">

为了进一步验证这个结论，本文又做了表4-4的对比实验。这一次不进行预训练，直接让“节点表示”的模型和“节点表示+结构表示”的模型在下游任务上进行训练，然后评估两个模型的准确率。经实验发现，当前模型的准确率还是高于传统模型的，当前模型的准确率高出了2%。这从另一个角度说明了，以“节点表示+结构表示”为结构的模型性能确实优于以“节点表示”为结构的模型。

至此，本文从两个角度上完成了对“节点表示+结构表示”的结构的性能的评估，两个角度上都证明“节点表示+结构表示”的结构的性能优于“节点表示”的结构的性能。这是一个可喜可贺的结论！

另外，将表2与表3进行对比，可以发现，以Cluster Distance + Cluster Distance为预训练任务的N&S-GNN模型在下游任务上的性能表现比直接在下游任务上进行训练的两个模型准确率都要高，这说明了N&S-GNN预训练模型打败了直接训练的两个模型，证明了N&S-GNN预训练模型的可迁移性很强！
第三轮对比，用了两个领域的数据集：在生物领域完成了同一数据集下的不同任务，经测试，N&S-GNN预训练模型在众多预训练模型中的表现优异，当前模型在下游任务上的准确率均高于传统Masking模型、Edgepred模型、Contextpred模型和Infomax模型。这在一定程度上证明了N&S-GNN模型的有效性；在化学领域上，要完成不同数据集下的不同任务，训练难度和对模型的要求高于生物领域的数据集上验证，N&S-GNN模型表现仍然领先。生物领域结果如表4所示，化学领域结果如表5所示。

表4 生物领域预训练模型结果比对

<img width="296" alt="image" src="https://user-images.githubusercontent.com/85494471/187850131-a3b8f80b-9c09-46cb-bdbd-f19392b4d2e8.png">

表5 化学领域预训练模型结果比对

<img width="583" alt="image" src="https://user-images.githubusercontent.com/85494471/187850175-52a31c75-6455-4358-8bcb-0b68c4b71491.png">


# 结论

本文基于传统模型的嵌入表示“节点表示”和“图表示”，提出了“结构表示”的概念，用“结构表示”学习图中节点周围的子图的结构信息。在预训练模型中增加“结构表示”，能使下游任务的模型性能比只有“节点表示”的模型提高0.3-3%的准确率；在不做预训练的模型中，在下游任务模型中增加“结构表示”也会提升2%的准确率。
基于“节点表示+结构表示”的结构，本文提出了一种新的图神经网络预训练模型N&S-GNN，面向图分类任务开展。我们为模型实现了掩盖属性预测和图簇距离预测两个预训练任务，基于此，为节点表示和结构表示设计了4种预训练任务的组合，通过实验，确定出使N&S-GNN模型性能表现最好的预训练任务组合为“Cluster Distance + Cluster Distance”组合。

将N&S预训练模型与基准模型进行对比，模型们在生物领域数据集完成同一数据集下的不同的图分类任务，在化学领域数据集上完成不同数据集下的不同的图分类任务。在两类任务中，N&S-GNN模型与众多基准模型中均表现优异。可以说，N&S-GNN预训练模型，较好的抓住了图中较为通用的结构特征，完成了本文的预期任务。

# 参考文献

[1]Hu W, Liu B, Gomes J, et al. Strategies for pre-training graph neural networks[J]. arXiv preprint arXiv:1905.12265, 2019.

[2]Qiu J, Chen Q, Dong Y, et al. Gcc: Graph contrastive coding for graph neural network pre-training[C]//Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020: 1150-1160.

[3]Scarselli F, Gori M, Tsoi A C, et al. The graph neural network model[J]. IEEE transactions on neural networks, 2008, 20(1): 61-80.
