1.模型准确率：
	测试集的准确率为：0.9061， 独立验证集的准确率为：0.9023
2.文件夹信息说明：
	data文件夹:
		available_data：模型训练和测试用的数据的文件夹；
		original_data: 原始文件；
	model_structure文件夹：
		模型结构：将“Resnet152”最后的全连接层，替换为输出通道数分别为256，64，5的三层全连接层；
	result文件夹：
		模型权重；
	Dataset_partition.ipynb：划分训练集，测试集，独立验证集的代码；
	Predict.py: 验证模型在测试集和独立验证集的准确率的代码；
	Train.ipynb: 模型训练的代码。
3.预训练模型信息：
	预训练模型来自于 torchvision.models.resnet152(pretrained=True), 可参考：https://pytorch.org/vision/stable/generated/torchvision.models.resnet152.html?highlight=resnet152#torchvision.models.resnet152
4.备注：
	在加载训练好的模型权重来复现测试集和独立验证集的结果的时候（Predict.py），发现复现的结果和原始的结果略有差异，而且每次运行的结果都略有不同，比如第一次是0.905，第二次0.907，第三次0.902，后经查找，发现是模型中某些地方可能存在随机性的问题，而且这种随机性不能被model.eval()消除，但是目前我还没有找到这种随机性究竟发生在哪里。为了确保每次运行的结果一致性，所以在Predict.py中加入了随机数种子来控制这种随机性（torch.manual_seed(2022)），得到最终的复现结果为测试集的准确率为：0.9061， 独立验证集的准确率为：0.9023，但是这个结果和模型训练过程输出的结果还是略有差异（模型训练过程中输出的测试集的准确率为：0.9053， 独立验证集的准确率为：0.9073）。

