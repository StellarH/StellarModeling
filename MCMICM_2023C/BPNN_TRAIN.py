# 
# Copyright (c) 2023 BlazingStellar
# 
# This program and the accompanying materials are made available 
# under the terms of the 
#      Eclipse Public License 2.0 
# which is available at 
#      http://www.eclipse.org/legal/epl-2.0 
# 
# SPDX-License-Identifier: EPL-2.0
# ****************************************************************************

import numpy
import paddle
from paddle.io import Dataset, DataLoader
from paddle.nn import functional

eerie = "0.77078 0.54875 -0.57226 -0.022805 0.58387 -0.19071 0.58289 0.4308 -0.6842 0.85256 -0.34181 -0.56369 0.13379 0.25333 -0.19371 0.11782 -0.50497 -0.2021 -0.78906 -0.81052 -0.35362 0.97211 -0.24189 -0.45017 0.96672 0.33294 -1.9523 1.6079 1.1893 -0.043454 0.50027 -0.4198 0.83259 -1.0258 -0.040808 -0.11181 0.1205 -0.96936 -0.22799 -0.096723 0.035656 0.14521 -0.55703 -0.23898 -0.162 0.3873 1.0396 -0.36315 -0.50369 -0.14816"
eerie = [float(x) for x in eerie.split()]
eerie = paddle.to_tensor(eerie)

def main():
	feature = 50
	hidden = 28
	output = 7
	
	trainData, evalData = WordsDataset.get("data/vecs.txt")
	
	model = Net(feature, hidden, output)
	
	# paddle.summary(model, input_size=(None, seqLen), dtypes='int64')
	
	train(model, 20, trainData, evalData)
	s = model(eerie)
	print(s)
	print(s.sum())
	

class Net(paddle.nn.Layer):
	
	def __init__(self, feature, hidden, output):
		super(Net, self).__init__()
		self.feature = paddle.nn.Linear(feature, hidden)
		self.predict = paddle.nn.Linear(hidden, output)
	
	def forward(self, x):
		x = functional.relu(self.feature(x))
		x = self.predict(x)
		return x


def train(model, epochs, trainData, evalData):
	trainLoader = DataLoader(trainData, batch_size=5, shuffle=True)
	evalLoader = DataLoader(evalData, batch_size=59, shuffle=True)
	
	optimizer = paddle.optimizer.SGD(parameters=model.parameters())
	loss  = paddle.nn.MSELoss()
	
	for epoch in range(epochs):
		model.train()
		for batch, (x, y) in enumerate(trainLoader):
			prediction = model(x)
			prediction = prediction.squeeze(-1)
			l = loss(prediction, y)
			optimizer.clear_grad()  # 消除梯度
			l.backward()  # 反向传播
			optimizer.step()  # 执行
			# print(f"epoch: {epoch}, train loss: {l.numpy()[0]:f}")
		model.eval()
		for batch, (x, y) in enumerate(evalLoader):
			prediction = model(x)
			prediction = prediction.squeeze(-1)
			l = loss(prediction, y)
			print(f"epoch: {epoch}, loss: {l.numpy()[0]:f}")


class WordsDataset(Dataset):
	
	def __init__(self, dataSet):
		super(WordsDataset, self).__init__()
		self.dataSet = dataSet
	
	def __getitem__(self, item):
		data, label = self.dataSet[item]
		
		return paddle.to_tensor(data), paddle.to_tensor(label)
	
	def __len__(self):
		return len(self.dataSet)
	
	@staticmethod
	def get(file):
		datas = []
		with open(file, "r") as f:
			for line in f:
				label, data = line.split("\t")
				label = [float(x) for x in label.strip().split()]
				data = [float(x) for x in data.strip().split()]
				datas.append([data, label])
		return WordsDataset(datas[: 300]), WordsDataset(datas[300:])


if __name__ == '__main__':
	main()
