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
import pandas
from matplotlib import pyplot
from paddle import nn
from paddle.io import Dataset, DataLoader
from paddle.nn import functional

feature = 50
hidden = (20, 15, 16, 12)

eerie = "0.77078 0.54875 -0.57226 -0.022805 0.58387 -0.19071 0.58289 0.4308 -0.6842 0.85256 -0.34181 -0.56369 0.13379 0.25333 -0.19371 0.11782 -0.50497 -0.2021 -0.78906 -0.81052 -0.35362 0.97211 -0.24189 -0.45017 0.96672 0.33294 -1.9523 1.6079 1.1893 -0.043454 0.50027 -0.4198 0.83259 -1.0258 -0.040808 -0.11181 0.1205 -0.96936 -0.22799 -0.096723 0.035656 0.14521 -0.55703 -0.23898 -0.162 0.3873 1.0396 -0.36315 -0.50369 -0.14816"
eerie = [float(x) for x in eerie.split()]
eerie = paddle.to_tensor([[eerie]])


def main():
	load("model.pdparams")


def load(file):
	words = pandas.read_csv("data/Problem_C_Data_Wordle.csv")["Word"].values
	trainData, testData = WordsDataset.get("data/vecs.txt")
	
	layerStateDict = paddle.load(file)
	model = LSTM(feature, hidden)
	
	model.set_state_dict(layerStateDict)
	ts = paddle.to_tensor([0.]*7)
	for i in range(len(testData)):
		x, y = testData[i]
		s = model(paddle.to_tensor([x]))
		print(s - y)
		ts += (s - y).abs()
		width = 0.2
		a = ["1 try", "2 tries", "3 tries", "4 tries", "5 tries", "6 tries",
		     "7 or more tries (X)"]
		pandas.DataFrame({
      "True value": y.numpy(),
      "Predicted value": s.numpy(),
		}, index=a).plot.bar(fontsize=10)
		pyplot.title(words[i + 300])
		pyplot.ylim(0)
		pyplot.xticks(rotation=10)
		# pyplot.bar(b[0], s.numpy(), width)
		# pyplot.bar(b[1], y.numpy(), width)
		# pyplot.show()
	print(ts/59)


def trainAndSave():
	trainData, testData = WordsDataset.get("data/vecs.txt")
	
	model = LSTM(feature, hidden)
	train(model, 39, trainData, testData)
	
	s = model(eerie)
	print(s)
	print(s.sum())
	
	c = int(input("save?"))
	if c:
		paddle.save(model.state_dict(), f"model{c}.pdparams")


def train(model, epochs, trainData, testData):
	trainLoader = DataLoader(trainData, batch_size=3)
	testLoader = DataLoader(testData, batch_size=59)
	
	optimizer = paddle.optimizer.SGD(0.01, model.parameters())
	loss = paddle.nn.MSELoss()
	
	for epoch in range(epochs):
		model.train()
		for batch, (x, y) in enumerate(trainLoader):
			prediction = model(x)
			prediction = prediction.squeeze(-1)
			l = loss(prediction, y)
			# if loss.data <= 0.06:
			# 	break
			optimizer.clear_grad()  # 消除梯度
			l.backward()  # 反向传播
			optimizer.step()  # 执行
		model.eval()
		for batch, (x, y) in enumerate(testLoader):
			l = loss(model(x), y)
			print(f"epoch {epoch}, loss {l.numpy()[0]:f}")


class LSTM(nn.Layer):
	def __init__(self, featureSize, hiddenSize):
		super(LSTM, self).__init__()
		self.liner0 = nn.Linear(featureSize, hiddenSize[0])
		self.lstm = nn.LSTM(hiddenSize[0], hiddenSize[1])
		self.gru = nn.GRU(hiddenSize[1], hiddenSize[2])
		self.liner1 = nn.Linear(hiddenSize[2], hiddenSize[3])
		self.liner2 = nn.Linear(hiddenSize[3], 7)
	
	def forward(self, x):
		x = self.liner0(x)
		x = functional.relu(x)
		_, (x, _) = self.lstm(x)
		x = self.gru(x)
		x = paddle.squeeze(x, axis=(0, 1))
		x = self.liner1(x)
		x = self.liner2(x)
		return x


class WordsDataset(Dataset):
	
	def __init__(self, dataSet):
		super(WordsDataset, self).__init__()
		self.dataSet = dataSet
	
	def __getitem__(self, item):
		data, label = self.dataSet[item]
		
		return paddle.to_tensor(data, "float32"), paddle.to_tensor(label, "float32")
	
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
				datas.append([[data], label])
		return WordsDataset(datas[: 300]), WordsDataset(datas[300:])


if __name__ == '__main__':
	main()
