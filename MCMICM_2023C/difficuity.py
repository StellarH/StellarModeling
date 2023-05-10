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

import math

import numpy
import numpy as np
import pandas
# import sympy
from matplotlib import pyplot
from scipy import optimize

with open("data/words.txt") as f:
	dic = f.read().strip().split()
words = pandas.read_csv("data/Problem_C_Data_Wordle.csv")["Word"].values


def main():
	sim()
	# difficulty()


def difficulty():
	df = pandas.read_csv("data/likely.tsv", sep="\t", index_col="word")
	# 	.merge(
	# 	pandas.read_csv("data/fixes.tsv", sep="\t"), on="word")
	# df["fixes"] = numpy.log(df["fixes"])/10
	# df["fixes"][np.isinf(df["fixes"])] = 0
	# df["likely"] += (1 - df["likely"])*df["fixes"]
	# df = df.T.drop("fixes").T
	def frequency(x):
		x = [[k, 1] for k in x]
		cnt = pandas.DataFrame(x, columns=["word", "1"]).groupby("word")["1"].count()
		if cnt.max() == 1:
			return 0.0
		elif cnt.max() == 2 and len(cnt) == 4:
			return 0.3
		elif cnt.max() == 2 and len(cnt) == 3:
			return 0.5
		return  0.7
	df["difficulty"] = [frequency(x) for x in df.index]
	df["difficulty"] = df["likely"] + (1 - df["likely"])*df["difficulty"]
	print(df[df.index == "eerie"])
	pyplot.figure(figsize=(10, 5))
	pyplot.plot(numpy.sort(df["difficulty"].values))
	pyplot.savefig("fig/difficulty.png", dpi=220)
	pyplot.show()
	0


def fix():
	with open("data/fix.txt") as f:
		p, l = f.read().strip().split("--")
	p = p.strip().split()
	p = [p[:: 2], [int(x) for x in p[1:: 2]]]
	l = l.strip().split()
	l = [l[:: 2], [int(x) for x in l[1:: 2]]]
	fall = []
	for word in words:
		val = 0
		if word[: 2] in p[0]:
			val += p[1][p[0].index(word[: 2])]
		elif word[-2:] in l[0]:
			val += l[1][l[0].index(word[-2:])]
		fall.append(val)
	pandas.DataFrame({
		'word': words,
		'likely': fall
	}).to_csv("data/fixes.tsv", sep="\t", index=False)


ret = [(math.exp(x) - 1)/1.96 for x in range(5)]


def similarity(w0, w1):
	count = 0
	for i in range(5):
		if w0[i] == w1[i]:
			count += 1
	return ret[count]


def func(x, a, b, c, d):
	return a/(b + c*numpy.exp(-x)) + d


def sim():
	all = []
	for word in words:
		val = 0
		for w in dic:
			if w != word:
				val += similarity(word, w)
		all.append(val/(len(dic) - 1))
	print(max(all), words[all.index(max(all))])
	print(min(all), words[all.index(min(all))])
	# print(all[dic.index("eerie")], "eerie")
	all = sorted(all)
	pyplot.figure(figsize=(10, 5))
	pyplot.plot(all, numpy.arange(len(all)))
	pyplot.savefig("fig/sim.png", dpi=220)
	pyplot.show()
	
	a, b, c, d = optimize.curve_fit(func, all, numpy.arange(len(all)))[0]
	print(a, b, c, d)
	
	# pandas.DataFrame({
	# 	'word' : words,
	# 	'likely': all
	# }).to_csv("data/likely.tsv", sep="\t", index=False)


if __name__ == '__main__':
	main()
