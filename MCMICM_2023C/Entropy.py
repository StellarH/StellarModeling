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
import pandas


def main():
	with open("data/words.txt") as f:
		words = f.read().strip().split()
	
	df = pandas.read_csv("data/words.csv", header=None)
	df = numpy.log2(df)
	df["entropy"] = df.apply(lambda x: x.mean())
	df["word"] = words
	0

#
# def main():
# 	with open("data/words.txt") as f:
# 		words = f.read().splitlines()
#
# 	entropy = []
#
# 	for i in range(len(words)):
# 		print(i, words[i])
# 		e = [words[i]]
# 		for w in words:
# 			word = list(words[i])
# 			fix = []
# 			for j in range(5):
# 				if word[j] == w[j]:
# 					fix.append(words[i][j])
# 					word[j] = '/'
# 				else:
# 					fix.append('/')
# 			may = []
# 			for j in range(5):
# 				if fix[j] == '/' and w[j] in word:
# 					may.append(w[j])
# 			t = 0
# 			for w1 in words:
# 				for j in range(5):
# 					if not mutuexclu(list(w1), fix, may):
# 						t += 1
# 			e.append(t)
# 		entropy.append(e)
#
# 	pandas.DataFrame(entropy).T.to_csv()
#


def mutuexclu(w, f, m):
	for i in range(5):
		if f[i] != '/' and f[i] != w[i]:
			if w[i] in m:
				m[m.index(w[i])] = '/'
			else:
				return False
	return True

if __name__ == '__main__':
	main()