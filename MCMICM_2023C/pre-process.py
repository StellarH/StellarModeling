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

df = pandas.read_csv("data/Problem_C_Data_Wordle.csv")
words = [x for x in df["Word"].values]
wordsResult = df[
	["1 try", "2 tries", "3 tries", "4 tries", "5 tries", "6 tries",
	 "7 or more tries (X)"]].values
letters = "abcdefghijklmnopqrstuvwxyz"

with open("data/glove.6B.50d.txt") as file:
	with open("data/vecs.txt", "w") as out:
		i = 0
		for line in file:
			print(i)
			i += 1
			word = line.split(maxsplit=2)[0]
			if len(word) == 5:
				try:
					index = words.index(word)
				except ValueError:
					continue
				else:
					for n in wordsResult[words.index(word)]:
						out.write(f"{n/100} ")
					out.write("\t")
					for n in line.split()[1:]:
						out.write(f"{n} ")
					out.write("\n")

# pandas.Series(wordsVec).to_csv("data/wordVec.txt", sep=' ')
