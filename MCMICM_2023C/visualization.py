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
from matplotlib import pyplot

df = pandas.read_csv("data/Problem_C_Data_Wordle.csv")
df = df.iloc[:: -1].reset_index(drop=True)
df["Date"] = df["Date"].astype("datetime64")
df = df.set_index("Date")


def main():
	# NRR()
	# hint()
	letterFreq()
	# wordFreq()
	# eerie()


def savefig(caller):
	pyplot.savefig(f"fig/vision.{caller.__name__}.png", dpi=220)


def NRR():
	pyplot.figure(figsize=(15, 5))
	pyplot.subplot(121)
	pyplot.plot(df.index,
	            df[["Number of  reported results", "Number in hard mode"]])
	pyplot.subplot(122)
	pyplot.plot(df.index, df["Number in hard mode"])
	
	savefig(NRR)
	pyplot.show()


def hint():
	pyplot.figure(figsize=(10, 5))
	pyplot.hist(df[
		            ["1 try", "2 tries", "3 tries", "4 tries", "5 tries", "6 tries",
		             "7 or more tries (X)"]], bins=20)
	pyplot.legend(["1 try", "2 tries", "3 tries", "4 tries", "5 tries", "6 tries",
	               "7 or more tries (X)"])
	pyplot.show()


def letterFreq():
	dfLetter = df[["Word", "Number of  reported results", "Number in hard mode"]]
	dfLetter["Hard"] = dfLetter["Number in hard mode"]/dfLetter[
		"Number of  reported results"]
	
	def frequency(x):
		x = [[k, 1] for k in x]
		cnt = pandas.DataFrame(x, columns=["word", "1"]).groupby("word")[
			"1"].count()
		if cnt.max() == 1:
			return 0.25
		elif cnt.max() == 2 and len(cnt) == 4:
			return 0.5
		elif cnt.max() == 2 and len(cnt) == 3:
			return 0.75
		return 1
	
	dfLetter["freq"] = pandas.Series([frequency(x)/4 for x in dfLetter["Word"]],
	                                 index=dfLetter.index)
	pyplot.figure(figsize=(15, 5))
	pyplot.plot(dfLetter.index, dfLetter["Hard"])
	# pyplot.plot(dfLetter.index, dfLetter["freq"])
	# pyplot.legend(["Hard", "Frequency"])
	savefig(letterFreq)
	pyplot.show()


def wordFreq():
	dfWord = df[["Word", "Number of  reported results", "Number in hard mode"]]
	dfWord["Hard"] = dfWord["Number in hard mode"]/dfWord[
		"Number of  reported results"]
	d = pandas.read_csv("data/Freq.csv", index_col="words")["freq"].to_dict()
	dfWord["freq"] = dfWord["Word"].map(d).astype(float)
	print(dfWord["Hard"].corr(dfWord["freq"]))
	corr = [0]
	for i in range(0, len(dfWord) - 2):
		corr.append(dfWord["freq"][i: i + 3].corr(dfWord["Hard"][i: i + 3]))
	corr += [0]
	print(numpy.array(corr[1: -2]).mean())
	pyplot.figure(figsize=(15, 5))
	pyplot.plot(dfWord.index, dfWord["Hard"])
	pyplot.plot(dfWord.index, dfWord["freq"])
	pyplot.plot(dfWord.index, corr)
	pyplot.legend(["Hard", "Frequency"])
	savefig(wordFreq)
	pyplot.show()


def eerie():
	eerie_ = [
		[0.02075949, 0.04445188, 0.19381508, 0.32124731, 0.20005831, 0.12082927,
		 0.05114071],
		[0.01103882, 0.04104563, 0.20883048, 0.29595566, 0.24367663, 0.15820143,
		 0.03027240]
	]
	pyplot.figure(figsize=(10, 5))
	pyplot.bar(["1 try", "2 tries", "3 tries", "4 tries", "5 tries", "6 tries",
	            "7 or more tries (X)"], eerie_[0])
	pyplot.grid()
	savefig(eerie)
	pyplot.show()


if __name__ == "__main__":
	main()
