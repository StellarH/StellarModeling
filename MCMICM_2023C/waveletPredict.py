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

from datetime import datetime

import numpy
import pandas
import pywt
import seaborn
from matplotlib import pyplot
from statsmodels import api
from statsmodels.tsa.arima.model import ARIMA

level = 3
wavelets = [0, "coif2", "db9"]
date = [0, pandas.date_range("20220307", "20221101"),
        pandas.date_range("20220506", "20221231")]

df = pandas.read_csv("data/Problem_C_Data_Wordle.csv")
df = df.iloc[:: -1].reset_index(drop=True)
df["Date"] = df["Date"].astype("datetime64")
df.set_index(df["Date"])
ser = df["Number of  reported results"]
ser = [ser[: 59]] + [ser[x: x + 60] for x in range(59, 359, 60)]
diff = [0, 1, 1]


def main():
	pyplot.figure(figsize=(15, 5))
	for i in (1, 2):
		s = ser[i].append(ser[i + 1]).append(ser[i + 2])
		pyplot.subplot(1, 2, i)
		wavelet = wavelets[i]
		coeff = pywt.wavedec(s, wavelet, mode="sym", level=level)
		d = diff[i]
		models = [getARMAModel(coeff[i], d) for i in range(len(coeff))]
		coeffAll = pywt.wavedec(s.append(ser[i + 3]), wavelet, mode="sym",
		                        level=level)
		delta = [len(coeffAll[i]) - len(coeff[i]) for i in range(len(coeff))]
		coeffNew = [models[i].predict(start=1, end=len(coeff[i]) + delta[i])
		            for i in range(len(coeff))]
		denoised = pywt.waverec(coeffNew, wavelet)
		pyplot.plot(date[i], s.append(ser[i + 3]).values)
		pyplot.plot(date[i][-61:], denoised[-61:], color="green")
	pyplot.savefig("fig/waveletOthers.png", dpi=220)
	pyplot.show()


def getARMAModel(N, diff):
	n = N.copy()
	for t in range(diff):
		n = numpy.diff(n)
	# AIC准则求解模型阶数p, q
	aic = api.tsa.arma_order_select_ic(N, ic='aic')
	order = aic['aic_min_order']
	print(order)
	order = (order[0], diff, order[1])
	model = ARIMA(N, order=order).fit()
	return model


if __name__ == '__main__':
	main()
