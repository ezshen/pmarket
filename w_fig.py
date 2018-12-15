xs = [0, 0.03, 0.06, 0.08, 0.1,  0.16, 0.22, 0.28, 0.34, 0.4,  0.46, 0.52, 0.58, 0.64]
ys = [6.406957294062918, 6.197989648416991, 6.1035408201677255, 6.0164734637, 6.355156379287257, 6.618233860595365, 7.144672234900092, 7.503449597633973, 8.073992220169087, 10.56154368626637, 11.59328610753921, 14.707203332233565, 16.676659353473013, 25.08061699978839]

from matplotlib import pyplot as plt

plt.plot(xs, ys)
plt.savefig("w.png")
plt.clf()
plt.plot(xs, ys)
plt.ylim(4, 9.3)
plt.savefig("w_zoom.png")