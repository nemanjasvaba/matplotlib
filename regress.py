from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt

# x = [1,2,3,4]
# y = [3,5,7,10] # 10, not 9, so the fit isn't perfect

x = np.random.randn(1000)*2+3
yerr = np.random.randn(1000)*0.4
y = x*(-2)-yerr

print x[:10], y[:10]

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit)
# fit_fn is now a function which takes in x and returns an estimate for y

plt.plot(x,y, 'yo', x, fit_fn(x), '--k')
# plt.xlim(0, 5)
# plt.ylim(0, 12)
plt.show()

coeff_d = linregress(x,y)

print coeff_d

# n = 50
# x = np.random.randn(n)
# y = 3 * np.random.randn(n)

# fig, ax = plt.subplots()
fit = np.polyfit(x, y, deg=1)

# ax.plot(x, fit[0] * x + fit[1], color='red')
# ax.scatter(x, y)
# ax.text(0.5, 10, r"$\int_a^b f(x)\mathrm{d}x$", horizontalalignment='center', fontsize=20)

plt.plot(x, fit[0] * x + fit[1], color='red')
plt.scatter(x, y)
if fit[1] <0:
    symbol = ''
else:
    symbol = '+'
plt.text(0.5, 10, "%f*x %f" % (fit[0], fit[1]) , horizontalalignment='center', fontsize=10)

plt.show()
