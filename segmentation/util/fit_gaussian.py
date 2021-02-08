import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import segmentation.util.overlap as ol

# Define some test data which is close to Gaussian
data = np.random.normal(size=10000)

hist, bin_edges = np.histogram(data, density=True)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

img_data_path = r'C:\Segmentation_working_area\test_volume'
og_3d = ol.create_3d_array_from_tiff(img_data_path)

# stitched array (Stardist)
sd_path = r'C:\Segmentation_working_area\stardist_testdata\masks'
sd_3d = ol.create_3d_array(sd_path)
sd_stitch, sd_nlen = ol.calc_all_overlaps(sd_3d)
sd_bright = ol.calc_brightness(og_3d, sd_stitch, sd_nlen)

y_data = np.array(sd_bright['1'])
X_data = np.array(np.arange(len(y_data)))

print(f'y: {y_data} \n x: {X_data}')

# Define model function to be used to fit to the data above:
# Adapt it to as many gaussians you may want
# by copying the function with different A2,mu2,sigma2 parameters

def gauss2(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))

height = len(y_data)/4

# p0 is the initial guess for the fitting coefficients initialize them differently so the optimization algorithm works better
p0 = [np.mean(y_data), height , height, np.mean(y_data), height * 3, height]

# optimize and in the end you will have 6 coeff (3 for each gaussian)
coeff, var_matrix = curve_fit(gauss2, X_data, y_data, p0=p0)

# you can plot each gaussian separately using
pg1 = np.zeros_like(p0)
pg1[0:3] = coeff[0:3]
pg2 =np.zeros_like(p0)
pg2[0:3] = coeff[3:]

g1 = gauss2(X_data, *pg1)
g2 = gauss2(X_data, *pg2)

plt.plot(X_data, y_data, label='Data')
plt.plot(X_data, g1, label='Gaussian1')
plt.plot(X_data, g2, label='Gaussian2')

plt.show()

# coeff contains necessary information!