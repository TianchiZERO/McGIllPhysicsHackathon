import numpy as np
import math

def bgr2y(image):
    return 16. + (64.738 * image[:, :, 2] + 129.057 * image[:, :, 1] + 25.064 * image[:, :, 0]) / 256.
def bgr2cb(image):
    return 128. + (-37.945 * image[:, :, 2] - 74.494 * image[:, :, 1] + 112.439 * image[:, :, 0]) / 256.
def bgr2cr(image):
    return 128. + (112.439 * image[:, :, 2] - 94.154 * image[:, :, 1] - 18.285 * image[:, :, 0]) / 256.
def ycbcr2brg(y,cb,cr):
    r = 298.082 * y / 256. + 408.583 * cr / 256. - 222.921
    g = 298.082 * y / 256. - 100.291 * cb / 256. - 208.120 * cr / 256. + 135.576
    b = 298.082 * y / 256. + 516.412 * cb / 256. - 276.836
    return np.array([b, g, r]).transpose([1, 2, 0])
def psnr(prediction, reference, max_val=255.0):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    prediction and refenrence are in numpy form and the values are from 0 to 1
    """
  
    img_diff = prediction - reference
    tmp= math.sqrt(np.mean((img_diff) ** 2))
    if tmp == 0:
        return 100
    else:
        return 20*math.log10(max_val/tmp)
       
