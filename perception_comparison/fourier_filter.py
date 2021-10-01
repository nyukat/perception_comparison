import numpy as np

def calc_dist_from_center(img):
    h, w = img.shape
    center = [int(w/2), int(h/2)]
    Y, X = np.ogrid[:h, :w]
    return np.sqrt((X - center[0])**2 + (Y - center[1])**2)

def calc_gaussian_mask(img, is_low_pass, radius):
    dist_from_center = calc_dist_from_center(img)
    mask = np.exp(-dist_from_center**2 / (2 * radius**2))
    if is_low_pass:
        return mask
    else:
        return 1 - mask

def min_max_normalize(x):
    x -= x.min()
    x /= x.max()
    return x

def fourier_filter(img, is_low_pass, radius):
    assert (radius < min(img.shape))
    scaling_factor = np.sqrt(img.size)
    freq = np.fft.fftshift(np.fft.fft2(img)) / scaling_factor
    mask = calc_gaussian_mask(img, is_low_pass, radius)
    freq *= mask
    filtered = np.fft.ifft2(np.fft.ifftshift(freq)).real * scaling_factor
    filtered = min_max_normalize(filtered)
    filtered *= 4095
    filtered = filtered.astype('uint16')
    return filtered