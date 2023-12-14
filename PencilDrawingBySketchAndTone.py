import numpy as np
import cv2
from skimage import io, color, filters, transform, exposure
from scipy import signal, sparse
import matplotlib.pyplot as plt

kernel_size = 8
stroke_width = 1
num_of_directions = 8
gradient_method = 1
smooth_kernel = "gauss"


def gen_stroke_map(img):
    height = img.shape[0]
    width = img.shape[1]

    if smooth_kernel == "gauss":
        smooth_im = filters.gaussian(img, sigma=np.sqrt(2))
    else:
        smooth_im = filters.median(img)

    if not gradient_method:
        imX = np.zeros_like(img)
        diffX = img[:, 1:width] - img[:, 0:width - 1]
        imX[:, 0:width - 1] = diffX
        imY = np.zeros_like(img)
        diffY = img[1:height, :] - img[0:height - 1, :]
        imY[0:height - 1, :] = diffY
        G = np.sqrt(np.square(imX) + np.square(imY))
    else:
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        G = np.sqrt(np.square(sobelx) + np.square(sobely))

    basic_ker = np.zeros((kernel_size * 2 + 1, kernel_size * 2 + 1))
    basic_ker[kernel_size + 1, :] = 1  # ------- (horizontal line)

    res_map = np.zeros((height, width, num_of_directions))

    for d in range(num_of_directions):
        ker = transform.rotate(basic_ker, (d * 180) / num_of_directions)
        res_map[:, :, d] = signal.convolve2d(G, ker, mode='same')

    max_pixel_indices_map = np.argmax(res_map, axis=2)

    C = np.zeros_like(res_map)
    for d in range(num_of_directions):
        C[:, :, d] = G * (max_pixel_indices_map == d)

    if not stroke_width:
        for w in range(1, stroke_width + 1):
            if (kernel_size + 1 - w) >= 0:
                basic_ker[kernel_size + 1 - w, :] = 1
            if (kernel_size + 1 + w) < (kernel_size * 2 + 1):
                basic_ker[kernel_size + 1 + w, :] = 1

    S_tag_sep = np.zeros_like(C)
    for d in range(num_of_directions):
        ker = transform.rotate(basic_ker, (d * 180) / num_of_directions)
        S_tag_sep[:, :, d] = signal.convolve2d(C[:, :, d], ker, mode='same')

    S_tag = np.sum(S_tag_sep, axis=2)

    S_tag_normalized = (S_tag - np.min(S_tag.ravel())) / (np.max(S_tag.ravel()) - np.min(S_tag.ravel()))
    S = 1 - S_tag_normalized
    return S


def gen_tone_map(img, w_group=0):
    w_mat = np.array([[11, 37, 52],
                      [29, 29, 42],
                      [2, 22, 76]])
    w = w_mat[w_group, :]

    # dark: [0-85]
    # mild: [86-170]
    # bright: [171-255]

    u_b = 225
    u_a = 105
    sigma_b = 9
    mu_d = 90
    sigma_d = 11

    num_pixel_vals = 256
    p = np.zeros(num_pixel_vals)
    for v in range(num_pixel_vals):
        p1 = (1 / sigma_b) * np.exp(-(255 - v) / sigma_b)
        if u_a <= v <= u_b:
            p2 = 1 / (u_b - u_a)
        else:
            p2 = 0
        p3 = (1 / np.sqrt(2 * np.pi * sigma_d)) * np.exp((-np.square(v - mu_d)) / (2 * np.square(sigma_d)))
        p[v] = w[0] * p1 + w[1] * p2 + w[2] * p3 * 0.01

    p_normalized = p / np.sum(p)
    P = np.cumsum(p_normalized)

    h = exposure.histogram(img, nbins=256)
    H = np.cumsum(h / np.sum(h))

    # histogram matching:
    lut = np.zeros_like(p)
    for v in range(num_pixel_vals):
        dist = np.abs(P - H[v])
        argmin_dist = np.argmin(dist)
        lut[v] = argmin_dist
    lut_normalized = lut / num_pixel_vals
    J = lut_normalized[(255 * img).astype(np.int)]

    J_smoothed = filters.gaussian(J, sigma=np.sqrt(2))
    return J_smoothed


def gen_pencil_texture(img, H, J):
    lamda = 0.2
    height = img.shape[0]
    width = img.shape[1]

    H_res = cv2.resize(H, (width, height), interpolation=cv2.INTER_CUBIC)
    H_res_reshaped = np.reshape(H_res, (height * width, 1))
    logH = np.log(H_res_reshaped)

    J_res = cv2.resize(J, (width, height), interpolation=cv2.INTER_CUBIC)
    J_res_reshaped = np.reshape(J_res, (height * width, 1))
    logJ = np.log(J_res_reshaped)

    logH_sparse = sparse.spdiags(logH.ravel(), 0, height * width, height * width)  # 0 - from main diagonal
    e = np.ones((height * width, 1))
    ee = np.concatenate((-e, e), axis=1)
    diags_x = [0, height * width]
    diags_y = [0, 1]
    dx = sparse.spdiags(ee.T, diags_x, height * width, height * width)
    dy = sparse.spdiags(ee.T, diags_y, height * width, height * width)

    A = lamda * ((dx @ dx.T) + (dy @ dy.T)) + logH_sparse.T @ logH_sparse
    b = logH_sparse.T @ logJ

    beta = sparse.linalg.cg(A, b, tol=1e-6, maxiter=60)

    beta_reshaped = np.reshape(beta[0], (height, width))

    T = np.power(H_res, beta_reshaped)

    return T


def gen_pencil_drawing(img, rgb=False, w_group=0, pencil_texture_path="", stroke_darkness=1,
                       tone_darkness=1):
    if not rgb:
        im = img
    else:
        yuv_img = color.rgb2yuv(img)
        im = yuv_img[:, :, 0]

    S = gen_stroke_map(im)
    S = np.power(S, stroke_darkness)

    # plt.imshow(S, cmap='gray')
    # # plt.title('S')
    # plt.axis('off')
    # plt.savefig('output_images/messi_s.png', bbox_inches='tight', pad_inches=0)
    # plt.show()

    J = gen_tone_map(im, w_group=w_group)

    plt.imshow(J, cmap='gray')
    # plt.title('J')
    plt.axis('off')
    plt.savefig('output_images/messi_j_5.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    if not pencil_texture_path:
        pencil_texture = io.imread('./pencils/pencil0.jpg', as_gray=True)
    else:
        pencil_texture = io.imread(pencil_texture_path, as_gray=True)

    T = gen_pencil_texture(im, pencil_texture, J)
    T = np.power(T, tone_darkness)

    plt.imshow(T, cmap='gray')
    # plt.title('T')
    plt.axis('off')
    plt.savefig('output_images/messi_t_5.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    # # The final Y channel:
    R = np.multiply(S, T)

    plt.imshow(R, cmap='gray')
    # plt.title('R')
    plt.axis('off')
    plt.savefig('output_images/messi_r_5.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    if not rgb:
        return R
    else:
        yuv_img[:, :, 0] = R
        return exposure.rescale_intensity(color.yuv2rgb(yuv_img), in_range=(0, 1))


if __name__ == '__main__':
    messi_img = io.imread('./input_images/messi.jpg')

    # plt.imshow(messi_img)
    # # plt.title('Input image')
    # plt.axis('off')
    # plt.show()

    pencil_tex = './pencils/pencil0.jpg'
    messi_im_pen = gen_pencil_drawing(messi_img,
                                      rgb=True, w_group=0, pencil_texture_path=pencil_tex,
                                      stroke_darkness=2, tone_darkness=1.5)
    plt.imshow(messi_im_pen)
    # plt.title('Pencil Sketch')
    plt.axis('off')
    plt.savefig('output_images/messi_sketch_5.png', bbox_inches='tight', pad_inches=0)
    plt.show()
