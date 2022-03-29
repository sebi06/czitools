from pylibCZIrw import czi as pyczi
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import os
from sklearn.linear_model import LinearRegression
import numpy as np
import math

basedir = r"D:\Testdata_Zeiss\N2V\Noisy_Images_Camera\820m Bin1"

img_orig = ["820m_FL_000,1ms_Bin1x_G1_Adapt1x.czi",
            "820m_FL_000,2ms_Bin1x_G1_Adapt1x.czi",
            "820m_FL_000,5ms_Bin1x_G1_Adapt1x.czi",
            # "820m_FL_020ms_Bin1x_G1_Adapt1x.czi"
            ]

img_n2v = ["820m_FL_000,1ms_Bin1x_G1_Adapt1x-N2V_trained_01-02-05.czi",
           "820m_FL_000,2ms_Bin1x_G1_Adapt1x-N2V_trained_01-02-05.czi",
           "820m_FL_000,5ms_Bin1x_G1_Adapt1x-N2V_trained_01-02-05.czi",
           # "820m_FL_020ms_Bin1x_G1_Adapt1x-N2V_trained_01-02-05.czi"
           ]

# gt_imagefile = r"820m_FL_050ms_Bin1x_G1_Adapt1x.czi"
gt_imagefile = r"820m_FL_020ms_Bin1x_G1_Adapt1x.czi"

psnr_orig = []
psnr_n2v = []
mse_orig = []
ssim_orig = []
mse_n2v = []
ssim_n2v = []


def calculate_scaled_psnr(gt, pred):
    gt = gt.astype(float)
    lr = LinearRegression()
    lr.fit(pred.flatten().reshape(-1, 1), gt.flatten().reshape(-1, 1))
    scaled = pred * lr.coef_ + lr.intercept_
    max_v = gt.max()
    min_v = gt.min()
    gt = 255 * (gt - min_v) / (max_v - min_v)
    scaled = 255 * (scaled - min_v) / (max_v - min_v)
    mse = np.mean((gt - scaled) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    return psnr


# read the gt image
with pyczi.open_czi(os.path.join(basedir, gt_imagefile)) as czidoc_gt:
    img_gt = czidoc_gt.read()[..., 0]
    print("Shape GT:", img_gt.shape)
    mse_gt = mean_squared_error(img_gt, img_gt)
    ssim_gt = ssim(img_gt, img_gt)  # , data_range=img_gt.max() - img_gt.min())
    print("MSE GT:", mse_gt)
    print("SSIM GT:", ssim_gt)

for imgfile_orig in img_orig:
    filepath_orig = os.path.join(basedir, imgfile_orig)
    with pyczi.open_czi(filepath_orig) as czidoc_orig:
        img_orig = czidoc_orig.read()[..., 0]
        print("Shape Orig", imgfile_orig, img_orig.shape)
        # psnr_orig.append(peak_signal_noise_ratio(img_gt, img_orig))
        psnr_orig.append(calculate_scaled_psnr(img_gt, img_orig))
        mse_orig.append(mean_squared_error(img_gt, img_orig))
        ssim_orig.append(ssim(img_gt, img_orig, data_range=img_orig.max() - img_orig.min()))

for imgfile_n2v in img_n2v:
    filepath_n2v = os.path.join(basedir, imgfile_n2v)
    with pyczi.open_czi(filepath_n2v) as czidoc_n2v:
        img_n2v = czidoc_n2v.read()[..., 0]
        print("Shape Orig", imgfile_n2v, img_n2v.shape)
        # psnr_n2v.append(peak_signal_noise_ratio(img_gt, img_n2v))
        psnr_n2v.append(calculate_scaled_psnr(img_gt, img_n2v))
        mse_n2v.append(mean_squared_error(img_gt, img_n2v))
        ssim_n2v.append(ssim(img_gt, img_n2v, data_range=img_n2v.max() - img_n2v.min()))

print("PSNR + MSE + SSIM - Original Images")
print(psnr_orig)
print(mse_orig)
print(ssim_orig)
print("PSNR + MSE + SSIM - N2V")
print(psnr_n2v)
print(mse_n2v)
print(ssim_n2v)
