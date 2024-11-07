import cv2
import numpy as np
import yaml
from xpoint.models.RIFT2.src.phase_congruency.phasecong import phasecong
from joblib import Parallel, delayed


class RIFT2:
    def __init__(self, config_file=None, **external_params):
        # Default configuration dictionary
        self.default_config = {
            'nscale': 4,             # Number of wavelet scales for phase congruency computation
            'norient': 6,            # Number of orientations for phase congruency computation
            'npt': 1000,             # Number of keypoints to retain after feature detection
            'minWaveLength': 3,      # Minimum wavelength for the filters in the phase congruency computation
            'mult': 1.6,             # Scaling factor between successive filters in the phase congruency computation
            'sigmaOnf': 0.75,        # Ratio of the standard deviation of the Gaussian describing the log Gabor filter's transfer function in the frequency domain to the filter center frequency
            'g': 3,                  # Sharpness of the phase congruency computation
            'k': 1,                  # Noise compensation factor for phase congruency
            'patchsize': 96,        # Size of the patches to extract for feature description
            'no': 6,                 # Number of orientations for the orientation histogram
            'nbin': 6,               # Number of bins for the orientation histogram
            'is_ori': 1,             # Flag to compute the orientation of keypoints (1: compute, 0: do not compute)
            'ori_peak_ratio': 0.8    # Ratio used for peak selection in the orientation histogram
        }

        if config_file:
            with open(config_file, 'r') as file:
                file_config = yaml.safe_load(file)
            self.config = {**self.default_config, **file_config}
        else:
            self.config = self.default_config
        
        self.config.update(external_params)

    def _convert_to_grayscale(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def feature_detection(self, im):
        config = self.config
        m, _, _, _, _, eo, _ = phasecong(im, nscale=config['nscale'], norient=config['norient'], 
                                         minWaveLength=config['minWaveLength'], mult=config['mult'], 
                                         sigmaOnf=config['sigmaOnf'], g=config['g'], k=config['k'])
        a = np.max(m)
        b = np.min(m)
        m = (m - b) / (a - b)

        m_image = (m * 255).astype(np.uint8)
        eo = np.transpose(eo, (1, 0, 2, 3))

        fast = cv2.FastFeatureDetector_create(threshold=1, nonmaxSuppression=True)
        keypoints = fast.detect(m_image, None)

        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
        keypoints = keypoints[:config['npt']]
        kpts = np.array([kp.pt for kp in keypoints]).T

        return kpts, m, eo

    def feature_description(self, img_hw, eo, kpts):
        config = self.config
        n = kpts.shape[1]
        yim, xim = img_hw
        CS = np.zeros((yim, xim, config['no']))

        for j in range(config['no']):
            for i in range(4):
                CS[:, :, j] += np.abs(eo[i][j])

        MIM = np.argmax(CS, axis=2)
        des = np.zeros((config['no'] * config['no'] * config['nbin'], n))

        def process_keypoint(k):
            x = kpts[0, k]
            y = kpts[1, k]
            r = config['patchsize']
            ang = kpts[2, k]

            patch = self.extract_patches(MIM, x, y, round(r / 2), ang)
            patch = cv2.resize(patch, (r + 1, r + 1), interpolation=cv2.INTER_LINEAR)
            h, _ = np.histogram(patch, bins=np.arange(1, config['no'] + 2))
            idx = np.argmax(h)
            patch_rot = patch - idx + 1
            patch_rot[patch_rot < 0] += config['no']

            ys, xs = patch_rot.shape
            histo = np.zeros((config['no'], config['no'], config['nbin']))

            for j in range(config['no']):
                for i in range(config['no']):
                    clip = patch_rot[round((j) * ys / config['no']):round((j + 1) * ys / config['no']),
                                      round((i) * xs / config['no']):round((i + 1) * xs / config['no'])]
                    histo[j, i, :] = np.histogram(clip, bins=np.arange(1, config['nbin'] + 2))[0]

            histo = histo.flatten()

            if np.linalg.norm(histo) != 0:
                histo = histo / np.linalg.norm(histo)

            return histo

        des = Parallel(n_jobs=-1)(delayed(process_keypoint)(k) for k in range(n))
        des = np.array(des).T

        return des

    def extract_patches(self, img, x, y, s, t):
        img = img.astype(np.float32)
        h, w = img.shape[:2]
        m = img.shape[2] if img.ndim == 3 else 1

        x = np.clip(np.round(x).astype(int), 0, w - 1)
        y = np.clip(np.round(y).astype(int), 0, h - 1)

        s = int(round(s))
        t = np.deg2rad(t)

        patchsize = s * 2 + 1
        xg, yg = np.meshgrid(np.arange(-s, s + 1), np.arange(-s, s + 1))
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        xygrot = R @ np.vstack([xg.ravel(), yg.ravel()])
        xygrot[0, :] += x
        xygrot[1, :] += y

        xr = xygrot[0, :]
        yr = xygrot[1, :]
        xf = np.floor(xr).astype(int)
        yf = np.floor(yr).astype(int)
        xp = xr - xf
        yp = yr - yf

        patch = np.zeros((patchsize, patchsize, m))

        valid_mask = (xf >= 0) & (xf <= w - 2) & (yf >= 0) & (yf <= h - 2)
        xf = xf[valid_mask]
        yf = yf[valid_mask]
        xp = xp[valid_mask]
        yp = yp[valid_mask]

        ind1 = np.ravel_multi_index((yf, xf), (h, w))
        ind2 = np.ravel_multi_index((yf, xf + 1), (h, w))
        ind3 = np.ravel_multi_index((yf + 1, xf), (h, w))
        ind4 = np.ravel_multi_index((yf + 1, xf + 1), (h, w))

        for ch in range(m):
            imgch = img[:, :, ch] if m > 1 else img
            ivec = ((1 - yp) * (xp * imgch.ravel()[ind2] + (1 - xp) * imgch.ravel()[ind1]) +
                    yp * (xp * imgch.ravel()[ind4] + (1 - xp) * imgch.ravel()[ind3]))
            temp = np.zeros((patchsize, patchsize))
            temp.ravel()[valid_mask] = ivec
            patch[:, :, ch] = temp

        return patch

    def compute_orientation(self, key, im):
        config = self.config
        if config['is_ori'] == 1:
            n = 24
            ORI_PEAK_RATIO = config['ori_peak_ratio']
            h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

            gradient_x = cv2.filter2D(im, -1, h, borderType=cv2.BORDER_REPLICATE)
            gradient_y = cv2.filter2D(im, -1, h.T, borderType=cv2.BORDER_REPLICATE)
            gradientImg = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
            temp_angle = np.arctan2(gradient_y, gradient_x)
            temp_angle = np.degrees(temp_angle)
            temp_angle[temp_angle < 0] += 360
            gradientAng = temp_angle

        feat_index = 0
        kpts = np.zeros((3, key.shape[1] * 6))
        for k in range(key.shape[1]):
            x = int(round(key[0, k]))
            y = int(round(key[1, k]))
            r = int(round(config['patchsize']))

            x1 = max(1, x - r // 2)
            y1 = max(1, y - r // 2)
            x2 = min(x + r // 2, im.shape[1] - 1)
            y2 = min(y + r // 2, im.shape[0] - 1)

            if y2 - y1 != r or x2 - x1 != r:
                continue

            if config['is_ori'] == 1:
                angle = self.orientation(x, y, gradientImg, gradientAng, r, n, ORI_PEAK_RATIO)

                for i in range(len(angle)):
                    kpts[:, feat_index] = [x, y, angle[i]]
                    feat_index += 1
            else:
                kpts[:, feat_index] = [x, y, 0]
                feat_index += 1

        kpts = kpts[:, kpts[0, :] != 0]

        return kpts

    def orientation(self, x, y, gradientImg, gradientAng, patchsize, n, ORI_PEAK_RATIO):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(patchsize + 1), int(patchsize + 1)))
        Sa = se.astype(np.uint8)
        hist, max_value = self.calculate_orientation_hist(x, y, patchsize / 2, gradientImg, gradientAng, n, Sa)

        mag_thr = max_value * ORI_PEAK_RATIO
        ANG = []
        for k in range(n):
            k1 = n - 1 if k == 0 else k - 1
            k2 = 0 if k == n - 1 else k + 1
            if hist[k] > hist[k1] and hist[k] > hist[k2] and hist[k] > mag_thr:
                bin = k - 1 + 0.5 * (hist[k1] - hist[k2]) / (hist[k1] + hist[k2] - 2 * hist[k])
                if bin < 0:
                    bin = n + bin
                elif bin >= n:
                    bin = bin - n
                angle = (360 / n) * bin
                ANG.append(angle)
        return ANG

    def calculate_orientation_hist(self, x, y, radius, gradient, angle, n, Sa):
        sigma = radius / 3
        radius_x_left = int(x - radius)
        radius_x_right = int(x + radius)
        radius_y_up = int(y - radius)
        radius_y_down = int(y + radius)

        radius_x_left = max(0, radius_x_left)
        radius_x_right = min(gradient.shape[1], radius_x_right + 1)
        radius_y_up = max(0, radius_y_up)
        radius_y_down = min(gradient.shape[0], radius_y_down + 1)

        sub_gradient = gradient[radius_y_up:radius_y_down, radius_x_left:radius_x_right]
        sub_angle = angle[radius_y_up:radius_y_down, radius_x_left:radius_x_right]

        X = np.arange(-(x - radius_x_left), (radius_x_right - x))
        Y = np.arange(-(y - radius_y_up), (radius_y_down - y))
        XX, YY = np.meshgrid(X, Y)

        gaussian_weight = np.exp(-(XX ** 2 + YY ** 2) / (2 * sigma ** 2))
        W1 = sub_gradient * gaussian_weight
        W = np.double(Sa[:W1.shape[0], :W1.shape[1]]) * np.double(W1)

        bin = np.round(sub_angle * n / 360).astype(int)
        bin[bin >= n] -= n
        bin[bin < 0] += n

        temp_hist = np.zeros(n)
        for i in range(n):
            wM = W[bin == i]
            if wM.size > 0:
                temp_hist[i] = np.sum(wM)

        hist = np.zeros(n)
        hist[0] = (temp_hist[n - 2] + temp_hist[2]) / 16 + \
                  4 * (temp_hist[n - 1] + temp_hist[1]) / 16 + temp_hist[0] * 6 / 16
        hist[1] = (temp_hist[n - 1] + temp_hist[3]) / 16 + \
                  4 * (temp_hist[0] + temp_hist[2]) / 16 + temp_hist[1] * 6 / 16
        hist[2:n - 2] = (temp_hist[0:n - 4] + temp_hist[4:n]) / 16 + \
                        4 * (temp_hist[1:n - 3] + temp_hist[3:n - 1]) / 16 + temp_hist[2:n - 2] * 6 / 16
        hist[n - 2] = (temp_hist[n - 4] + temp_hist[0]) / 16 + \
                      4 * (temp_hist[n - 3] + temp_hist[n - 1]) / 16 + temp_hist[n - 2] * 6 / 16
        hist[n - 1] = (temp_hist[n - 3] + temp_hist[1]) / 16 + \
                      4 * (temp_hist[n - 2] + temp_hist[0]) / 16 + temp_hist[n - 1] * 6 / 16

        max_value = np.max(hist)
        return hist, max_value

    def process_features(self, img1, img2):
        img1_grayscale = self._convert_to_grayscale(img1)
        img2_grayscale = self._convert_to_grayscale(img2)

        img1_hw = img1_grayscale.shape
        img2_hw = img2_grayscale.shape

        print('RIFT2 feature detection')
        key1, m1, eo1 = self.feature_detection(img1_grayscale)
        key2, m2, eo2 = self.feature_detection(img2_grayscale)

        print('RIFT2 main orientation calculation')
        kpts1 = self.compute_orientation(key1, m1)
        kpts2 = self.compute_orientation(key2, m2)

        print('RIFT2 feature description')
        des1 = self.feature_description(img1_hw, eo1, kpts1)
        des2 = self.feature_description(img2_hw, eo2, kpts2)

        kp1 = kpts1.T
        kp2 = kpts2.T
        des1 = des1.T
        des2= des2.T

        kp1 = [cv2.KeyPoint(x=float(point[0]), y=float(point[1]), size=1) for point in kp1]
        kp2 = [cv2.KeyPoint(x=float(point[0]), y=float(point[1]), size=1) for point in kp2]

        if des1.dtype != np.float32:
            des1 = des1.astype(np.float32)
        if des2.dtype != np.float32:
            des2= des2.astype(np.float32)

        return kp1, des1, kp2, des2
    
    def detectAndCompute(self, img1,mask):
        img1_grayscale = self._convert_to_grayscale(img1)

        img1_hw = img1_grayscale.shape

        #print('RIFT2 feature detection')
        key1, m1, eo1 = self.feature_detection(img1_grayscale)

        #print('RIFT2 main orientation calculation')
        kpts1 = self.compute_orientation(key1, m1)

        #print('RIFT2 feature description')
        des1 = self.feature_description(img1_hw, eo1, kpts1)

        kp1 = kpts1.T
        des1 = des1.T

        kp1 = np.array([cv2.KeyPoint(x=float(point[0]), y=float(point[1]), size=1) for point in kp1])

        if des1.dtype != np.float32:
            des1 = des1.astype(np.float32)


        return kp1, des1

    def __call__(self, img1, img2):
        kp1, des1, kp2, des2 = self.process_features(img1, img2)
        return kp1, des1, kp2, des2
