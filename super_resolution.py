import cv2

def bilinear_interpolation(img, scale):
    h, w = img.shape[:2]
    new_h, new_w = h * scale, w * scale
    # 双一次補間
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return scaled_img

def bicubic_interpolation(img, scale):
    h, w = img.shape[:2]
    new_h, new_w = h * scale, w * scale
    # 双一次補間
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return scaled_img

def fsrcnn_x4(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("models/FSRCNN_x4.pb")  # 学習済みモデル
    sr.setModel("fsrcnn", 4)      # モデル名と倍率
    scaled_img = sr.upsample(img)     # 画像に適用
    return scaled_img

def edsr_x4(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("models/EDSR_x4.pb")  # 学習済みモデル
    sr.setModel("edsr", 4)      # モデル名と倍率
    scaled_img = sr.upsample(img)     # 画像に適用
    return scaled_img


if __name__ == '__main__':

    import sys
    import os

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    scale = int(sys.argv[3])

    input_img = cv2.imread(input_path)
    os.makedirs(output_path, exist_ok=True)

    scaled_img_bilinear = bilinear_interpolation(input_img, scale)
    scaled_img_bicubic = bicubic_interpolation(input_img, scale)
    scaled_img_fsrcnn = fsrcnn_x4(input_img)
    scaled_img_edsr = edsr_x4(input_img)

    cv2.imwrite(os.path.join(output_path, 'scaled_image_bilinear.png'), scaled_img_bilinear)
    cv2.imwrite(os.path.join(output_path, 'scaled_image_bicubic.png'), scaled_img_bicubic)
    cv2.imwrite(os.path.join(output_path, 'scaled_image_fsrcnn.png'), scaled_img_fsrcnn)
    cv2.imwrite(os.path.join(output_path, 'scaled_image_edsr.png'), scaled_img_edsr)

