from skimage.measure import label, regionprops, find_contours
import numpy as np
from torch.utils.data import DataLoader
import torch
from utils.transforms import transform_logits
import cv2
import scipy.cluster
from PIL import Image
from torch.utils import data
from utils.transforms import get_affine_transform


class SimpleFolderDataset(data.Dataset):
    def __init__(self, root, input_size=[512, 512], transform=None):
        self.root = root
        self.input_size = input_size
        self.transform = transform
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]
        self.input_size = np.asarray(input_size)

        # self.file_list = os.listdir(self.root)

    def __len__(self):
        return 1

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        img_name = 'out.jpg'
        # img_path = os.path.join(self.root, img_name)
        img = self.root
        h, w, _ = img.shape

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        input = self.transform(input)
        meta = {
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return input, meta


def get_color(color, csv):
    B, G, R = color
    min_value = 10000
    name_color = None
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=min_value):
            min_value = d
            name_color = csv.loc[i,"color_name"]
    return name_color


def rgb_to_hex(bgr):
    rgb = (bgr[2], bgr[1], bgr[0])
    col = '#%02x%02x%02x' % rgb
    return col


def take_area(tup):
    return (tup[2]-tup[0]) * (tup[3]-tup[1])


def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border


def mtb(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes


def main(image, input_size, model, transform, labels):

    dataset = SimpleFolderDataset(root=image, input_size=input_size, transform=transform)

    dataloader = DataLoader(dataset)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            image, meta = batch
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            output = model(image)
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)

            uniques = np.unique(parsing_result)
            detected_pieces = [labels[u] for u in uniques]
            return detected_pieces, parsing_result

    return [], []


def main_process(image_path, input_size, model, transform, labels):
    image = image_path

    person_image = image

    items, result_img = main(person_image, input_size, model, transform, labels)
    uniques = np.unique(result_img)

    return result_img, uniques


def post_proces(image, final_result, uniques, csv, labels):
    masker = np.zeros_like(image)
    final_mask = masker.copy()
    detections = {}
    for u in uniques:
        white_mask = masker.copy()
        white_mask[final_result == u] = 255
        white_mask = cv2.cvtColor(white_mask, cv2.COLOR_BGR2GRAY)
        bboxes = mtb(white_mask)
        bboxes = sorted(bboxes, key=take_area, reverse=True)
        if len(bboxes) > 0:
            box = bboxes[0]
            masked_image = masker.copy()
            masked_image[final_result == u] = image[final_result == u]
            piece = masked_image[box[1]: box[3], box[0]: box[2]]

            im = Image.fromarray(piece)

            im = im.resize((150, 150))
            ar = np.asarray(im)
            shape = ar.shape
            ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

            codes, dist = scipy.cluster.vq.kmeans(ar, 2)

            codes_list = []
            for c in codes:
                if np.sum(c) > 730 or np.sum(c) < 20:
                    pass
                else:
                    codes_list.append(list(c))
            if len(codes_list) == 0:
                codes_list.append([250, 250, 250])
            codes = np.array(codes_list)
            sum_list = [np.sum(arr) for arr in codes]
            indexing = sum_list.index(max(sum_list))
            color_get = codes[indexing]
            color = [int(col) for col in color_get]
            color_name = get_color(color, csv)

            item = labels[u]
            detections[item] = (color_name, color)

            final_mask[final_result == u] = color
    return final_mask, detections

def get_color_palette(detections):
    data = {'bg': (), 'hair': (), 'skin': (), 'shirt': (), 'pants': (), 'dress': (),
            'skirt': (), 'shoes': (), 'hat': (), 'sunglasses': (), 'bag': (), 'belt': (), 'scarf': ()}

    if 'Background' in list(detections.keys()):
        data['bg'] = detections['Background']
    if 'Hair' in list(detections.keys()):
        data['hair'] = detections['Hair']
    if 'Shirt' in list(detections.keys()):
        data['shirt'] = detections['Shirt']
    if 'Pants' in list(detections.keys()):
        data['pants'] = detections['Pants']
    if 'Dress' in list(detections.keys()):
        data['dress'] = detections['Dress']
    if 'Skirt' in list(detections.keys()):
        data['skirt'] = detections['Skirt']
    if 'Hat' in list(detections.keys()):
        data['hat'] = detections['Hat']
    if 'Bag' in list(detections.keys()):
        data['bag'] = detections['Bag']
    if 'Scarf' in list(detections.keys()):
        data['scarf'] = detections['Scarf']
    if 'Sunglasses' in list(detections.keys()):
        data['sunglasses'] = detections['Sunglasses']
    if 'Belt' in list(detections.keys()):
        data['belt'] = detections['Belt']

    if 'Face' in list(detections.keys()):
        data['skin'] = detections['Face']
    elif 'Left-leg' in list(detections.keys()):
        data['skin'] = detections['Left-leg']
    elif 'Right-leg' in list(detections.keys()):
        data['skin'] = detections['Right-leg']
    elif 'Left-arm' in list(detections.keys()):
        data['skin'] = detections['Left-arm']
    elif 'Right-arm' in list(detections.keys()):
        data['skin'] = detections['Right-arm']

    if 'Left-shoe' in list(detections.keys()):
        data['shoes'] = detections['Left-shoe']
    elif 'Right-shoe' in list(detections.keys()):
        data['shoes'] = detections['Right-shoe']

    return data
