from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt


COLOR_TABLE = [
    'Red', 'Green', 'Blue', 'Yellow',
    'Purple', 'Orange', 'DarkGreen', 'Purple',
    'YellowGreen', 'Maroon', 'Teal',
    'DarkGoldenrod', 'Peru', 'DarkRed', 'Tan',
    'AliceBlue', 'LightBlue', 'Cyan', 'Teal',
    'SpringGreen', 'SeaGreen', 'Lime', 'DarkGreen',
    'YellowGreen', 'Ivory', 'Olive', 'DarkGoldenrod',
    'Orange', 'Tan', 'Peru', 'Seashell',
    'Coral', 'RosyBrown', 'Maroon', 'DarkRed',
    'WhiteSmoke', 'LightGrey', 'Gray'
] * 10



def draw_bbox_text(drawObj, ymin, xmin, ymax, xmax, text, color, bd=2):
    drawObj.rectangle((xmin, ymin, xmax, ymin+bd), fill=color)
    drawObj.rectangle((xmin, ymax-bd, xmax, ymax), fill=color)
    drawObj.rectangle((xmin, ymin, xmin+bd, ymax), fill=color)
    drawObj.rectangle((xmax-bd, ymin, xmax, ymax), fill=color)
    strlen = len(text)
    # drawObj.rectangle((xmin, ymin, xmin+strlen*6+5, ymin+12), fill='Black')
    drawObj.text((xmin+3, ymin), text, fill=color)


def draw_ploygon_bbox_text(drawObj, anticlockwise_points, text, color, bd=2):
    anticlockwise_points = np.array(anticlockwise_points).reshape((4,2)).tolist()
    anticlockwise_points = list(map(tuple,anticlockwise_points))
    for i in range(len(anticlockwise_points)):
        if i != 3:
            drawObj.line([anticlockwise_points[i],anticlockwise_points[i+1]],fill=color,width=bd)
        else:
            drawObj.line([anticlockwise_points[i], anticlockwise_points[0]], fill=color, width=bd)
    drawObj.text((anticlockwise_points[0][0]+3,anticlockwise_points[0][1]),text,fill=color)




def show_polygon_bbox(img, boxes, labels, NAME_TAB, file_name=None, scores=None,
                matplotlib=False, lb_g=True):
    '''
    img:      image_path(str) or PIL.Image obj
    boxes:    list (N, 8), clockwise points xy,
    labels:   (N)
    NAME_TAB: ['background', 'class_1', 'class_2', ...]
    file_name: 'out.bmp' or None
    scores:   (N)
    '''
    if lb_g: bg_idx = 0
    else: bg_idx = -1
    if not isinstance(img,Image.Image):
        assert isinstance(img,str)
        img = Image.open(img)

    if img.mode != 'RGB':
        img = img.convert('RGB')
    # if not isinstance(img, Image.Image):
        # img = transforms.ToPILImage()(img)
    width, height = img.size
    drawObj = ImageDraw.Draw(img)
    for box_id in range(len(boxes)):
        lb = int(labels[box_id])
        if lb > bg_idx:
            box = boxes[box_id]
            # if NAME_TAB is not None:
            if scores is None:
                draw_ploygon_bbox_text(drawObj, box, NAME_TAB[lb], color=COLOR_TABLE[lb])
            else:
                str_score = str(float(scores[box_id]))[:5]
                str_out = NAME_TAB[lb] + ': ' + str_score
                draw_ploygon_bbox_text(drawObj, box, str_out, color=COLOR_TABLE[lb])
    if file_name is not None:
        img.save(file_name)
    else:
        if matplotlib:
            plt.imshow(img, aspect='equal')
            plt.show()
        else: img.show()







def test():
    import cv2
    from utils.dataset import DOTADataset
    import random
    dataset = DOTADataset('/Volumes/hy_mobile/03data/DOTA-v1.5', split='train', image_folder_name='min_split_',
                          anno_folder_name='annotations_split_')

    for i in range(30):
        #i = random.randint(0,len(dataset)-1)
        i = 82
        img, target, idx, path = dataset[i]
        print(f'origin_target:{target}')
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print(target)
        target = target.convert('xyxyxyxy')
        show_polygon_bbox(img, target.bbox, target.get_field("labels"), dataset.NAME_TAB,
                          file_name=None,
                          scores=None,matplotlib=False,
                         lb_g=True)
        print(path)
        print(target)


if __name__ == '__main__':
    test()