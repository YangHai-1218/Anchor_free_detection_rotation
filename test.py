import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from backbone import vovnet39, vovnet57, resnet50, resnet101
from argument import get_args
from model import ATSS,Efficientnet_Bifpn_ATSS
from utils import transform, DOTADataset, collate_fn
from distributed import (
    get_rank,
    synchronize,
)
from train import (
    data_sampler,
)
from utils import Tester
from evaluate import map_to_origin_image
from tools.visualize import show_polygon_bbox

def save_predictions_to_images(dataset, predictions,args):
    # 
    if get_rank() != 0:
        return
        
    for id, pred in enumerate(predictions):
        orig_id = dataset.id2img[id]

        if len(pred) == 0:
            continue

        img_meta = dataset.get_image_meta(id)
        width = img_meta['width']
        height = img_meta['height']
        pred = map_to_origin_image(img_meta, pred, flipmode='no', resize_mode='letterbox')
        #pred = pred.resize((width, height))
        
        boxes = pred.bbox.tolist()
        scores = pred.get_field('scores').tolist()
        ids = pred.get_field('labels').tolist()

        img_name = img_meta['file_name']
        img_baseName = os.path.splitext(img_name)[0]
        # 
        print('saving ' + img_name + ' ...')
        imgroot = dataset.root
        show_polygon_bbox(os.path.join(imgroot, img_name), boxes, ids, dataset.NAME_TAB,
                          file_name=os.path.join(args.working_dir, img_name), scores=scores,
                          score_threshold=0.3)

        # categories = [dataset.id2category[i] for i in ids]
        # for k, box in enumerate(boxes):
        #     category_id = categories[k]
        #     score = scores[k]
            # csv_file.write("%s,%s.xml,%f,%d,%d,%d,%d\n" % (CLASS_NAME[category_id], img_baseName, score,\
            #     int(box[0] + 0.5), int(box[1] + 0.5), int(box[2] + 0.5), int(box[3] + 0.5)))

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

    args = get_args()
    working_dir = args.working_dir
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
        synchronize()

    device = 'cuda'

    # valid_trans = transform.Compose(
    #     [
    #         transform.Resize(args.test_min_size, args.test_max_size),
    #         transform.ToTensor(),
    #         transform.Normalize(args.pixel_mean, args.pixel_std)
    #     ]
    # )
    test_set = DOTADataset(path=args.path, split='val_loss', image_folder_name='min_split_',
                            anno_folder_name='annotations_split_')
    test_trans = transform.Compose(
        [
            transform.Resize_For_Efficientnet(compund_coef=5),
            transform.ToTensor(),
            transform.Normalize(args.pixel_mean, args.pixel_std)
        ]
    )
    test_set.set_transform(test_trans)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch,
        sampler=data_sampler(test_set, shuffle=False, distributed=args.distributed),
        num_workers=args.num_workers,
        collate_fn=collate_fn(args),
    )

    # backbone = vovnet39(pretrained=False)
    # backbone = resnet18(pretrained=False)
    backbone = resnet50(pretrained=False)
    model = ATSS(args, backbone)
    #model = Efficientnet_Bifpn_ATSS(args, compound_coef=args.backbone_coef, load_backboe_weight=False)

    # load weight
    model_file = args.weight_path
    chkpt = torch.load(model_file, map_location='cpu')  # load checkpoint
    model.load_state_dict(chkpt['model'])
    print('load weights from ' + model_file)

    model = model.to(device)
    
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )


    tester = Tester(args, test_loader, test_set, device)
    predictions = tester(model, 1)
    save_predictions_to_images(test_set, predictions, args)


