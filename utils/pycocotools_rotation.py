from pycocotools.cocoeval import COCOeval
import numpy as np
#from ops import polygon_iou
import torch

class test:
    def __init__(self):
        self.a = 1
class Rotation_COCOeval(COCOeval):
    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]

        g_np = np.array(g).reshape(-1, 8)
        d_np = np.array(d).reshape(-1, 8)
        g_tensor = torch.from_numpy(g_np).to(torch.float64)
        d_tensor = torch.from_numpy(d_np).to(torch.float64)

        ious = polygon_iou(d_tensor, g_tensor).numpy()
        return ious

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100, recallThr=None):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | recall={:<9}| area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)
            recallStr = '{:0.2f}:{:0.2f}'.format(p.recThrs[0], p.recThrs[-1]) \
                if recallThr is None else '{:0.2f}'.format(recallThr)
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if recallThr is not None:
                    r = np.where(recallThr == p.recThrs)[0]
                    s = s[:, r]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, recallStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((15,))

            # iou=0.3 recall
            stats[0] = _summarize(0, iouThr=0.3, maxDets=self.params.maxDets[2])
            # iou=0.3 , average precision
            stats[1] = _summarize(1, iouThr=0.3, maxDets=self.params.maxDets[2], recallThr=round(stats[0], 2))
            stats[2] = stats[0] * stats[1] * 2/(stats[0]+stats[1])
            print(f'F1 score:{stats[2]}')

            stats[3] = _summarize(1)
            stats[4] = _summarize(1, iouThr=.3, maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[6] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[7] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[8] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[10] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[11] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[12] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[13] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[14] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()
