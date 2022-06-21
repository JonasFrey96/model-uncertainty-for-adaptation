from ucdr.datasets import ScanNet
from torchvision import transforms as tv_transform
from torch.utils.data import Dataset
from PIL import Image
from ucdr.visu import Visualizer
import numpy as np


import torch
import imageio


class ScanNetAdapter(ScanNet):
    def __init__(
        self,
        root="/media/scratch2/jonfrey/datasets/scannet/",
        mode="train",
        scenes=[],
        output_trafo=None,
        output_size=(480, 640),
        degrees=10,
        flip_p=0.5,
        jitter_bcsh=[0.3, 0.3, 0.3, 0.05],
        sub=10,
        data_augmentation=True,
        label_setting="default",
        confidence_aux=0,
        labels_generic="",
        pseudo_root=None,
        transforms=None,
    ):

        self.pseudo = pseudo_root is not None
        if self.pseudo:
            from pathlib import Path

            self.tgt_train_lst = [str(s) for s in Path(pseudo_root).rglob("*.png")]
            self.pseudo = True
            self.transforms = transforms
            mode = "train"
            output_trafo = None
            degrees = 0
            flip_p = 0
            jitter_bcsh = [0, 0, 0, 0]
            data_augmentation = False

            self.norm = tv_transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        else:
            output_trafo = tv_transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        super(ScanNetAdapter, self).__init__(
            root,
            mode,
            scenes,
            output_trafo,
            output_size,
            degrees,
            flip_p,
            jitter_bcsh,
            sub,
            data_augmentation,
            label_setting,
            confidence_aux,
            labels_generic,
        )

        if self.pseudo:
            self._length = len(self.tgt_train_lst)

    def __getitem__(self, index):
        if self.pseudo:
            k = self.tgt_train_lst[index]
            idx = (k.split("/")[-1].split("_")[-1])[:-4]
            scene = "_".join(k.split("/")[-1].split("_")[-3:-1])

            p = self.image_pths[0]
            p = p.replace("scene0000_00", scene)
            p = p.replace("0.jpg", f"{idx}.jpg")

            image = Image.open(p).convert("RGB")
            label = Image.open(k)

            img = imageio.imread(p)
            img = torch.from_numpy(img).type(torch.float32).permute(2, 0, 1) / 255  # C H W range 0-1
            img_out, _ = self._augmenter.apply(img, [img[0][None].type(torch.float32)], only_crop=True)

            if False:
                visu = Visualizer("/media/Data/Datasets/tmp")
                lab = torch.from_numpy(np.array(label)[None]).type(torch.long)
                lab[lab == 255] = -1
                visu.plot_detectron(img_out, (lab + 1)[0], store=True, tag=f"{scene}_{idx}")

            # img_out = Image.fromarray(np.uint8((img_out.permute(1,2,0) * 255).numpy()))

            # image, label = self.transforms((img_out, label))
            image = self.norm(img_out)
            return image, torch.from_numpy(np.array(label)), (self.tgt_train_lst[index].split("/")[-1])[:-4]

        img, label, aux_label, aux_valid, img_ori = super(ScanNetAdapter, self).__getitem__(index)
        global_idx = self.global_to_local_idx[index]
        p = self.image_pths[global_idx].split("/")
        p = p[-3] + "_" + p[-1]
        return img, label, p
