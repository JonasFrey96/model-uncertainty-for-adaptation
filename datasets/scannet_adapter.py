from ucdr.datasets import ScanNet
from torchvision import transforms
from torch.utils.data import Dataset

class ScanNetAdapter(ScanNet):
    def __init__(self, 
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
                labels_generic=""):
        
        output_trafo = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        super(ScanNetAdapter, self).__init__(root, mode, scenes, output_trafo ,output_size,degrees, flip_p,jitter_bcsh, sub,data_augmentation, label_setting, confidence_aux,labels_generic)
        
    def __getitem__(self, index):
        img, label, aux_label, aux_valid, img_ori = super(ScanNetAdapter, self).__getitem__(index)
        
        global_idx = self.global_to_local_idx[index]
        p = self.image_pths[global_idx].split("/")
        p = p[-3] + "_" + p[-1]
        return img, label, p