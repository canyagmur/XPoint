import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import xpoint.utils as utils
from xpoint.models.SwinTransformerV2 import SwinTransformerV2
from xpoint.models.RegNet import RegNet


# from multipoint.models.vmamba import build_vssm_model
# from multipoint.models.vmamba.MYCONFIG import get_configmultipoint


# try:
#     from xpoint.models.VMamba.classification.models import build_vssm_model
#     from xpoint.models.VMamba.classification.models.MYCONFIG import get_config
# except:
#     print("VMamba not found")

VMAMBA_IMPORT_FLAG = False

try:
    from xpoint.models.vmamba_src.VMamba import build_vssm_model
    from xpoint.models.vmamba_src.MYCONFIG import get_config
    VMAMBA_IMPORT_FLAG = True
except ImportError as e:
    VMAMBA_IMPORT_FLAG = False
    print("Error: VMamba module could not be imported. Please ensure all necessary packages are installed. (https://github.com/MzeroMiko/VMamba)")
    print(f"Detailed error message: {e}")

from xpoint.models.vmamba_src.VMamba import build_vssm_model
from xpoint.models.vmamba_src.MYCONFIG import get_config





import contextlib

class XPoint(nn.Module):
    default_config = {
        'multispectral': True,
        'descriptor_head': True,
        'intepolation_mode': 'bilinear',
        'descriptor_size': 256,
        'normalize_descriptors': True,
        'final_batchnorm': True,
        'reflection_pad': True,
        'bn_first': False,
        'double_convolution': True,
        'channel_version': 0,
        'verbose': False,
        'mixed_precision': False,
        'force_return_logits': False,
        'takes_pair' : False, #added by me
        'homography_regression_head' : #added by me
            {
            'check': False,
            'type' : 'HomographyNet'
            },
        'use_attention' :  #added by me
            {
            'check' : False,
            'type' : 'SimpleViT',
            'height' : 256,
            'width' : 256,
            'pretrained' :
                 {
                    'check' : True,
                    'type_dir' : "model_weights/swinv2-imagenet/base256"
                }
            },
    }



    def __init__(self, config = None):
        super(XPoint, self).__init__()

        if config:
            self.config = utils.dict_update(copy.deepcopy(self.default_config), config)
        else:
            self.config = self.default_config

        if self.config['reflection_pad']:
            self.pad_method = nn.ReflectionPad2d
        else:
            self.pad_method = nn.ZeroPad2d

        if self.config['channel_version'] == 0:
            self.n_channels = [1, 64, 64, 128, 128]
            self.head_channels = 256 #256#

        elif self.config['channel_version'] == 1:
            self.n_channels = [1, 32, 64, 96, 128]
            self.head_channels = self.config['descriptor_size']

        elif self.config['channel_version'] == 2:
            self.n_channels = [1, 8, 16, 32, 64]
            self.head_channels = self.config['descriptor_size']

        else:
            print('Unknown channel_version: ', self.config['channel_version'])
            self.n_channels = [1, 64, 64, 128, 128]
            self.head_channels = 256

        self.encoder_downsample_ratio = 8 #ADDED BY ME
        
        
        if self.config['multispectral']:
            self.encoder_thermal = self.generate_encoder()
            self.encoder_optical = self.generate_encoder()
        else:
            self.encoder = self.generate_encoder()

        if self.config['homography_regression_head']['check']:
            assert  self.config["takes_pair"], "RegNet can only be used with takes_pair=True"
            self.hm_regressor = RegNet()


        self.detector_head_last_dim = self.encoder_downsample_ratio**2+1 #ADDED BY ME
        #print(self.detector_head_last_dim)
        # detector head
        self.detector_head_convolutions = [
            self.pad_method(1),
            nn.Conv2d(self.n_channels[4], self.head_channels, 3),
            *self.getNonlinearity(self.head_channels),
            nn.Conv2d(self.head_channels, self.detector_head_last_dim , 1), #65 --> self.encoder_downsample_ratio**2 + 1
        ]

        if self.config['final_batchnorm']:
            self.detector_head_convolutions.append(nn.BatchNorm2d(self.detector_head_last_dim)) #65 --> self.encoder_downsample_ratio**2 + 1

        self.detector_head_convolutions = nn.Sequential(*self.detector_head_convolutions)

        self.softmax = nn.Softmax2d()
        self.shuffle = nn.PixelShuffle(self.encoder_downsample_ratio)

        if self.config['descriptor_head']:
            self.descriptor_head_convolutions = [
                self.pad_method(1),
                nn.Conv2d(self.n_channels[4], self.head_channels, 3),
                *self.getNonlinearity(self.head_channels),
                nn.Conv2d(self.head_channels, self.config['descriptor_size'], 1),
            ]

            if self.config['final_batchnorm']:
                self.descriptor_head_convolutions.append(nn.BatchNorm2d(self.config['descriptor_size']))

            self.descriptor_head_convolutions = nn.Sequential(*self.descriptor_head_convolutions)
        

        if self.config['verbose']:
            print('MultiPoint number of trainable parameter: ' + str(sum([p.numel() for p in self.parameters()])))

    def takes_pair(self):
        return  self.config['takes_pair']
    def get_encoder_downsample_ratio(self):
        return self.encoder_downsample_ratio

    # def prep_hm_regression_input(self,optical_data, thermal_data, optical_homography, thermal_homography, top_left_point=[0,0], patch_size_h_w=[128,128]):
    #     import numpy as np
    #     b,c,h,w = optical_data.shape
    #     top_left_point = np.array(top_left_point)
    #     top_right_point = top_left_point+[patch_size_h_w[1],0] 
    #     bottom_left_point = top_left_point+[0,patch_size_h_w[0]]
    #     bottom_right_point = top_left_point+[patch_size_h_w[1],patch_size_h_w[0]]
    #     four_points = [top_left_point, top_right_point, bottom_right_point, bottom_left_point]
        
    #     # Create arrays to store the results

    #     H_four_points = np.zeros((b, 4, 2), dtype=np.int32)

    #     # Loop over the batch dimension
    #     for i in range(b):
    #         perturbed_four_points = []
    #         for point in four_points:
    #             point_hom = np.array([[point[0]], [point[1]], [1]])
    #             point_hom_transformed = optical_homography[i].cpu().numpy() @ thermal_homography[i].cpu().numpy() @ point_hom
    #             perturbed_four_points.append([int(point_hom_transformed[0]),int(point_hom_transformed[1])])
    #         H_four_points[i] = np.subtract(np.array(perturbed_four_points), np.array(four_points))

    #     concatenated = np.concatenate((optical_data.cpu().numpy(), thermal_data.cpu().numpy()), axis=1)

    #     return concatenated , H_four_points
    
    def set_force_return_logits(self, value):
        if not isinstance(value, bool):
            raise ValueError('set_force_return_logits: The input value needs to be a bool')

        self.config['force_return_logits']= value

    def forward(self, data):
        with torch.cuda.amp.autocast() if self.config['mixed_precision'] else contextlib.suppress():
            if not self.takes_pair():
                return self.forward_impl(data)
            else:
                #me
                data_optical = data["optical"]
                data_thermal = data["thermal"]
                pred_optical = self.forward_impl(data_optical)
                pred_thermal = self.forward_impl(data_thermal)

                pred_hm_regression=None
                if self.config["homography_regression_head"]["check"]:
                    if self.config["homography_regression_head"]["type"] == "HomographyNet":
                        reg_inputs = torch.empty(0).to(data_optical['image'].device)
                        for i in range(len(data_optical["image"])):
                            pred_optical_reg = torch.mean(pred_optical["encoder_output"][i],dim=0).unsqueeze(0).unsqueeze(0)
                            pred_thermal_reg = torch.mean(pred_thermal["encoder_output"][i],dim=0).unsqueeze(0).unsqueeze(0)
                            if torch.equal(data_optical["homography"][i].cpu(),torch.eye(3)):
                                reg_input = torch.cat((pred_optical_reg,pred_thermal_reg),axis=1)
                                
                            else :
                                reg_input = torch.cat((pred_thermal_reg,pred_optical_reg),axis=1)
                            
                            reg_inputs = torch.cat((reg_inputs,reg_input), dim=0)
                        #reg_inputs = data['hm_input']
                        #print(reg_inputs.shape)
                        pred_hm_regression = self.hm_regressor(reg_inputs)
                    elif self.config["homography_regression_head"]["type"] == "RegNet" or self.config["homography_regression_head"]["type"] == "SRHENLighterNet" or self.config["homography_regression_head"]["type"] == "SRHENNet":
                        pred_hm_regression = self.hm_regressor(pred_optical["encoder_output"],pred_thermal["encoder_output"])
                    elif self.config["homography_regression_head"]["type"] == "MS2CAHENet":
                        pred_hm_regression = self.hm_regressor(pred_optical["encoder_output"],pred_thermal["encoder_output"])

                return pred_optical, pred_thermal, pred_hm_regression


    def disp_heatmap(self,original,x,original2,x2):
            ##try to vis
            import ntpath
            import os
            import pathlib
            import warnings

            import GPUtil
            import cv2
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            import torch.nn as nn

            #swin_output = swin_output.permute(0, 2, 3, 1)
            print(x.shape)
            #x = x.permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            x = x.permute(0,2,1).squeeze(0).detach().cpu().numpy()
            print(x.shape)
            
            
            emb1_attn = np.mean(x.reshape(32*32, 64), axis=1)
            indices = emb1_attn.argsort()[:int(-0.9 * emb1_attn.shape[0])]
            #print( emb1_attn.argsort())
            emb1_attn[indices] = 0
            emb1_attn = emb1_attn.reshape(32, 32)
            emb1_attn = 255 * (emb1_attn - emb1_attn.min()) / (emb1_attn.max() - emb1_attn.min())
            emb1_attn = np.uint8(emb1_attn)
            emb1_attn = cv2.resize(emb1_attn, (256, 256), interpolation=cv2.INTER_CUBIC)

            x2 = x2.permute(0,2,1).squeeze(0).detach().cpu().numpy()
            emb2_attn = np.mean(x2.reshape(32*32,64), axis=1)
            indices = emb2_attn.argsort()[:int(-0.9 * emb2_attn.shape[0])]
            emb2_attn[indices] = 0
            emb2_attn = emb2_attn.reshape(32, 32)
            emb2_attn = 255 * (emb2_attn - emb2_attn.min()) / (emb2_attn.max() - emb2_attn.min())
            emb2_attn = np.uint8(emb2_attn)
            emb2_attn = cv2.resize(emb2_attn, (256, 256), interpolation=cv2.INTER_CUBIC)

            def display(rgb, attn1, nir_orig, attn2):
                fig, ax = plt.subplots(nrows=2, ncols=2)
                ax[0, 0].imshow(rgb,cmap="gray")
                ax[0, 0].axis('off')
                ax[0, 0].set_title('Input Image')

                ax[0, 1].imshow(rgb)
                ax[0, 1].imshow(attn1, alpha=0.25, cmap='jet')
                ax[0, 1].axis('off')
                ax[0, 1].set_title('Attention')

                ax[1, 0].imshow(nir_orig, cmap="gray")
                ax[1, 0].axis('off')
                ax[1, 0].set_title('Input Image')

                ax[1, 1].imshow(nir_orig, cmap="gray")
                ax[1, 1].imshow(attn2, alpha=0.25, cmap='jet')
                ax[1, 1].axis('off')
                ax[1, 1].set_title('Attention')

                plt.show()
            orig = original[0].permute(1,2,0).detach().cpu().numpy()
            orig2 = original2[0].permute(1,2,0).detach().cpu().numpy()
            display(orig/255, emb1_attn,orig2/255, emb2_attn)
            plt.close()

    def forward_impl(self, data):
        if self.config['multispectral']:
            # create a tensor with the output shape of the encoder
            shape = data['image'].shape
            
            tensor_dtype = torch.float
            if self.config['mixed_precision']:
                tensor_dtype = torch.half
            x = torch.zeros((shape[0], self.n_channels[4], int(shape[2]/self.encoder_downsample_ratio), int(shape[3]/self.encoder_downsample_ratio)), dtype = tensor_dtype).to(data['image'].device)

            # check if there is at least one optical image in the batch
            if len(data['is_optical'].shape) ==1: #added by me
                data['is_optical']=data['is_optical'].unsqueeze(dim=0) #added by me
            
            num_optical = data['is_optical'][:,0].sum()
            num_thermal = shape[0] - num_optical
            if num_optical > 0.0:
                x[data['is_optical'][:,0],:] = self.encoder_optical(data['image'][data['is_optical'][:,0]])

            if num_thermal > 0.0:
                x[~data['is_optical'][:,0],:] = self.encoder_thermal(data['image'][~data['is_optical'][:,0],:])

            encoder_output = x.clone().detach() # NEW added
        
        else:
            x = self.encoder(data['image'])
            encoder_output = x.clone().detach()

        #print(x.shape)
        prob, logits = self.detector_head(x)
        #print(logits.shape)
        out = {'prob': prob,
               'logits': logits,
        }


        if self.config['descriptor_head']:
            desc = self.descriptor_head(x)
            out['desc'] = desc
        out['encoder_output']  = encoder_output
        return out

    def getNonlinearity(self, N):
        if self.config['bn_first']:
            return nn.BatchNorm2d(N), nn.ReLU(True)
        else:
            return nn.ReLU(True), nn.BatchNorm2d(N)

    def getConvolutionBlock(self, N_in, N_out):
        if self.config['double_convolution']:
            return (self.pad_method(1), nn.Conv2d(N_in, N_out, 3), *self.getNonlinearity(N_out),
                    self.pad_method(1), nn.Conv2d(N_out, N_out, 3), *self.getNonlinearity(N_out))
        else:
            return (self.pad_method(1), nn.Conv2d(N_in, N_out, 3), *self.getNonlinearity(N_out))
        
    def getHomograhpyRegressorBlock(self,N_in,N_out,pool=False):
        if pool:
            return (nn.Conv2d(N_in,N_out,3,padding=1),nn.BatchNorm2d(N_out),
                    nn.ReLU(),nn.MaxPool2d(2))
        else:
            return (nn.Conv2d(N_in,N_out,3,padding=1),nn.BatchNorm2d(N_out),
                    nn.ReLU())



    def detector_head(self, x):
        logits = self.detector_head_convolutions(x).to(torch.float)
        #print("logits : ",logits.shape)
        if self.training or self.config['force_return_logits']:
            #print(logits.shape)
            #logits = F.interpolate(logits,size=(512,512),mode="bicubic",align_corners=True)
            return None, logits
        else:
            prob = self.softmax(logits)
            prob = self.shuffle(prob[:,:-1])
            #prob = F.interpolate(prob, size=(512, 640), mode='nearest')
            #prob = F.interpolate(prob,size=(512,512),mode="bicubic",align_corners=True)
            return prob,None #,logits #, None

    def descriptor_head(self, x):
        x = self.descriptor_head_convolutions(x).to(torch.float)

        if self.config['normalize_descriptors']:
            x = torch.nn.functional.normalize(x, p=2, dim=1)

        #print("desc",x.shape)
        #x = F.interpolate(x,size=(64,80),mode="bicubic",align_corners=True)
        #print("desc after",x.shape)
        return x
    

    

    #self.n_channels (0) = [1, 64, 64, 128, 128]
    #self.head_channels = 256
    def generate_encoder(self):
        
        if self.config["use_attention"]["check"]:
            if self.config["use_attention"]["type"] =="Swinv2":
                if self.config['use_attention']['pretrained']['check']:
                    swin_config_object = utils.dict_to_obj(self.config["use_attention"]["model_parameters"]) #utils.get_yaml_object(self.config['use_attention']['pretrained']['type_dir'] )
                    input_image_size = self.config["use_attention"]["height"],self.config["use_attention"]["width"]
                    #print(input_image_size,swin_config_object.DATA.IMG_SIZE)
                    encoder = SwinTransformerV2(input_image_size=input_image_size,
                            pretrained_img_size= swin_config_object.DATA.IMG_SIZE,                    
                            patch_size=4, #swin_config_object.MODEL.SWINV2.PATCH_SIZE,
                            in_chans=3,#swin_config_object.MODEL.SWINV2.IN_CHANS,
                            #num_classes=swin_config_object.MODEL.NUM_CLASSES,
                            embed_dim=swin_config_object.MODEL.SWINV2.EMBED_DIM,
                            depths=swin_config_object.MODEL.SWINV2.DEPTHS,
                            num_heads=swin_config_object.MODEL.SWINV2.NUM_HEADS,
                            window_size=swin_config_object.MODEL.SWINV2.WINDOW_SIZE,
                            #mlp_ratio=swin_config_object.MODEL.SWINV2.MLP_RATIO,
                            #qkv_bias=swin_config_object.MODEL.SWINV2.QKV_BIAS,
                            #drop_rate=swin_config_object.MODEL.DROP_RATE,
                            drop_path_rate=swin_config_object.MODEL.DROP_PATH_RATE,
                            #ape=swin_config_object.MODEL.SWINV2.APE,
                            #patch_norm=swin_config_object.MODEL.SWINV2.PATCH_NORM,
                            #use_checkpoint=swin_config_object.TRAIN.USE_CHECKPOINT,
                            #pretrained_window_sizes=swin_config_object.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES, #dont forget here!!
                            downsample_ratio_beginning=4,)
                    #print(swin_config_object.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
                    print("Input image size : ",input_image_size)
                    print("Pretrained Image Size ",swin_config_object.DATA.IMG_SIZE)
                    self.n_channels[4] = int(swin_config_object.MODEL.SWINV2.EMBED_DIM *8/4/4) #not sure if this general case ?  
                    self.encoder_downsample_ratio = encoder.overall_downsample_ratio //4
                else :
                    embed_dim = 96
                    input_image_size = self.config["use_attention"]["height"],self.config["use_attention"]["width"]
                    encoder = SwinTransformerV2(input_image_size=input_image_size,
                            pretrained_img_size=input_image_size,                    
                            patch_size=4, #swin_config_object.MODEL.SWINV2.PATCH_SIZE,
                            in_chans=3,#swin_config_object.MODEL.SWINV2.IN_CHANS,
                            #num_classes=swin_config_object.MODEL.NUM_CLASSES,
                            embed_dim=embed_dim,
                            depths=[2,2], #2,2,6,2
                            num_heads=[3,6,12,24],
                            window_size=8,
                            #mlp_ratio=swin_config_object.MODEL.SWINV2.MLP_RATIO,
                            #qkv_bias=swin_config_object.MODEL.SWINV2.QKV_BIAS,
                            #drop_rate=swin_config_object.MODEL.DROP_RATE,
                            drop_path_rate=0.2,
                            #ape=swin_config_object.MODEL.SWINV2.APE,
                            #patch_norm=swin_config_object.MODEL.SWINV2.PATCH_NORM,
                            #use_checkpoint=swin_config_object.TRAIN.USE_CHECKPOINT,
                            downsample_ratio_beginning=4,)
                    #print(swin_config_object.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
                    self.n_channels[4] = embed_dim*2 #96*2 since there are only 2 elements in dephts
                    self.encoder_downsample_ratio = 8
            elif self.config["use_attention"]["type"] =="VMamba":
                assert VMAMBA_IMPORT_FLAG, "VMamba not found, please install necessary packages"
                if self.config['use_attention']['pretrained']['check']:
                    mamba_config = get_config(self.config['use_attention']['pretrained']["yaml_file"],self.config["use_attention"]["model_parameters"])
                    self.n_channels[4] = mamba_config.MODEL.VSSM.EMBED_DIM // 2
                    encoder = build_vssm_model(mamba_config)
                    
                    print("Depths : ",mamba_config.MODEL.VSSM.DEPTHS)
                    print("Using pretrained VMAMBA config : ", self.config['use_attention']['pretrained']["yaml_file"])
                else:
                    mypath = "/home/wasproject/Desktop/Can/XPoint/model_weights/encoder-pretrained/vmamba-imagenet/vssm_tiny_segmentation_special/vssm_tiny.yaml" 
                    mamba_config = get_config(mypath)
                    self.n_channels[4] = mamba_config.MODEL.VSSM.EMBED_DIM // 2
                    encoder = build_vssm_model(mamba_config)
                    print("Using raw VMAMBA config : ", mypath)


        else:
                encoder = nn.Sequential(
                *self.getConvolutionBlock(self.n_channels[0], self.n_channels[1]),

                nn.MaxPool2d(2,2),

                *self.getConvolutionBlock(self.n_channels[1], self.n_channels[2]),

                nn.MaxPool2d(2,2),

                *self.getConvolutionBlock(self.n_channels[2], self.n_channels[3]),

                nn.MaxPool2d(2,2),

                *self.getConvolutionBlock(self.n_channels[3], self.n_channels[4]),
            )
            

        return encoder
    




def main():
    model = XPoint()
    
    n = 4  # Desired batch size
    input_data = {
        "optical": {
            "image": torch.rand(n, 1, 512, 640),  # n optical images
            "is_optical": torch.ones(n, 1, dtype=torch.bool),  # n True values
            "homography": torch.eye(3).unsqueeze(0).expand(n, -1, -1)  # Identity homography for n samples
        },
        "thermal": {
            "image": torch.rand(n, 1, 512, 640),  # n thermal images
            "is_optical": torch.zeros(n, 1, dtype=torch.bool),  # n False values
            "homography": torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).unsqueeze(0).expand(n, -1, -1)  # 90 degree rotation homography for n samples
        }
    }
    
    pred_optical, pred_thermal, pred_hm_regression = model(input_data)
    print(pred_hm_regression)
    
    

if __name__ == "__main__":
    main()