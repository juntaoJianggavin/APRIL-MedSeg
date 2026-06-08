# Complete Network Architectures

[中文文档](networks_CN.md)

This project supports 128 complete network architectures (136 registered, size variants merged; 123 standard + 13 text-guided), accessible via the `architecture` field.

## CNN (35)

CNN-based architectures, classic UNet and variants.

| Name | Paper | Published | GitHub | YAML |
|---|---|---|---|---|
| `attention_unet` | Attention U-Net | MIDL 2018 | [ozan-oktay/Attention-Gating-Network](https://github.com/ozan-oktay/Attention-Gating-Network) | [basic](../../configs/architectures/combinations/general/attention_unet_basic.yaml) |
| `unetpp` | UNet++ | DLMIA 2018 | [MrGiovanni/UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus) | [emcad](../../configs/architectures/networks/general/unetpp.yaml) |
| `r2unet` | R2U-Net | IEEE Access 2018 | - | [basic](../../configs/architectures/networks/general/r2unet.yaml) |
| `multiresunet` | MultiResUNet | Neural Networks 2020 | - | [basic](../../configs/architectures/networks/general/multiresunet.yaml) |
| `resunet_a` | ResUNet-a | ISPRS 2020 | - | [basic](../../configs/architectures/networks/general/resunet_a.yaml) |
| `resunetpp` | ResUNet++ | ISM 2019 | - | [basic](../../configs/architectures/networks/general/resunetpp.yaml) |
| `unet3plus` | UNet 3+ | ICASSP 2020 | [ZJUGiveLab/UNet-Version](https://github.com/ZJUGiveLab/UNet-Version) | [basic](../../configs/architectures/networks/general/unet3plus.yaml) |
| `denseunet` | DenseUNet | - | - | [basic](../../configs/architectures/networks/general/denseunet.yaml) |
| `scseunet` | scSE-UNet (Squeeze-Excitation) | MICCAI 2018 | - | [basic](../../configs/architectures/networks/general/scseunet.yaml) |
| `sa_unet` | SA-UNet (Spatial Attention) | IEEE TIM 2021 | - | [basic](../../configs/architectures/networks/general/sa_unet.yaml) |
| `kiunet` | KiU-Net | MICCAI 2020 | [jeya-maria-jose/KiU-Net-pytorch](https://github.com/jeya-maria-jose/KiU-Net-pytorch) | [basic](../../configs/architectures/networks/general/kiunet.yaml) |
| `pan` | PAN (Pyramid Attention Network) | BMVC 2018 | - | [basic](../../configs/architectures/networks/general/pan.yaml) |
| `linknet` | LinkNet | VCIP 2017 | - | [basic](../../configs/architectures/networks/general/linknet.yaml) |
| `pspnet` | PSPNet | CVPR 2017 | - | [basic](../../configs/architectures/networks/general/pspnet.yaml) |
| `fr_unet` | FR-UNet (Full-Resolution) | IEEE TMI 2022 | - | [basic](../../configs/architectures/networks/general/fr_unet.yaml) |
| `dcsaunet` | DCSAU-Net | Computers in Biology and Medicine 2023 | [xq141839/DCSAU-Net](https://github.com/xq141839/DCSAU-Net) | [basic](../../configs/architectures/networks/general/dcsaunet.yaml) |
| `cfanet` | CFA-Net | Computers in Biology and Medicine 2024 | [ZhangJD-ong/CFA-Net](https://github.com/ZhangJD-ong/CFA-Net) | [basic](../../configs/architectures/networks/general/cfanet.yaml) |
| `mednext` | MedNeXt | MICCAI 2023 | [MIC-DKFZ/MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) | [emcad](../../configs/architectures/combinations/general/mednext_emcad.yaml), [cascade_full](../../configs/architectures/combinations/general/mednext_cascade_full.yaml), [cfm](../../configs/architectures/combinations/general/mednext_cfm.yaml) |
| `nnunet_2d` | nnU-Net (2D) | Nature Methods 2021 | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) | [basic](../../configs/architectures/networks/general/nnunet_2d.yaml) |
| `acc_unet` | ACC-UNet | MICCAI 2023 | - | [basic](../../configs/architectures/networks/general/acc_unet.yaml) |
| `cmunext` | CMUNeXt | arXiv 2023 | - | [basic](../../configs/architectures/networks/general/cmunext.yaml) |
| `mew_unet` | MEW-UNet | arXiv 2024 | - | [basic](../../configs/architectures/networks/general/mew_unet.yaml) |
| `lv_unet` | LV-UNet (Lightweight) | - | - | [basic](../../configs/architectures/networks/general/lv_unet.yaml) |
| `ege_unet` | EGE-UNet | arXiv 2023 | [JCruan519/EGE-UNet](https://github.com/JCruan519/EGE-UNet) | [basic](../../configs/architectures/networks/general/ege_unet.yaml) |
| `malunet` | MALUNet | arXiv 2022 | - | [basic](../../configs/architectures/networks/general/malunet.yaml) |
| `lite_unet` | Lite-UNet | - | - | [basic](../../configs/architectures/networks/general/lite_unet.yaml) |
| `mk_unet` | MK-UNet | - | - | [basic](../../configs/architectures/networks/general/mk_unet.yaml) |
| `u_lite` | U-Lite | arXiv 2022 | - | [basic](../../configs/architectures/networks/general/u_lite.yaml) |
| `aau_net` | AAU-Net | IEEE JBHI 2023 | [CGPxy/AAU-net](https://github.com/CGPxy/AAU-net) | [basic](../../configs/architectures/networks/general/aau_net.yaml) |
| `cmu_net` | CMU-Net | Bioinformatics 2024 | - | [basic](../../configs/architectures/networks/general/cmu_net.yaml) |
| `dscnet` | DSCNet | MICCAI 2023 | - | [basic](../../configs/architectures/networks/general/dscnet.yaml) |
| `dconnnet` | DconnNet | MICCAI 2023 | - | [basic](../../configs/architectures/networks/general/dconnnet.yaml) |
| `stu_net` | STU-Net | arXiv 2023 | - | [basic](../../configs/architectures/networks/general/stu_net.yaml) |
| `polyper` | Polyper | - | - | [basic](../../configs/architectures/networks/general/polyper.yaml) |
| `hovernet_lite` | HoverNet Lite | - | - | [basic](../../configs/architectures/networks/general/hovernet_lite.yaml) |

## Transformer (35)

Transformer-based segmentation networks.

| Name | Paper | Published | GitHub | YAML |
|---|---|---|---|---|
| `transunet` | TransUNet | arXiv 2021 | [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet) | [cascade_full](../../configs/architectures/combinations/general/transunet_cascade_full.yaml) |
| `swinunet` | Swin-UNet | ECCV 2022 | [HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) | [segformer](../../configs/architectures/combinations/general/swinunet_segformer.yaml) |
| `medt` | MedT (Medical Transformer) | MICCAI 2021 | [jeya-maria-jose/Medical-Transformer](https://github.com/jeya-maria-jose/Medical-Transformer) | [basic](../../configs/architectures/networks/general/medt.yaml) |
| `daeformer` | DAEFormer | ICLR 2023 | - | [emcad](../../configs/architectures/combinations/general/daeformer_emcad.yaml) |
| `missformer` | MISSFormer | IEEE TMI 2022 | - | [basic](../../configs/architectures/networks/general/missformer.yaml) |
| `h2former` | H2Former | IEEE TMI 2023 | - | [basic](../../configs/architectures/networks/general/h2former.yaml) |
| `hiformer` | HiFormer | WACV 2023 | - | [cascade](../../configs/architectures/combinations/general/hiformer_cascade.yaml) |
| `mctrans` | MCTrans | MICCAI 2021 | - | [cascade_emcad](../../configs/architectures/combinations/general/mctrans_cascade_emcad.yaml) |
| `mtunet` | MT-UNet | MICCAI 2022 | - | [basic](../../configs/architectures/networks/general/mtunet.yaml) |
| `scaleformer` | ScaleFormer | MICCAI 2022 | - | [cascade_full](../../configs/architectures/combinations/general/scaleformer_cascade_full.yaml) |
| `fatnet` | FAT-Net | IEEE TMI 2022 | - | [basic](../../configs/architectures/networks/general/fatnet.yaml) |
| `nnformer_2d` | nnFormer (2D) | MICCAI 2022 | - | [basic](../../configs/architectures/networks/general/nnformer_2d.yaml) |
| `transfuse` | TransFuse | MICCAI 2021 | - | [basic](../../configs/architectures/networks/general/transfuse.yaml) |
| `levit_unet` | LeViT-UNet | ML4H 2022 | - | [basic](../../configs/architectures/networks/general/levit_unet.yaml) |
| `transatt_unet` | TransAttUNet | arXiv 2022 | - | [basic](../../configs/architectures/networks/general/transatt_unet.yaml) |
| `da_transunet` | DA-TransUNet | arXiv 2023 | - | [basic](../../configs/architectures/networks/acdc/da_transunet.yaml) |
| `ds_transunet` | DS-TransUNet | arXiv 2022 | - | [basic](../../configs/architectures/networks/acdc/ds_transunet.yaml) |
| `uctransnet_full` | UCTransNet (full) | AAAI 2022 | - | [uctransnet](../../configs/architectures/combinations/general/uctransnet.yaml) |
| `uctransnet_enc` | UCTransNet (encoder-only) | AAAI 2022 | - | [uctransnet](../../configs/architectures/combinations/general/uctransnet.yaml) |
| `mobile_u_vit` | Mobile-UViT | - | - | [basic](../../configs/architectures/networks/general/mobile_u_vit.yaml) |
| `cswin_unet` | CSWin-UNet | - | - | [basic](../../configs/architectures/networks/general/cswin_unet.yaml) |
| `fcbformer` | FCBFormer | MICCAI 2022 | - | [basic](../../configs/architectures/networks/general/fcbformer.yaml) |
| `pvt_unet` | PVT-UNet | - | - | [emcad](../../configs/architectures/combinations/general/pvtv2_emcad.yaml), [cascade_full](../../configs/architectures/combinations/general/pvtv2_cascade_full.yaml), [cfm](../../configs/architectures/combinations/general/pvtv2_cfm.yaml) |
| `transnetr` | TransNetR | IEEE Access 2023 | - | [basic](../../configs/architectures/networks/general/transnetr.yaml) |
| `polyp_pvt` | Polyp-PVT | MICCAI 2021 | - | [basic](../../configs/architectures/networks/general/polyp_pvt.yaml) |
| `cascade` | CASCADE | MICCAI 2023 | - | [resnet34](../../configs/architectures/combinations/general/cascade_resnet34.yaml) |
| `hsnet` | HSNet | MedIA 2023 | - | [basic](../../configs/architectures/networks/general/hsnet.yaml) |
| `ssformer` | SSFormer | MICCAI 2022 | - | [basic](../../configs/architectures/networks/general/ssformer.yaml) |
| `ldnet` | LDNet | MICCAI 2022 | - | [basic](../../configs/architectures/networks/general/ldnet.yaml) |
| `esfpnet` | ESFPNet | MICCAI 2022 | - | [basic](../../configs/architectures/networks/general/esfpnet.yaml) |
| `mist` | MIST | IEEE TMI 2023 | - | [basic](../../configs/architectures/networks/general/mist.yaml) |
| `double_unet` | DoubleU-Net | CBMS 2020 | - | [basic](../../configs/architectures/networks/general/double_unet.yaml) |
| `sepnet` | SEPNet | - | - | [basic](../../configs/architectures/networks/general/sepnet.yaml) |
| `ctnet` | CTNet | - | - | [basic](../../configs/architectures/networks/general/ctnet.yaml) |
| `nulite` | NuLite | - | - | [basic](../../configs/architectures/networks/general/nulite.yaml) |

## Mamba / SSM (24)

Mamba / State-Space Model based networks.

| Name | Paper | Published | YAML |
|---|---|---|---|
| `mamba_unet` | Mamba-UNet | arXiv 2024 | [basic](../../configs/architectures/networks/general/mamba_unet.yaml) |
| `h_vmunet` | H-vmunet | arXiv 2024 | [basic](../../configs/architectures/networks/general/h_vmunet.yaml) |
| `lightm_unet` | LightM-UNet | arXiv 2024 | [basic](../../configs/architectures/networks/general/lightm_unet.yaml) |
| `swin_umamba` | Swin-UMamba | arXiv 2024 | [basic](../../configs/architectures/networks/general/swin_umamba.yaml) |
| `umamba_bot` | U-Mamba (bottleneck) | arXiv 2024 | [cascade_full](../../configs/architectures/combinations/general/umamba_cascade_full.yaml), [cfm](../../configs/architectures/combinations/general/umamba_cfm.yaml), [emcad](../../configs/architectures/combinations/general/umamba_emcad.yaml) |
| `umamba_enc` | U-Mamba (encoder) | arXiv 2024 | [cascade_full](../../configs/architectures/combinations/general/umamba_cascade_full.yaml) |
| `ultralight_vmunet` | UltraLight VM-UNet | arXiv 2024 | [basic](../../configs/architectures/networks/general/ultralight_vmunet.yaml) |
| `vm_unet` | VM-UNet | arXiv 2024 | [basic](../../configs/architectures/networks/general/vm_unet.yaml) |
| `vm_unet_v2` | VM-UNet V2 | arXiv 2024 | [basic](../../configs/architectures/networks/general/vm_unet_v2.yaml) |
| `lkm_unet` | LKM-UNet | arXiv 2024 | [basic](../../configs/architectures/networks/general/lkm_unet.yaml) |
| `log_vmamba` | LoG-VMamba | arXiv 2024 | [basic](../../configs/architectures/networks/general/log_vmamba.yaml) |
| `vmkla_unet` | VMKLA-UNet | arXiv 2024 | [basic](../../configs/architectures/networks/general/vmkla_unet.yaml) |
| `ultralbm_unet` | UltraLBM-UNet | arXiv 2024 | [basic](../../configs/architectures/networks/general/ultralbm_unet.yaml) |
| `nnmamba_2d` | nnMamba (2D) | arXiv 2024 | [basic](../../configs/architectures/networks/general/nnmamba_2d.yaml) |
| `polyp_mamba` | Polyp-Mamba | arXiv 2024 | [basic](../../configs/architectures/networks/general/polyp_mamba.yaml) |
| `hc_mamba` | HC-Mamba | arXiv 2024 | [basic](../../configs/architectures/networks/general/hc_mamba.yaml) |
| `ac_mambaseg` | AC-MambaSeg | arXiv 2024 | [basic](../../configs/architectures/networks/general/ac_mambaseg.yaml) |
| `dcm_net` | DCM-Net | arXiv 2024 | [basic](../../configs/architectures/networks/general/dcm_net.yaml) |
| `dermomamba` | DermoMamba | arXiv 2024 | [basic](../../configs/architectures/networks/general/dermomamba.yaml) |
| `mucm_net` | MUCM-Net | arXiv 2024 | [basic](../../configs/architectures/networks/general/mucm_net.yaml) |
| `serp_mamba` | Serp-Mamba | arXiv 2024 | [basic](../../configs/architectures/networks/general/serp_mamba.yaml) |
| `skin_mamba` | SkinMamba | arXiv 2024 | [basic](../../configs/architectures/networks/general/skin_mamba.yaml) |
| `mamba_vesselnet_pp` | Mamba-VesselNet++ | arXiv 2024 | [basic](../../configs/architectures/networks/general/mamba_vesselnet_pp.yaml) |
| `vim_unet` | ViM-UNet | arXiv 2024 | [basic](../../configs/architectures/networks/general/vim_unet.yaml) |
| `uu_mamba` | UU-Mamba | arXiv 2024 | [basic](../../configs/architectures/networks/general/uu_mamba.yaml) |

## SAM (10)

SAM (Segment Anything Model) based networks.

| Name | Paper | Published | YAML |
|---|---|---|---|
| `sam_b` | SAM ViT-Base | ICCV 2023 | [cascade_full](../../configs/architectures/combinations/general/sam_vit_cascade_full.yaml), [cfm](../../configs/architectures/combinations/general/sam_vit_cfm.yaml), [emcad](../../configs/architectures/combinations/general/sam_vit_emcad.yaml) |
| `sam_l` | SAM ViT-Large | ICCV 2023 | [cascade_full](../../configs/architectures/combinations/general/sam_vit_cascade_full.yaml) |
| `mobile_sam` | MobileSAM | arXiv 2023 | [basic](../../configs/architectures/networks/general/mobile_sam.yaml) |
| `sam2` | SAM 2 | arXiv 2024 | [basic](../../configs/architectures/networks/general/sam2.yaml) |
| `medsam` | MedSAM | Nature Comms 2024 | [emcad](../../configs/architectures/combinations/general/medsam_encoder_emcad.yaml) |
| `samus` | SAMUS | arXiv 2023 | [basic](../../configs/architectures/networks/general/samus.yaml) |
| `sam_med2d` | SAM-Med2D | arXiv 2023 | [basic](../../configs/architectures/networks/general/sam_med2d.yaml) |
| `sammed2d_wrapper` | SAMMed2D (wrapper) | arXiv 2023 | [qata_covid19](../../configs/architectures/foundation/sam/qata_covid19_sammed2d.yaml) |
| `medical_sam_adapter` | Medical SAM Adapter | arXiv 2023 | [basic](../../configs/architectures/networks/general/medical_sam_adapter.yaml) |
| `samed` | SAMed | arXiv 2023 | [basic](../../configs/architectures/networks/general/samed.yaml) |
| `auto_sam` | AutoSAM | arXiv 2023 | [basic](../../configs/architectures/networks/general/auto_sam.yaml) |
| `lite_medsam` | Lite-MedSAM | arXiv 2024 | [qata_covid19](../../configs/architectures/foundation/sam/qata_covid19_lite_medsam.yaml) |

## KAN / MLP (4)

| Name | Paper | Published | YAML |
|---|---|---|---|
| `ukan` | U-KAN | arXiv 2024 | [basic](../../configs/architectures/networks/general/ukan.yaml) |
| `wav_kan_unet` | Wav-KAN UNet | arXiv 2024 | [basic](../../configs/architectures/networks/general/wav_kan_unet.yaml) |
| `unext` | UNeXt | MICCAI 2022 | [basic](../../configs/architectures/networks/general/unext.yaml) |
| `rolling_unet` | Rolling-UNet | arXiv 2024 | [basic](../../configs/architectures/networks/general/rolling_unet.yaml) |
| `rolling_unet_s` | Rolling-UNet (small) | arXiv 2024 | [basic](../../configs/architectures/networks/general/rolling_unet_s.yaml) |
| `rolling_unet_m` | Rolling-UNet (medium) | arXiv 2024 | [basic](../../configs/architectures/networks/general/rolling_unet_m.yaml) |
| `rolling_unet_l` | Rolling-UNet (large) | arXiv 2024 | [basic](../../configs/architectures/networks/general/rolling_unet_l.yaml) |

## RWKV (4)

| Name | Paper | Published | YAML |
|---|---|---|---|
| `u_rwkv` | U-RWKV | arXiv 2024 | [unet](../../configs/architectures/combinations/general/rwkv_unet.yaml), [small](../../configs/architectures/combinations/general/rwkv_unet_small.yaml), [tiny](../../configs/architectures/combinations/general/rwkv_unet_tiny.yaml) |
| `rwkv_unet` | RWKV-UNet | arXiv 2024 | [emcad](../../configs/architectures/combinations/general/rwkv_emcad.yaml), [cascade_full](../../configs/architectures/combinations/general/rwkv_cascade_full.yaml), [cfm](../../configs/architectures/combinations/general/rwkv_cfm.yaml) |
| `md_rwkv_unet` | MD-RWKV-UNet | arXiv 2024 | [basic](../../configs/architectures/networks/general/md_rwkv_unet.yaml) |
| `rir_zigzag` | RIR-Zigzag | arXiv 2024 | [yaml](../../configs/architectures/combinations/general/rir_zigzag.yaml) |

## Linear Attention (3)

| Name | Paper | Published | YAML |
|---|---|---|---|
| `ttt_unet` | TTT-UNet | arXiv 2024 | [basic](../../configs/architectures/networks/general/ttt_unet.yaml) |
| `xlstm_unet_bot` / `xlstm_unet_enc` | xLSTM-UNet | arXiv 2024 | [bot](../../configs/architectures/networks/general/xlstm_unet_bot.yaml), [enc](../../configs/architectures/networks/general/xlstm_unet_enc.yaml) |
| `u_vixlstm` | U-VixLSTM | arXiv 2024 | [basic](../../configs/architectures/networks/general/u_vixlstm.yaml) |

## Text-guided (13)

Text-guided segmentation models with forward signature `(image, text=None)`.

| Name | Paper | Published | YAML |
|---|---|---|---|
| `tganet` | TGANet | MICCAI 2022 | [synapse_clip](../../configs/training_paradigms/text_guided/synapse_clip.yaml) |
| `lvit` | LViT | IEEE TMI 2023 | [mosmed_plus_lvit](../../configs/training_paradigms/text_guided/mosmed_plus_lvit.yaml), [qata_covid19_lvit](../../configs/training_paradigms/text_guided/qata_covid19_lvit.yaml) |
| `languide` | LanGuideMedSeg | MICCAI 2023 | [mosmed_plus_languide](../../configs/training_paradigms/text_guided/mosmed_plus_languide.yaml), [qata_covid19_languide](../../configs/training_paradigms/text_guided/qata_covid19_languide.yaml) |
| `clip_universal` | CLIP-Driven Universal Model | ICCV 2023 | [synapse_clip_large](../../configs/training_paradigms/text_guided/synapse_clip_large.yaml) |
| `cris` | CRIS | CVPR 2022 | [synapse_clip](../../configs/training_paradigms/text_guided/synapse_clip.yaml) |
| `biomedparse` | BiomedParse | Nature Methods 2024 | - |
| `tpro` | TPRO | ECCV 2024 | - |
| `salip` | SaLIP | arXiv 2024 | - |
| `causal_clipseg` | Causal CLIPSeg | arXiv 2024 | - |
| `medclip_sam` | MedCLIP-SAM | arXiv 2024 | [synapse_grounding_dino_medsam](../../configs/training_paradigms/text_guided/synapse_grounding_dino_medsam.yaml) |
| `tp_drseg` | TP-DRSeg | arXiv 2024 | - |
| `cxrclipseg` | CXR-CLIPSeg | arXiv 2024 | - |
| `medisee` | MediSee (MLLM) | arXiv 2024 | [mosmed_plus_medisee](../../configs/training_paradigms/text_guided/mosmed_plus_medisee.yaml), [qata_covid19_medisee](../../configs/training_paradigms/text_guided/qata_covid19_medisee.yaml) |

## YAML Usage Example

```yaml
model:
  num_classes: 9
  img_size: 224
  architecture: transunet
  encoder:
    in_channels: 3
  arch_params: {}

data:
  type: synapse
  img_size: 224
  train_dir: ./data/Synapse/train_npz
  test_dir: ./data/Synapse/test_vol_h5
  train_list: ./data/Synapse/lists/lists_Synapse/train.txt
  test_list: ./data/Synapse/lists/lists_Synapse/test_vol.txt

training:
  epochs: 200
  batch_size: 16
  num_workers: 4
  loss:
    name: compound
    params:
      losses:
        - name: ce
          weight: 0.4
        - name: dice
          weight: 0.6
  optimizer:
    name: adamw
    lr: 0.0001
    weight_decay: 0.01
  scheduler:
    name: cosine
    min_lr: 0.000001
```
