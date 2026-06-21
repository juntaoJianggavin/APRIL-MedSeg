# Decoders (`medseg/models/decoders/`)

[中文文档](README_CN.md)

U-shape upsampling decoders that consume multi-scale encoder features and produce the final segmentation map. All decoders follow a unified interface: `forward(bottleneck_feat, skip_features)`.

## Registered Decoders (47 keys)

### basic/ — Basic Upsampling

| Registry Key | Source File | Description |
|---|---|---|
| `unet` / `deconv_upcat` | `unet_decoder.py` | UNet decoder: ConvTranspose (halve ch) → concat skip → DoubleConv. Alias `deconv_upcat` = **up**sample-then-**cat** order |
| `bilinear` | `bilinear_decoder.py` | Simple bilinear upsample + 1×1 conv (fastest baseline) |
| `deconv` / `deconv_catup` | `deconv_decoder.py` | Transposed-conv decoder: concat skip → ConvTranspose (merge ch) → conv. Alias `deconv_catup` = **cat**-then-**up**sample order |
| `dw_sep` | `dw_sep_decoder.py` | Depthwise-separable conv decoder (lightweight) |

### dense/ — Dense Connection

| Registry Key | Source File | Description |
|---|---|---|
| `unetpp` | `unetpp_decoder.py` | UNet++ dense skip decoder |
| `unet3plus` | `unet3plus_decoder.py` | UNet3+ full-scale skip decoder |

### cascade/ — Cascade Refinement

| Registry Key | Source File | Description |
|---|---|---|
| `cascade` | `cascade_decoder.py` | Cascaded refinement decoder |
| `cascade_full` | `cascade_full_decoder.py` | Full cascaded decoder (all stages cascaded) |
| `cascade_emcad` | `cascade_emcad_decoder.py` | CASCADE with EMCAD aggregation |
| `cfm` | `cfm_decoder.py` | CFM (Cascade Fusion Module) decoder |
| `emcad` | `emcad_decoder.py` | EMCAD multi-scale aggregation decoder |
| `edldnet` | `edldnet_decoder.py` | EDLDNet decoder |
| `gcascade` | `gcascade_decoder.py` | G-CASCADE decoder (gate-based cascade) |
| `gcascade_cat` | `gcascade_decoder.py` | G-CASCADE decoder (concat variant) |
| `merit_add` | `merit_decoder.py` | MERIT decoder (additive fusion) |
| `merit_cat` | `merit_decoder.py` | MERIT decoder (concat fusion) |

### pyramid/ — Pyramid Aggregation

| Registry Key | Source File | Description |
|---|---|---|
| `upernet` | `upernet_decoder.py` | UPerNet FPN + PPM decoder |
| `deeplabv3` | `deeplabv3_decoder.py` | DeepLabV3 ASPP-based decoder |

### mlp/ — MLP-Based

| Registry Key | Source File | Description |
|---|---|---|
| `mlp` | `mlp_decoder.py` | MLP-based decoder (SegFormer-style) |
| `segformer` | `segformer_decoder.py` | SegFormer MLP decoder |

### transformer/ — Transformer (Q/K/V self-attention or cross-attention blocks)

| Registry Key | Source File | Description |
|---|---|---|
| `daeformer` | `daeformer_decoder.py` | DAEFormer cross-attention decoder (CrossAttentionBlock) |
| `missformer` | `missformer_decoder.py` | MISSFormer Transformer decoder (PatchExpand + TransformerBlock) |
| `mtunet` | `mtunet_decoder.py` | MT-UNet decoder (window + axial + memory-efficient attention) |
| `nnformer` | `nnformer_decoder.py` | nnFormer decoder (PatchExpand + Swin window attention) |
| `swinunet` | `swinunet_decoder.py` | Swin-UNet patch-expanding decoder (SwinTransformerBlock) |

### attention/ — Attention (gates, spatial/channel attention, context aggregation)

| Registry Key | Source File | Description |
|---|---|---|
| `attention` | `attention_decoder.py` | Attention gate at each skip level (Attention U-Net) |
| `banet` | `banet_decoder.py` | BANet boundary-aware spatial attention decoder |
| `ccnet` | `ccnet_decoder.py` | CCNet criss-cross attention decoder (Q/K/V + softmax) |
| `lawin` | `lawin_decoder.py` | Lawin large-window attention decoder + spatial pyramid |
| `ocrnet` | `ocrnet_decoder.py` | OCRNet object-contextual representations (Q/K/V attention for context) |
| `uctransnet` | `uctransnet_decoder.py` | UCTransNet channel-wise cross-attention (CCA, SE-Net style) + bilinear upsample |

### mamba/ — Mamba

| Registry Key | Source File | Description |
|---|---|---|
| `vmunet` | `vmunet_decoder.py` | VM-UNet Mamba-SSM decoder |

### specific/ — Network-Specific (unique mechanisms)

| Registry Key | Source File | Description |
|---|---|---|
| `cfanet` | `cfanet_decoder.py` | CFA-Net cross-level feature aggregation decoder |
| `dcsaunet` | `dcsaunet_decoder.py` | DCSAU-Net dual-path decoder |
| `ege_unet` | `ege_unet_decoder.py` | EGNet edge-guided decoder |
| `fatnet` | `fatnet_decoder.py` | FAT-Net frequency-aware Transformer decoder |
| `ffparser` | `ffparser_decoder.py` | FF-Parser frequency-domain feature filtering decoder |
| `h2former` | `h2former_decoder.py` | H2Former bilinear upsample + concat decoder |
| `ham` | `ham_decoder.py` | HAM (Hamburger) NMF-based global context modeling decoder |
| `hiformer` | `hiformer_decoder.py` | HiFormer ConvUpsample decoder |
| `kiunet` | `kiunet_decoder.py` | KI-UNet decoder |
| `malunet` | `malunet_decoder.py` | MALUNet mixed-skip decoder |
| `rwkv_unet` | `rwkv_unet_decoder.py` | RWKV sequence-model decoder |
| `scaleformer` | `scaleformer_decoder.py` | ScaleFormer 4-stage UNet decoder |
| `transunet` | `transunet_decoder.py` | TransUNet cascaded up-conv decoder |

## Usage in YAML Config

```yaml
model:
  decoder:
    name: attention         # any registered key
    params:
      encoder_channels: [64, 256, 512, 1024, 2048]
      decoder_channels: [256, 128, 64, 32, 16]
```
