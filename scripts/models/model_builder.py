import torch
import torch.nn as nn
import os

def build_model(num_classes, is_16bit=False, backbone="swin-large", model_arch="mask2former"):
    """
    Builds the segmentation model based on the chosen architecture.
    """
    in_channels = 4 if is_16bit else 3
    print(f"Building Model Architecture: {model_arch.upper()} with in_channels={in_channels}, num_classes={num_classes}")

    if model_arch == "unet":
        import segmentation_models_pytorch as smp
        
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
        
        # 통일된 MC Dropout 프로토콜: 최종 분류기(segmentation_head) 직전 1곳에만 Dropout 적용
        model.final_dropout = nn.Dropout(p=0.1)
        original_forward = model.segmentation_head.forward
        def patched_forward(self, *args, **kwargs):
            if len(args) > 0:
                args = list(args)
                if model.final_dropout.training:
                    args[0] = model.final_dropout(args[0])
            return original_forward(*args, **kwargs)
        model.segmentation_head.forward = patched_forward.__get__(model.segmentation_head)

        model.gradient_checkpointing_enable = lambda: print("Gradient Checkpointing not natively supported for SMP models, skipping.")
        return model

    elif model_arch == "deeplabv3plus":
        import segmentation_models_pytorch as smp
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
        
        # 통일된 MC Dropout 프로토콜: 최종 분류기(segmentation_head) 직전 1곳에만 Dropout 적용
        model.final_dropout = nn.Dropout(p=0.1)
        original_forward = model.segmentation_head.forward
        def patched_forward(self, *args, **kwargs):
            if len(args) > 0:
                args = list(args)
                if model.final_dropout.training:
                    args[0] = model.final_dropout(args[0])
            return original_forward(*args, **kwargs)
        model.segmentation_head.forward = patched_forward.__get__(model.segmentation_head)

        model.gradient_checkpointing_enable = lambda: print("Gradient Checkpointing not natively supported for SMP models, skipping.")
        return model

    elif model_arch == "segformer":
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
        
        # We use MiT-b2 by default
        repo_id = "nvidia/mit-b2"
        try:
            config = SegformerConfig.from_pretrained(repo_id)
        except Exception:
            config = SegformerConfig()
            
        config.num_labels = num_classes
        model = SegformerForSemanticSegmentation.from_pretrained(repo_id, config=config, ignore_mismatched_sizes=True)
        
        # Handle 4-channels
        if in_channels != 3:
            first_layer = model.segformer.encoder.patch_embeddings[0].proj
            old_weights = first_layer.weight.data
            new_layer = nn.Conv2d(in_channels, first_layer.out_channels, 
                                  kernel_size=first_layer.kernel_size, 
                                  stride=first_layer.stride, 
                                  padding=first_layer.padding)
            new_layer.weight.data[:, :3, :, :] = old_weights
            new_layer.weight.data[:, 3:, :, :] = old_weights.mean(dim=1, keepdim=True)
            model.segformer.encoder.patch_embeddings[0].proj = new_layer

        # 통일된 MC Dropout 프로토콜: 최종 분류기(classifier) 직전 1곳에만 Dropout 적용
        model.final_dropout = nn.Dropout(p=0.1)
        original_forward = model.decode_head.classifier.forward
        def patched_forward(self, *args, **kwargs):
            if len(args) > 0:
                args = list(args)
                if model.final_dropout.training:
                    args[0] = model.final_dropout(args[0])
            return original_forward(*args, **kwargs)
        model.decode_head.classifier.forward = patched_forward.__get__(model.decode_head.classifier)

        # Segformer doesn't have gradient_checkpointing_enable natively without wrapping, add dummy
        if not hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable = lambda: print("Gradient Checkpointing not natively supported for Segformer here, skipping.")
        return model

    else:
        # Default: Mask2Former
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
        
        if backbone == "swin-tiny":
            offline_folder = "mask2former-swin-tiny"
            repo_id = "facebook/mask2former-swin-tiny-cityscapes-semantic"
        else:
            offline_folder = "mask2former-swin-large"
            repo_id = "facebook/mask2former-swin-large-cityscapes-semantic"
            
        local_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "offline_setup", "models", offline_folder))
        if not os.path.exists(local_model_dir):
            workspace_dir = rf"D:\SU_work\QGIS_GeoActive_ML_Studio\offline_setup\models\{offline_folder}"
            if os.path.exists(workspace_dir):
                local_model_dir = workspace_dir
                
        if os.path.exists(local_model_dir):
            config = Mask2FormerConfig.from_pretrained(local_model_dir)
        else:
            print(f"Warning: Local model dir not found. Falling back to online download.")
            config = Mask2FormerConfig.from_pretrained(repo_id)
            
        config.num_labels = num_classes
        
        if os.path.exists(local_model_dir):
            model = Mask2FormerForUniversalSegmentation.from_pretrained(local_model_dir, config=config, ignore_mismatched_sizes=True)
        else:
            model = Mask2FormerForUniversalSegmentation.from_pretrained(repo_id, config=config, ignore_mismatched_sizes=True)
        
        if in_channels != 3:
            first_layer = model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection
            old_weights = first_layer.weight.data
            new_layer = nn.Conv2d(in_channels, first_layer.out_channels, 
                                  kernel_size=first_layer.kernel_size, 
                                  stride=first_layer.stride, 
                                  padding=first_layer.padding)
            new_layer.weight.data[:, :3, :, :] = old_weights
            new_layer.weight.data[:, 3:, :, :] = old_weights.mean(dim=1, keepdim=True)
            model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection = new_layer

        # 통일된 MC Dropout 프로토콜: 최종 쿼리 분류기 및 마스크 예측기 직전 1곳에만 Dropout 적용
        model.final_dropout = nn.Dropout(p=0.1)
        
        original_class_forward = model.class_predictor.forward
        def patched_class_forward(self, *args, **kwargs):
            if len(args) > 0:
                args = list(args)
                if model.final_dropout.training:
                    args[0] = model.final_dropout(args[0])
            return original_class_forward(*args, **kwargs)
        model.class_predictor.forward = patched_class_forward.__get__(model.class_predictor)
        
        # 마스크 예측기 패치
        mask_pred_module = model.model.transformer_module.decoder.mask_predictor
        original_mask_forward = mask_pred_module.forward
        def patched_mask_forward(self, *args, **kwargs):
            if len(args) > 0:
                args = list(args)
                if model.final_dropout.training:
                    args[0] = model.final_dropout(args[0])
            return original_mask_forward(*args, **kwargs)
        mask_pred_module.forward = patched_mask_forward.__get__(mask_pred_module)

        return model
