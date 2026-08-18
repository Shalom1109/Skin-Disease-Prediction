import torch
import torch.nn as nn
import timm

class SkinSenseEnsemble(nn.Module):
    def __init__(self, num_classes=7, weight_convnext=0.4, weight_vit=0.3, weight_maxvit=0.3):
        super(SkinSenseEnsemble, self).__init__()
        # 1. ConvNeXt Base (Strong Feature Extractor)
        self.convnext = timm.create_model('convnext_base', pretrained=True, num_classes=num_classes)
        # 2. ViT Base (Global Context)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes, img_size=384)
        # 3. MaxViT Base (Hybrid local-global attention - SOTA for >99%)
        self.maxvit = timm.create_model('maxvit_base_tf_384.in1k', pretrained=True, num_classes=num_classes)
        
        # EXTREME FREEZING LOGIC
        # Freeze ALL deep parameters
        for model in [self.convnext, self.vit, self.maxvit]:
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze ONLY the final classification head
            for param in model.get_classifier().parameters():
                param.requires_grad = True
        
        self.w1 = weight_convnext
        self.w2 = weight_vit
        self.w3 = weight_maxvit
        
    def forward(self, x):
        out_convnext = self.convnext(x)
        out_vit = self.vit(x)
        out_maxvit = self.maxvit(x)
        
        # Weighted Triple Ensemble
        return (self.w1 * out_convnext + self.w2 * out_vit + self.w3 * out_maxvit)

def get_model(num_classes=7, checkpoint_path=None):
    model = SkinSenseEnsemble(num_classes=num_classes)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model
