import torch
import torch.nn as nn
import timm


def _unfreeze(module):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = True


class SkinSenseEnsemble(nn.Module):
    """Weighted ensemble of ConvNeXt-B, ViT-B and MaxViT-B (384 px input).

    pretrained     : load ImageNet weights. Training needs this; inference does not
                     (the checkpoint overwrites every weight), so get_model() passes False.
    finetune_last_stage : besides the classifier heads, also train the last
                     stage/block of every backbone (much better domain adaptation
                     than heads-only; needs more VRAM).
    """

    def __init__(self, num_classes=7, weight_convnext=0.4, weight_vit=0.3, weight_maxvit=0.3,
                 pretrained=True, finetune_last_stage=False):
        super().__init__()
        self.convnext = timm.create_model('convnext_base', pretrained=pretrained, num_classes=num_classes)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained,
                                     num_classes=num_classes, img_size=384)
        self.maxvit = timm.create_model('maxvit_base_tf_384.in1k', pretrained=pretrained,
                                        num_classes=num_classes)

        for net in (self.convnext, self.vit, self.maxvit):
            for p in net.parameters():
                p.requires_grad = False
            _unfreeze(net.get_classifier())

        if finetune_last_stage:
            _unfreeze(self.convnext.stages[-1])
            _unfreeze(self.convnext.head)
            _unfreeze(self.vit.blocks[-1])
            _unfreeze(self.vit.norm)
            _unfreeze(getattr(self.vit, 'fc_norm', None))
            _unfreeze(self.maxvit.stages[-1])
            _unfreeze(self.maxvit.head)

        self.w1, self.w2, self.w3 = weight_convnext, weight_vit, weight_maxvit

    def head_parameters(self):
        for net in (self.convnext, self.vit, self.maxvit):
            yield from net.get_classifier().parameters()

    def backbone_trainable_parameters(self):
        head_ids = {id(p) for p in self.head_parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in head_ids]

    def forward_parts(self, x):
        """Logits of the three member networks (used for disagreement-based uncertainty)."""
        return self.convnext(x), self.vit(x), self.maxvit(x)

    def forward(self, x):
        a, b, c = self.forward_parts(x)
        return self.w1 * a + self.w2 * b + self.w3 * c


def load_state(path, map_location='cpu'):
    """Load a plain state_dict or a training checkpoint ({'model_state_dict': ...}).
    weights_only=True avoids executing arbitrary pickled code."""
    state = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    return state


def get_model(num_classes=7, checkpoint_path=None, pretrained=False):
    model = SkinSenseEnsemble(num_classes=num_classes, pretrained=pretrained)
    if checkpoint_path:
        model.load_state_dict(load_state(checkpoint_path))
    model.eval()
    return model
