import torch
from torch import nn
from torchvision.transforms import transforms


class ReplaceRandomPoint(nn.Module):
    def __init__(self, num_to_replace=1, scale=1, DEBUG=False):
        self.num_to_replace = num_to_replace
        self.scale = scale
        self.DEBUG = DEBUG

    def forward(self, pts):
        i_to_replace = torch.randperm(pts.shape[0])[:self.num_to_replace]
        pt_sz = pts[0].shape
        if self.DEBUG:
            print(i_to_replace)
            print("Original points: ", pts)

        for i in i_to_replace:
            new_pt = self.scale * torch.rand(pt_sz)
            pts[i, :] = new_pt
            if self.DEBUG:
                print(new_pt)

        return pts


class ZeroRandomPoint(nn.Module):
    # TODO: better way to mask them?
    def __init__(self, num_to_replace=1):
        super(ZeroRandomPoint, self).__init__()
        self.num_to_replace = num_to_replace

    def forward(self, pts):
        assert pts.dim() == 3
        i_to_zero = torch.randperm(pts.shape[1])[:self.num_to_replace]
        for i in i_to_zero:
            pts[:, i, :] = 0.0
        return pts


class JitterAllPoints(nn.Module):
    # Random scaling, different for each point
    def __init__(self, percent_jitter=0.1):
        super(JitterAllPoints, self).__init__()
        self.percent_jitter = percent_jitter

    def forward(self, pts):
        jitter = 1.0 + self.percent_jitter * torch.rand(pts.shape)
        return pts * jitter


class TranslateAllPoints(nn.Module):
    # Random scaling, different for each point
    def __init__(self, percent_translation=0.1):
        super(TranslateAllPoints, self).__init__()
        self.percent_translation = percent_translation

    def forward(self, pts):
        translation = self.percent_translation * torch.rand(1) * torch.mean(pts)
        return pts + translation


class PointCloudAugmentationDINO(object):
    """
    Following augmentations:
        Replace neuron with outlier
        Add jitter to location
        Scale
    TODO:
        RandomAffine
        (see: https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html)
    """

    def __init__(self,
                 teacher_num_to_replace=0, student_num_to_replace=1,
                 teacher_percent_jitter=0.01, student_percent_jitter=0.1,
                 teacher_percent_translation=0.01, student_percent_translation=0.1,
                 student_crops_number=2):
        normalize = nn.Sequential(
            transforms.Normalize(0.0, 0.229),
        )

        replace_teacher_random = torch.jit.script(ZeroRandomPoint(num_to_replace=teacher_num_to_replace))
        replace_student_random = torch.jit.script(ZeroRandomPoint(num_to_replace=student_num_to_replace))
        jitter_teacher_random = torch.jit.script(JitterAllPoints(percent_jitter=teacher_percent_jitter))
        jitter_student_random = torch.jit.script(JitterAllPoints(percent_jitter=student_percent_jitter))
        translate_teacher_random = torch.jit.script(TranslateAllPoints(percent_translation=teacher_percent_translation))
        translate_student_random = torch.jit.script(TranslateAllPoints(percent_translation=student_percent_translation))

        # first global crop - NO replacement, only jitter and translation
        self.global_transfo1 = nn.Sequential(
            jitter_teacher_random,
            normalize,
        )
        # second global crop
        self.global_transfo2 = nn.Sequential(
            jitter_teacher_random,
            translate_teacher_random,
            replace_teacher_random,
            normalize,
        )
        # transformation for the local small crops
        self.student_crops_number = student_crops_number
        self.local_transfo = nn.Sequential(
            jitter_student_random,
            translate_student_random,
            replace_student_random,
            normalize,
        )

    def __call__(self, pts):
        crops = []
        crops.append(self.global_transfo1(pts))
        crops.append(self.global_transfo2(pts))
        for _ in range(self.student_crops_number):
            crops.append(self.local_transfo(pts))
        # Return full shape
        crops = torch.cat([torch.unsqueeze(c, dim=0).float() for c in crops], dim=0)
        # for c in crops:
        #     c = torch.unsqueeze(c, dim=0)
        return crops
