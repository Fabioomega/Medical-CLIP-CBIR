import torch
from transformer_maskgit.transformer_maskgit import CTViT
from CT_CLIP.ct_clip import CTCLIP
import accelerate
import numpy as np
import nibabel as nib


def strip_text_transformer_from_dict(state_dict):
    l = list(state_dict.keys())
    for key in l:
        if "text_transformer" in key:
            del state_dict[key]
    return state_dict


def get_img_latents(img, model) -> torch.Tensor:
    _, image_latents, _ = model(
        None, img, return_latents=True, device=torch.device("cuda")
    )
    return image_latents


def nii_img_to_tensor(path):
    img_data = nib.loadsave.load(path)
    img_data = img_data.get_fdata()
    img_data = np.transpose(img_data, (1, 2, 0))
    img_data = img_data * 1000
    hu_min, hu_max = -1000, 200
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = (((img_data + 400) / 600)).astype(np.float32)
    slices = []
    tensor = torch.tensor(img_data)
    # Get the dimensions of the input tensor
    target_shape = (480, 480, 240)
    # Extract dimensions
    h, w, d = tensor.shape
    # Calculate cropping/padding values for height, width, and depth
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)
    # Crop or pad the tensor
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]
    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before
    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before
    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before
    tensor = torch.nn.functional.pad(
        tensor,
        (
            pad_d_before,
            pad_d_after,
            pad_w_before,
            pad_w_after,
            pad_h_before,
            pad_h_after,
        ),
        value=-1,
    )
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)
    return tensor


image_encoder = CTViT(
    dim=512,
    codebook_size=8192,
    image_size=480,
    patch_size=20,
    temporal_patch_size=10,
    spatial_depth=4,
    temporal_depth=4,
    dim_head=32,
    heads=8,
)

clip = CTCLIP(
    image_encoder=image_encoder,
    text_encoder=None,
    dim_image=294912,
    dim_text=768,
    dim_latent=512,
    extra_latent_projection=False,  # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds=False,
    use_all_token_embeds=False,
    disable_text_encoder=True,
).cuda()
clip = clip.eval()

cos = torch.nn.CosineSimilarity(dim=0)

img1 = nii_img_to_tensor(
    r"data_volumes\dataset\valid\valid_1\valid_1_a\valid_1_a_1.nii.gz"
).cuda()
img2 = nii_img_to_tensor(
    r"data_volumes\dataset\valid\valid_1\valid_1_a\valid_1_a_2.nii.gz"
).cuda()

clip.load_state_dict(strip_text_transformer_from_dict(torch.load("CT-CLIP_v2.pt")))

# interesting_features = [139, 403, 157, 415, 159, 446, 62, 69, 453, 455, 462, 341, 345, 479, 233, 361, 366, 111, 371]
# bad_features = [385, 265, 266, 11, 393, 272, 276, 404, 406, 155, 33, 34, 163, 420, 421, 296, 424, 298, 44, 172, 51, 179, 437, 310, 311, 438, 187, 322, 325, 327, 203, 460, 333, 467, 475, 489, 109, 368, 503, 507, 382]

scale_up = 1.4
scale_down = 0

with torch.no_grad():

    imgs = [img1, img2]

    l1, l2 = get_img_latents(torch.stack(imgs), clip).chunk(len(imgs))
    l1, l2 = l1.squeeze(), l2.squeeze()
    # l1[interesting_features] *= scale_up
    # l2[interesting_features] *= scale_up
    # l1[bad_features] *= scale_down
    # l2[bad_features] *= scale_down

    # print((l1 + l2).argsort().squeeze()[-128:])

    print(cos(l1, l2))
