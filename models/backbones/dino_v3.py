import timm


vit_model_to_out_indices = {
    'vit_small_patch16_dinov3.lvd1689m': (3, 5, 7, 11),
    'vit_small_plus_patch16_dinov3.lvd1689m': (3, 5, 7, 11),
    'vit_base_patch16_dinov3.lvd1689m': (3, 5, 7, 11),
    'vit_large_patch16_dinov3.lvd1689m': (5, 11, 17, 23),
    'vit_huge_plus_patch16_dinov3.lvd1689m': (7, 15, 23, 31),
    'vit_7b_patch16_dinov3.lvd1689m': (9, 19, 29, 39),
}


def dino_v3_s():
    model_name = 'vit_small_patch16_dinov3.lvd1689m'
    out_indices = vit_model_to_out_indices[model_name]
    backbone = timm.create_model(
        model_name=model_name,
        features_only=True,
        dynamic_img_size=True,
        out_indices=out_indices,
        # pretrained=True,
        # cache_dir='./tmp-timm_cache/DINOv3',
    )
    return backbone

def dino_v3_s_plus():
    model_name = 'vit_small_plus_patch16_dinov3.lvd1689m'
    out_indices = vit_model_to_out_indices[model_name]
    backbone = timm.create_model(
        model_name=model_name,
        features_only=True,
        dynamic_img_size=True,
        out_indices=out_indices,
    )
    return backbone

def dino_v3_b():
    model_name = 'vit_base_patch16_dinov3.lvd1689m'
    out_indices = vit_model_to_out_indices[model_name]
    backbone = timm.create_model(
        model_name=model_name,
        features_only=True,
        dynamic_img_size=True,
        out_indices=out_indices,
    )
    return backbone

def dino_v3_l():
    model_name = 'vit_large_patch16_dinov3.lvd1689m'
    out_indices = vit_model_to_out_indices[model_name]
    backbone = timm.create_model(
        model_name=model_name,
        features_only=True,
        dynamic_img_size=True,
        out_indices=out_indices,
    )
    return backbone

def dino_v3_h_plus():
    model_name = 'vit_huge_plus_patch16_dinov3.lvd1689m'
    out_indices = vit_model_to_out_indices[model_name]
    backbone = timm.create_model(
        model_name=model_name,
        features_only=True,
        dynamic_img_size=True,
        out_indices=out_indices,
    )
    return backbone

def dino_v3_7b():
    model_name = 'vit_7b_patch16_dinov3.lvd1689m'
    out_indices = vit_model_to_out_indices[model_name]
    backbone = timm.create_model(
        model_name=model_name,
        features_only=True,
        dynamic_img_size=True,
        out_indices=out_indices,
    )
    return backbone
