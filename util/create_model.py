from models.vit_model import vit_base_patch16_224_front,vit_base_patch16_224_back,vit_base_patch16_224_mid 
 


def create_model(model_name, part, img_size, patch_size, aux_depth_list, num_classes, device):
    if model_name == 'vit_224_p16':
        if part == 'front':
            return vit_base_patch16_224_front(img_size=img_size,patch_size=patch_size,aux_depth_list=aux_depth_list, num_classes=num_classes).to(device)
        elif part == 'mid':
            return vit_base_patch16_224_mid(aux_depth_list=aux_depth_list,num_classes=num_classes).to(device)
        elif part == 'back':
            return vit_base_patch16_224_back(aux_depth_list=aux_depth_list,num_classes=num_classes).to(device)