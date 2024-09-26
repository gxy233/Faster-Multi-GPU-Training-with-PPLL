from models.vit_model import vit_base_patch16_224_front,vit_base_patch16_224_back,vit_base_patch16_224_mid 
# from models.resnetInfoPro import resnet32_front, resnet32_back
from models.resnetInfoPro_t import resnet32_front, resnet32_back


def create_model(model_name, part,args,comargs):
    if model_name.startswith('vit_224'):
        img_size=comargs.img_size
        patch_size=comargs.patch_size
        aux_depth_list=args.aux_depth_list
        num_classes=comargs.num_classes
        device=args.device
        
        if part == 'front':
            
            return vit_base_patch16_224_front(img_size=img_size,patch_size=patch_size,aux_depth_list=aux_depth_list, num_classes=num_classes).to(device)
        elif part == 'mid':
            return vit_base_patch16_224_mid(aux_depth_list=aux_depth_list,num_classes=num_classes).to(device)
        elif part == 'back':
            return vit_base_patch16_224_back(aux_depth_list=aux_depth_list,num_classes=num_classes).to(device)
        
    # if model_name == 'k2_resnet32':
    
        
    #     if part == 'front':
    #         layers_f=args.layers
    #         infopro_config_f=args.infopro_config
    #         local_module_num=comargs.local_module_num
    #         inplanes=args.inplanes
    #         batch_size=comargs.batch_size
    #         image_size=comargs.image_size
    #         dataset=comargs.dataset
    #         class_num=comargs.num_classes
    #         wide_list_f=args.wide_list
    #         device=args.device
            
    #         front= resnet32_front(layers=layers_f ,infopro_config=infopro_config_f, local_module_num=local_module_num, inplanes=inplanes, 
    #                                 batch_size=batch_size, image_size=image_size,
    #                                 dataset=dataset, class_num=class_num,
    #                                 wide_list=wide_list_f, dropout_rate=0,
    #                                 aux_net_config='1c2f', local_loss_mode='contrast',
    #                                 aux_net_widen=1, aux_net_feature_dim=128, device=device).to(device)
            
    #         return front
        
    #     elif part == 'mid':
    #         return 
        
    #     elif part == 'back':
    #         layers_b=args.layers
    #         infopro_config_b=args.infopro_config
    #         local_module_num=comargs.local_module_num
    #         inplanes=args.inplanes
    #         batch_size=comargs.batch_size
    #         image_size=comargs.image_size
    #         dataset=comargs.dataset
    #         class_num=comargs.num_classes
    #         wide_list_b=args.wide_list
    #         device=args.device
            
    #         back= resnet32_back(layers=layers_b ,infopro_config=infopro_config_b, local_module_num=local_module_num, inplanes=inplanes, 
    #                                 batch_size=batch_size, image_size=image_size,
    #                                 dataset=dataset, class_num=class_num,
    #                                 wide_list=wide_list_b, dropout_rate=0,
    #                                 aux_net_config='1c2f', local_loss_mode='contrast',
    #                                 aux_net_widen=1, aux_net_feature_dim=128, device=device).to(device)
        
    #     return back
    
    
    
    
    if model_name == 'k2_resnet32_t':
    
        
        if part == 'front':
        
        
            layers_f=args.layers
            infopro_config_f=args.infopro_config
            local_module_num=comargs.local_module_num
            inplanes=args.inplanes
            batch_size=comargs.batch_size
            image_size=comargs.image_size
            dataset=comargs.dataset
            class_num=comargs.class_num
            device=args.device
            aux_net_feature_dim=comargs.aux_net_feature_dim
            momentum=comargs.momentum
            aux_net_config=comargs.aux_net_config
            front= resnet32_front(layers=layers_f ,infopro_config=infopro_config_f, local_module_num=local_module_num, inplanes=inplanes, 
                                    batch_size=batch_size, image_size=image_size,
                                    dataset=dataset, class_num=class_num,
                                    dropout_rate=0, 
                                    aux_net_config=aux_net_config, local_loss_mode='contrast',
                                    aux_net_widen=1, aux_net_feature_dim=aux_net_feature_dim, momentum=momentum, device=device).to(device)
            
            return front
        
        elif part == 'mid':
            return 
        
        elif part == 'back':
            layers_b=args.layers
            infopro_config_b=args.infopro_config
            local_module_num=comargs.local_module_num
            inplanes=args.inplanes
            batch_size=comargs.batch_size
            image_size=comargs.image_size
            dataset=comargs.dataset
            class_num=comargs.class_num
            device=args.device
            aux_net_feature_dim=comargs.aux_net_feature_dim
            momentum=comargs.momentum
            aux_net_config=comargs.aux_net_config
            
            back= resnet32_back(layers=layers_b ,infopro_config=infopro_config_b, local_module_num=local_module_num, inplanes=inplanes, 
                                    batch_size=batch_size, image_size=image_size,
                                    dataset=dataset, class_num=class_num,
                                    dropout_rate=0,
                                    aux_net_config=aux_net_config, local_loss_mode='contrast',
                                    aux_net_widen=1, aux_net_feature_dim=aux_net_feature_dim, momentum=momentum, device=device).to(device)
        
        return back