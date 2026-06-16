def get_network(args):

    if 'pointmixer' == args.arch:
        from .pointmixer import getPointMixerSegNet as getNetwork
    elif 'pointmixer_panoptic' == args.arch:
        from .pointmixer import getPointMixerPanopticNet as getNetwork
    elif 'pointtransformer' == args.arch:
        from .pointtransformer import getPointTransformerSegNet as getNetwork
        # elif 'pointnet' == args.arch:
        #     from .pointnet   import getPointMixerSegNet as getNetwork
    else:
        raise NotImplementedError
    
    kwargs = \
        {
            'intraLayer': args.intraLayer,
            'interLayer': args.interLayer,
            'transup': args.transup,
            'transdown': args.transdown,
            'stride': args.downsample,
        }
    if args.arch in ('pointmixer', 'pointmixer_panoptic'):
        kwargs.update({
            'planes': getattr(args, 'pointmixer_planes', None),
            'width_multiplier': getattr(args, 'pointmixer_width_multiplier', 1.0),
            'share_planes': getattr(args, 'pointmixer_share_planes', 8),
        })
    model = getNetwork(c=args.fea_dim, k=args.classes, nsample=args.nsample, **kwargs)
    
    return model
