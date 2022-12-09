def parse_args(args):
    params = {}
    args_keys = args.keys()

    if 'minkowski' in args_keys:
        params['minkowski'] = float(args['minkowski'])
    if 'weightedModel' in args_keys:
        params['weighted_model'] = args['weightedModel']
    if 'quadraticFormModel' in args_keys:
        params['quadratic_form_model'] = args['quadraticFormModel']

    return params
