def get_lgca(geometry='hex', ib=False, **kwargs):
    if ib:
        if geometry in ['1d', 'lin', 'linear']:
            from .lgca_1d import IBLGCA_1D
            return IBLGCA_1D(**kwargs)
        else:
            print('Not implemented yet!')

    else:
        if geometry in ['1d', 'lin', 'linear']:
            from .lgca_1d import LGCA_1D
            return LGCA_1D(**kwargs)

        elif geometry in ['square', 'sq', 'rect', 'rectangular']:
            from .lgca_square import LGCA_Square
            return LGCA_Square(**kwargs)

        else:
            from .lgca_hex import LGCA_Hex
            return LGCA_Hex(**kwargs)
