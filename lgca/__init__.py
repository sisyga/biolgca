def get_lgca(geometry='hex', ib=False, ve=True, **kwargs):
    if not ve:
        if geometry in ['1d', 'lin', 'linear']:
            from lgca.lgca_1d import NoVE_LGCA_1D
            return NoVE_LGCA_1D(**kwargs, ve=ve)
        elif geometry in ['square', 'sq', 'rect', 'rectangular']:
            from lgca.lgca_square import NoVE_LGCA_Square
            return NoVE_LGCA_Square (**kwargs, ve=ve)
        elif geometry in ['hex']:
            from lgca.lgca_hex import NoVE_LGCA_Hex
            return NoVE_LGCA_Hex (**kwargs)


    if ib:
        if geometry in ['1d', 'lin', 'linear']:
            from lgca.lgca_1d import IBLGCA_1D
            return IBLGCA_1D(**kwargs)
        elif geometry in ['square', 'sq', 'rect', 'rectangular']:
            from lgca.lgca_square import IBLGCA_Square
            return IBLGCA_Square(**kwargs)

        else:
            from lgca.lgca_hex import IBLGCA_Hex
            return IBLGCA_Hex(**kwargs)

    else:
        if geometry in ['1d', 'lin', 'linear']:
            from lgca.lgca_1d import LGCA_1D
            return LGCA_1D(**kwargs)

        elif geometry in ['square', 'sq', 'rect', 'rectangular']:
            from lgca.lgca_square import LGCA_Square
            return LGCA_Square(**kwargs)

        else:
            from lgca.lgca_hex import LGCA_Hex
            return LGCA_Hex(**kwargs)
