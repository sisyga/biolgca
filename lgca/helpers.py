import numpy as np

def errors(lgca):
    inh_l = False
    for i in range(lgca.maxlabel.astype(int) + 1):
        if lgca.props['lab_m'][i] <= lgca.maxlabel_init:
            inh_l = True
        else:
            inh_l = False
    if inh_l:
        print('inheritance label passen')
    else:
        print('Fehler: inheritance label passen nicht!')

    if len(lgca.props['lab_m']) == len(lgca.props['r_b']) and len(lgca.props['r_b']) == lgca.maxlabel + 1:
        print('len(props) passt')
    else:
        print('Fehler: len(props) passen nicht!')

def aloha(who):
    print('aloha', who)