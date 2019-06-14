import numpy as np


def errors(lgca):
    print('---errors?---')
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


def count_fam(lgca):
    print('---genealogical research---')
    num = lgca.props['num_off']
    print('num', num)   #TODO: num stimmt manchmal nicht?
    print('genealogical tree:', num[1:])
    print('number of ancestors:', lgca.maxlabel_init)
    print('number of offsprings:', sum(num[1:]))
    print('max family number is %d with ancestor cell %d' % (
    max(num[1:]), num.index(max(num[1:]))))

    return max(num[1:])

def aloha(who):
    print('aloha', who)