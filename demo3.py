# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:06:07 2019

@author: Nr.12
"""

from lgca import get_lgca
from lgca.interactions import alignment

lgca = get_lgca(geometry='lin')
lgca2 = get_lgca(geometry='lin', ve=False)
alignment(lgca)
alignment(lgca2)