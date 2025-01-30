from lgca import get_lgca


lgca_sq = get_lgca(geometry='square', dims=4)
print(lgca_sq.c)

lgca_cb = get_lgca(geometry='cubic', dims=4)
print(lgca_cb.c)

