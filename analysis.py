import numpy as np
import itertools
from scipy.fft import *

midpoint_50 = (25.0, 21.650635094610966)
midpoint_100 = (50.0, 40.703193977868615)
midpoint_20 = (10.0, 9.526279441628825)
midpoint_30 = (15, 12.90381056766578)
mid_point_small = (2.5, 3.5)
midpoint_10 = (5, 4.3)

test_point_1 = (20, 13)
test_point_2 = (10, 13)

radius_100 = 48.5
radius_50 = 26
radius_30 = 9.5
radius_10 = 0.1

def initial_values(lgca, ecm):
    ecm.tensor_update(t=0)
    initial_coord_list = []
    for x in np.arange(1, lgca.lx + 1):
        for y in np.arange(1, lgca.ly + 1):
            array = ind_coord(lgca.coord_pairs, lgca.coord_pairs_hex, [(x, y)])
            array_ = sphere(coord=array)
            if array_ != None:
                coordinate = ind_coord(lgca.coord_pairs_hex, lgca.coord_pairs, [array_])
                lgca.nodes[coordinate[0], coordinate[1], :np.random.randint(5,6)] = 1
                ecm.scalar_field[coordinate] = 0.01
                initial_coord_list.append(coordinate)
    return initial_coord_list

def initial_values_hom(lgca, ecm):
    ecm.tensor_update(t=0)
    initial_coord_list = []
    for x in np.arange(1, lgca.lx + 1):
        for y in np.arange(1, lgca.ly + 1):
            lgca.nodes[x, y, :np.random.randint(1, 2)] = 1
            # ecm.scalar_field[coordinate] = np.random.uniform(0, 0)
            initial_coord_list.append((x,y))
    return initial_coord_list

def sphere(coord, radius =30, midpoint =midpoint_30):
    if (coord[0]-midpoint[0])**2 + (coord[1]-midpoint[1])**2 < radius:
        return coord

def grid_points(lgca, ecm):
    ecm.tensor_update(t=0)
    x_points = [3, 9, 15, 21, 27]
    x_points = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    y_points = x_points
    for x in x_points:
        for y in y_points:
            lgca.nodes[x, y, :np.random.randint(1,2)] = 1

def random_points(lgca , ecm, number_points=20):
    ecm.tensor_update(t=0)
    for _ in itertools.repeat(None, number_points ):
        lgca.nodes[np.random.randint(0, lgca.lx), np.random.randint(0, lgca.ly), : np.random.randint(1, 2)] = 1

def ind_coord(ind_array, coord_array, ind):
    indexlist = []
    for j in ind:
        for counter, i in enumerate(ind_array):
           if i == j:
               index = counter
               indexlist.append(index)
    return coord_array[indexlist[0]]

def hexgrid_lines(lgca, dir="E", length=20):
    midpoint = (25, 21.6)
    x_set= set(lgca.xcoords.ravel())
    y_set = set(lgca.ycoords.ravel())
    c_list = []
    if dir == "E":
        xy = [1, 0]
    if dir == "W":
        xy = [-1, 0]
    if dir == "NE":
        xy = [0.5, np.sqrt(3)/2]
    if dir == "NW":
        xy = [-0.5, np.sqrt(3) /2]
    if dir == "SE":
        xy = [0.5, -np.sqrt(3)/2]
    if dir == "SW":
        xy = [-0.5, -np.sqrt(3)/2]


    x_ = np.zeros(length)
    y_ = np.zeros(length)
    x_[0] = midpoint[0]
    y_[0] = midpoint[1]


    for i in np.arange(1,15):
        x_[i] =  x_[i-1] + xy[0]
        y_[i] = y_[i-1] + xy[1]

    for i in np.arange(len(x_)):
        x = min(x_set, key=lambda x:abs(x-x_[i]))
        y = min(y_set, key=lambda x:abs(x-y_[i]))
        x_set.remove(x)
        y_set.remove(y)
        x2 = min(x_set, key=lambda x:abs(x-x_[i]))
        y2 = min(y_set, key=lambda x:abs(x-y_[i]))

        if abs(x-x_[i]) + abs(y2 - y_[i]) > abs(x2-x_[i]) + abs(y - y_[i]):
            list = [(x,y), (x2, y), (x, y2), (x2, y2)]
        else :
            list = [(x,y), (x, y2), (x2, y), (x2, y2)]
        coordinate = ind_coord(lgca.coord_pairs_hex, lgca.coord_pairs, list)
        x_set = set(lgca.xcoords.ravel())
        y_set = set(lgca.ycoords.ravel())
        c_list.append(coordinate)

    return c_list
