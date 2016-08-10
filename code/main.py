# coding: utf-8
from location import Satellite, River, BGate
from mapper import Mapper
import sys


if __name__ == '__main__':

    print "Generating objects.."

    bg_coords = [(52.516288, 13.377689)]
    sat_coords = [(52.590117,13.39915), (52.437385,13.553989)]

    with open('../river_spree.txt', 'r') as f:
        line = f.readline()
        river_coords = []
        while line:
            lat, lon = [float(el) for el in line.strip('\n').split(',')]
            river_coords.append((lat, lon))
            line = f.readline()

    objects = [BGate(bg_coords, mean=4.7, mode=3.877),
                Satellite(sat_coords, 2.4),
                River(river_coords, 2.73)]

    sys.stdout.flush()

    mapper = Mapper(objects)
    for obj in objects:
        print "Creating map for %s.." % obj.name
        mapper.generate_map([obj], plot_type='lines')

    print "Creating final map.."

    mapper.generate_map(plot_type='lines', max_marker=True)
