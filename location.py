import numpy as np
from geopy.distance import vincenty
import scipy.stats as scs
from itertools import product

SW_LAT, SW_LON = 52.464011, 13.274099
NE_LAT, NE_LON = 52.586925, 13.521837

class Location(object):
    """
    This abstract class contains common attributes and methods
    of our 3 spatial probability distributions.
    """
    def __init__(self, name, coords):
        self.coords = coords
        self.name = name
        self.distribution = None

    def distance(self, point):
        """Returns distance from instance to point of interest."""
        pass

    def prob(self, point):
        """
        Returns probability of finding the candidate at this point.
        """
        path_dist = self.distance(point)
        return self.distribution.pdf(path_dist)

    def get_pdf(self, lats, lons):
        if not hasattr(self, 'pdf'):
            self.pdf = [self.prob((x, y)) for x, y in zip(lats, lons)]
        return self.pdf



class Satellite(Location):
    """
    This class represents the distribution around the great circle path
    of the satellite.
    """
    def __init__(self, coords, ci_range):
        super(Satellite, self).__init__(coords=coords, name='satellite')
        self.distribution = scs.norm(0, ci_range / 1.96)
        self.R = 6371

    def distance(self, point):
        """
        Computes the cross-track distance:
        min. distance between a point and a great circle path.
        """
        sat_start, sat_end = self.coords
        delta_13 = vincenty(sat_start, point).km / self.R
        theta_13 = self.bearing(sat_start, point)
        theta_12 = self.bearing(sat_start, sat_end)
        return np.arcsin(np.sin(delta_13) * np.sin(np.radians(theta_13 - theta_12))) * self.R

    def bearing(self, start, end):
        """Computes the bearing of a great circle path between two points."""
        start_lat, start_lon = np.radians(start)
        end_lat, end_lon = np.radians(end)
        delta_lon = np.radians(end[1] - start[1])

        br = np.arctan2(np.sin(delta_lon) * np.cos(end_lat),
                  np.cos(start_lat) * np.sin(end_lat) - np.sin(start_lat) * np.cos(end_lat) * np.cos(delta_lon))
        br = np.degrees(br) % 360
        return br


class River(Location):
    """
    This subclass represents the normal distribution around the river Spree.
    """
    def __init__(self, coords, ci_range):
        super(River, self).__init__(coords=coords, name='river')
        self.lines = self.make_linear(coords)
        self.distribution = scs.norm(0, ci_range / 1.96)

    def make_linear(self, coords):
        """Transforms list of coordinates into list of segments
        in an orthogonal plan.
        """
        xy_coords = [self.convert_xy(*lon_lat) for lon_lat in self.coords]
        return zip(xy_coords[:-1], xy_coords[1:])

    def distance(self, point):
        """
        Computes the smallest distance to one of the river's segments.
        """
        xy_point = self.convert_xy(*point)
        return min(self.line_distance(xy_point, line) for line in self.lines)

    def line_distance(self, point, line):
        """Distance from a given point to a segment of the river."""
        start, end = line
        line_x = end[0] - start[0]
        line_y = end[1] - start[1]

        line_length = np.sqrt(line_x**2 + line_y**2)

        u =  ((point[0] - start[0]) * line_x + (point[1] - start[1]) * line_y) / line_length**2

        u = np.clip(u, 0, 1)

        x = start[0] + u * line_x
        y = start[1] + u * line_y

        dx = x - point[0]
        dy = y - point[1]

        dist = np.sqrt(dx*dx + dy*dy)

        return dist

    def convert_xy(self, lon, lat):
        """Converts lon-lat into xy-coordinates."""
        x = (lon - SW_LON) * np.cos(SW_LAT*np.pi/180) * 111.323
        y = (lat - SW_LAT) * 111.323
        return x, y


class BGate(Location):
    """
    This subclass represents the log-normal distribution centered
    around the Branderburger Gate.
    """
    def __init__(self, coords, mean, mode):
        super(BGate, self).__init__(coords=coords, name='brandenburger_gate')
        self.distribution = self.set_distribution(mean, mode)

    def set_distribution(self, mean, mode):
        """
        Create lognormal distribution from given mean and mode.
        Distances are converted to km to prevent overflow.
        """
        scale = np.exp(mean)
        s = np.sqrt(np.log(scale / float(mode)))
        return scs.lognorm(s=s, scale=scale)

    def distance(self, point):
        """
        Distances are converted to km to prevent overflow.
        """
        return vincenty(point, self.coords[0]).km
