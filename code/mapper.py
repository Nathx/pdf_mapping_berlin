import gmplot
from location import Satellite, River, BGate
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import numpy as np

SW_LAT, SW_LON = 52.464011, 13.274099
NE_LAT, NE_LON = 52.586925, 13.521837

class Mapper(object):
    def __init__(self, objs, n=256):
        self.objects = objs
        self.latitudes, self.longitudes = self.generate_mesh_grid(n)

    def generate_mesh_grid(self, n):
        """
        Returns X and Y 1-D arrays of latitudes and longitudes
        gridding Berlin's map.
        """
        x = np.linspace(SW_LAT, NE_LAT, n)
        y = np.linspace(SW_LON, NE_LON, n)
        X, Y = np.meshgrid(x, y)
        return X.flatten(), Y.flatten()

    def pull_heatmap_idx(self, distribution, size=10000):
        """
        Sample points on map from a distribution to generate heatmap.
        """
        return np.random.choice(np.arange(distribution.size), size=size, p=distribution)

    def get_distribution(self, objs):
        """
        Get the distribution on the map from different objects.

        Distributions get combined through bayesian update:
        Starting from a uniform distribution, the new distributions from
        independent sources come as new likelihoods multiplying our prior
        to obtain the posterior. Normalization is done once, as a final step.
        """
        distribution = np.ones(self.latitudes.shape)
        distribution /= np.sum(distribution)
        for obj in objs:
            probs = obj.get_pdf(self.latitudes, self.longitudes)
            distribution *= probs
        distribution /= np.sum(distribution)
        return distribution

    def find_maximum(self, distribution):
        """
        Returns the coordinates of the maximum of a given distribution.
        """
        max_idx = np.argmax(distribution)
        return self.latitudes[max_idx], self.longitudes[max_idx]

    def generate_map(self, objs=None, plot_type='lines', max_marker=False,
            heatmap_size=20000, threshold=10):
        """
        Use gmplot module to generate map overlay of the given distributions.

        Input
        ----
        objs: Location objects with PDF as defined in location.py
        plot_type: type of overlay ('lines' or 'heatmap')
        max_marker: add a marker where the next top analyst is more likely to be
        size: Number of draws from the PDF to plot the heatmap.
        Threshold: Min. # of values to color an area in red on the heatmap.

        Returns
        ----
        HTML file containing the Map object
        """
        if not objs:
            objs = self.objects

        distribution = self.get_distribution(objs)
        gmap = gmplot.GoogleMapPlotter((SW_LAT + NE_LAT)/2, (SW_LON + NE_LON)/2, 11)

        # mark maximum likelihood
        if max_marker:
            x, y = self.find_maximum(distribution)
            gmap.scatter([x], [y],
                        c='r', marker=True)

        if plot_type=='lines':
            # add probability lines to the map
            X, Y = self.latitudes.reshape((256, 256)), self.longitudes.reshape((256, 256))
            probs = distribution.reshape((256, 256))
            C = plt.contour(X, Y, probs, 5, colors='black', linewidth=.1)
            color_map = self.get_color_map(C.levels)
            for i, level in enumerate(C.collections):
                for path in level.get_paths():
                    gmap.plot(*zip(*path.vertices), color=color_map[i], edge_width=3)

        elif plot_type=='heatmap':
            # generate heatmap
            heatmap_idx = self.pull_heatmap_idx(distribution, size=heatmap_size)
            gmap.heatmap(self.latitudes[heatmap_idx], self.longitudes[heatmap_idx],
                        radius=5, opacity=.4, threshold=threshold)

        # delimiting region of interest
        gmap.polygon([SW_LAT, SW_LAT, NE_LAT, NE_LAT], [SW_LON, NE_LON, NE_LON, SW_LON], face_alpha=0.01)

        fn = '../maps/{}.html'.format(objs[0].name if len(objs) == 1 else 'final_map')
        gmap.draw(fn)

    def get_color_map(self, levels):
        """Returns gradient of color from green to red.
        """
        sm = ScalarMappable(cmap='RdYlGn_r')
        normed_levels = levels / np.max(levels)
        colors = 255 * sm.to_rgba(normed_levels)[:, :3]
        return ['#%02x%02x%02x' % (r, g, b) for r,g,b in colors]
