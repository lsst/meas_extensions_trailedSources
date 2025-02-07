#
# This file is part of meas_extensions_trailedSources.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import unittest
import lsst.utils.tests
import lsst.meas.extensions.trailedSources
from lsst.meas.base.tests import AlgorithmTestCase
from lsst.utils.tests import classParameters
from lsst.geom import Point2I, Point2D, Box2I, Extent2I
from unittest.mock import patch

# Trailed-source length, angle, and centroid coordinates.
trail_lengths = np.array([5, 5, 10, 4])
trail_angles = np.array([100, 0, 5, 4])
trail_x_coords = np.array([100, 20, -20, 90])
trail_y_coords = np.array([100, 20, -30, 100])


class TrailedEdgeSource:
    """Holds a set of true trail parameters.
    """

    def __init__(self, instFlux, length, angle, xc, yc):
        self.instFlux = instFlux
        self.length = length
        self.angle = angle
        self.center = Point2D(xc, yc)
        self.x0 = xc - length / 2 * np.cos(angle)
        self.y0 = yc - length / 2 * np.sin(angle)
        self.x1 = xc + length / 2 * np.cos(angle)
        self.y1 = yc + length / 2 * np.sin(angle)


class TrailedTaskSetup:

    def makeTrailedSourceMeasurementTask(self, plugin=None, dependencies=(),
                                         config=None, schema=None,
                                         algMetadata=None):
        """Set up a measurement task for a trailed source plugin.
        """
        config = self.makeSingleFrameMeasurementConfig(plugin=plugin,
                                                       dependencies=dependencies)

        # Make sure the shape slot is base_SdssShape
        config.slots.shape = "base_SdssShape"
        return self.makeSingleFrameMeasurementTask(plugin=plugin,
                                                   dependencies=dependencies,
                                                   config=config,
                                                   schema=schema,
                                                   algMetadata=algMetadata)


# "Extend" meas.base.tests.TestDataset
class TrailedTestDataset(lsst.meas.base.tests.TestDataset):
    """A dataset for testing trailed source measurements.
    Given a `TrailedSource`, construct a record of the true values and an
    Exposure.
    """

    def __init__(self, bbox, threshold=10.0, exposure=None, **kwds):

        super().__init__(bbox, threshold, exposure, **kwds)

    def addTrailedSource(self, trail, edge=True):
        """Add a trailed source to the simulation.

        Re-implemented version of
        `lsst.meas.base.tests.TestDataset.addSource`. Numerically integrates a
        Gaussian PSF over a line to obtain an image of a trailed source and
        adds edge flags to the image.
        """
        record = self.catalog.addNew()
        record.set(self.keys["centroid"], trail.center)
        rng = np.random.default_rng(32)
        covariance = rng.normal(0, 0.1, 4).reshape(2, 2)
        covariance[0, 1] = covariance[1, 0]
        record.set(self.keys["centroid_sigma"], covariance.astype(np.float32))
        record.set(self.keys["shape"], self.psfShape)
        record.set(self.keys["isStar"], False)

        # Sum the psf at each
        numIter = int(2 * trail.length)
        xp = np.linspace(trail.x0, trail.x1, num=numIter)
        yp = np.linspace(trail.y0, trail.y1, num=numIter)
        for (x, y) in zip(xp, yp):
            pt = Point2D(x, y)
            im = self.drawGaussian(self.exposure.getBBox(), trail.instFlux,
                                   lsst.afw.geom.Ellipse(self.psfShape, pt))
            self.exposure.getMaskedImage().getImage().getArray()[:, :] += im.getArray()

        planes = self.exposure.mask.getMaskPlaneDict()
        dim = self.exposure.getBBox().getDimensions()

        # Add edge flags to the first and last 20 columns and rows.
        if edge:
            for y in range(20):
                self.exposure.mask.setMaskPlaneValues(planes['EDGE'], 0, dim[0] - 1, y)
                self.exposure.mask.setMaskPlaneValues(planes['EDGE'], 0, dim[0] - 1, y + dim[1] - 20)

            for y in range(dim[1]):
                self.exposure.mask.setMaskPlaneValues(planes['EDGE'], 0, 20, y)
                self.exposure.mask.setMaskPlaneValues(planes['EDGE'], dim[0] - 20, dim[0] - 1, y)

        totFlux = self.exposure.image.array.sum()
        self.exposure.image.array /= totFlux
        self.exposure.image.array *= trail.instFlux

        record.set(self.keys["instFlux"], trail.instFlux)
        self._installFootprint(record, self.exposure.getImage())

        return record, self.exposure.getImage()


# Following from test_trailedSources
@classParameters(length=trail_lengths, theta=trail_angles, xc=trail_x_coords, yc=trail_y_coords)
class TrailedEdgeSourcesTestCase(AlgorithmTestCase, lsst.utils.tests.TestCase):
    """ Test if ext_trailedSources_Naive_flag_edge is set correctly.

    Given a `TrailedSource`, test if the edge flag is set correctly in the
    source catalog after the
    `lsst.meas.extensions.trailedSources.Naive.Plugin.makeTrailedSourceMeasurementTask`
    has been run on the source catalog.
    """

    def setUp(self):
        self.center = Point2D(50.1, 49.8)
        self.bbox = Box2I(lsst.geom.Point2I(-20, -30), Extent2I(140, 160))
        self.dataset = TrailedTestDataset(self.bbox)

        # Trail which extends into edge pixels
        self.trail = TrailedEdgeSource(100000.0, self.length, self.theta, self.xc, self.yc)
        self.dataset.addTrailedSource(self.trail)

    def testEdgeFlag(self):
        """Test if edge flags are correctly set in NaivePlugin.py

        Given a `TrailedTestDataset`, run the NaivePlugin measurement and
        check that the trailed sources have the edge flag set. [100,100] does
        not contain any edge pixels and should not have a flag set, [20,20]
        crosses into the edge region on only one side and should have the edge
        flag set, and [-20,-30] extends off the chip and should have the edge
        flag set.
        """
        # Set up and run Naive measurement.
        task = TrailedTaskSetup.makeTrailedSourceMeasurementTask(self,
                                                                 plugin="ext_trailedSources_Naive",
                                                                 dependencies=("base_SdssCentroid",
                                                                               "base_SdssShape")
                                                                 )
        exposure, catalog = self.dataset.realize(5.0, task.schema, randomSeed=0)
        task.run(catalog, exposure)
        record = catalog[0]

        # Check that x0, y0 or x1, y1 is flagged as an edge pixel
        x1 = int(record['ext_trailedSources_Naive_x1'])
        y1 = int(record['ext_trailedSources_Naive_y1'])
        x0 = int(record['ext_trailedSources_Naive_x0'])
        y0 = int(record['ext_trailedSources_Naive_y0'])

        # Test Case with no edge pixels
        if record['truth_x'] == 100:
            # These are used to ensure the mask pixels the trailed sources are
            # compared with have the correct flags set
            begin_edge_pixel_set = (exposure.mask[Point2I(x0, y0)] & exposure.mask.getPlaneBitMask(
                'EDGE') != 0)
            end_edge_pixel_set = (exposure.mask[Point2I(x1, y1)] & exposure.mask.getPlaneBitMask(
                'EDGE') != 0)

            self.assertFalse(begin_edge_pixel_set)
            self.assertTrue(end_edge_pixel_set)

            # Make sure measurement edge flag is set, but Naive_flag is not.
            # A failed trailed source measurement with the edge flag
            # set means the edge flag was set despite the measurement
            # failing.
            self.assertTrue(record.get("ext_trailedSources_Naive_flag_edge"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag_off_image"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag_nan"))

            x1 = int(record['ext_trailedSources_Naive_x1'])
            y1 = int(record['ext_trailedSources_Naive_y1'])

            self.assertFalse(exposure.mask[Point2I(x0, y0)] & exposure.mask.getPlaneBitMask('EDGE') != 0)
            self.assertTrue(exposure.mask[Point2I(x1, y1)] & exposure.mask.getPlaneBitMask('EDGE') != 0)

        # Test case with one end of trail containing edge pixels
        elif record['truth_x'] == 20:
            begin_edge_pixel_set = (exposure.mask[Point2I(x0, y0)] & exposure.mask.getPlaneBitMask(
                'EDGE') != 0)
            end_edge_pixel_set = (exposure.mask[Point2I(x1, y1)] & exposure.mask.getPlaneBitMask(
                'EDGE') != 0)

            self.assertFalse(begin_edge_pixel_set)
            self.assertFalse(end_edge_pixel_set)

            # Make sure measurement Naive_flag_edge and Naive_flag not set
            self.assertFalse(record.get("ext_trailedSources_Naive_flag_edge"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag_off_image"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag_nan"))

            x1 = int(record['ext_trailedSources_Naive_x1'])
            y1 = int(record['ext_trailedSources_Naive_y1'])

            self.assertFalse(exposure.mask[Point2I(x0, y0)] & exposure.mask.getPlaneBitMask('EDGE') != 0)
            self.assertFalse(exposure.mask[Point2I(x1, y1)] & exposure.mask.getPlaneBitMask('EDGE') != 0)

        # Test case trail fully contained
        elif record["truth_x"] == 90:
            begin_edge_pixel_set = (exposure.mask[Point2I(x0, y0)] & exposure.mask.getPlaneBitMask(
                'EDGE') != 0)
            end_edge_pixel_set = (exposure.mask[Point2I(x1, y1)] & exposure.mask.getPlaneBitMask(
                'EDGE') != 0)

            self.assertFalse(begin_edge_pixel_set)
            self.assertFalse(end_edge_pixel_set)

            # Make sure measurement Naive_flag_edge and Naive_flag not set
            self.assertFalse(record.get("ext_trailedSources_Naive_flag_edge"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag_off_image"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag_nan"))

            x1 = int(record['ext_trailedSources_Naive_x1'])
            y1 = int(record['ext_trailedSources_Naive_y1'])

            self.assertFalse(exposure.mask[Point2I(x0, y0)] & exposure.mask.getPlaneBitMask('EDGE') != 0)
            self.assertFalse(exposure.mask[Point2I(x1, y1)] & exposure.mask.getPlaneBitMask('EDGE') != 0)

        # Test case with trailed source extending off chip.
        else:
            self.assertEqual(record['truth_x'], -20)
            self.assertTrue(record.get("ext_trailedSources_Naive_flag_edge"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag"))
            self.assertTrue(record.get("ext_trailedSources_Naive_flag_off_image"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag_nan"))

    def testNanFlag(self):
        """Test if nan flags are correctly set in NaivePlugin.py

        Given a `TrailedTestDataset`, run the NaivePlugin measurement which
        has trailed sources where one of the end point values results in a
        nan.
        """
        # Set up and run Naive measurement.
        task = TrailedTaskSetup.makeTrailedSourceMeasurementTask(self,
                                                                 plugin="ext_trailedSources_Naive",
                                                                 dependencies=("base_SdssCentroid",
                                                                               "base_SdssShape")
                                                                 )

        exposure, catalog = self.dataset.realize(5.0, task.schema, randomSeed=0)

        original_check_trail_function = task.plugins['ext_trailedSources_Naive'].check_trail
        # Used to simulate a trailed source where one of the coordinates is a
        # nan.

        def check_trail_mock(*args, **kwargs):
            measRecord = args[0]
            exposure = args[1]
            x0 = args[2]
            y0 = args[3]
            x1 = args[4]
            y1 = np.nan  # overriding to test NAN flagging
            length = args[6]
            measRecord['ext_trailedSources_Naive_y1'] = np.nan
            return original_check_trail_function(measRecord, exposure, x0, y0, x1, y1, length)

        # This patcher mocks check_trail so that one of the trailed sources
        # includes it is checking contains a nan at one of its endpoints.
        patcher = patch(
            'lsst.meas.extensions.trailedSources.NaivePlugin.SingleFrameNaiveTrailPlugin.check_trail',
            side_effect=check_trail_mock)
        patcher.start()
        task.run(catalog, exposure)
        record = catalog[0]

        # Test Case with no edge pixels, but one is set to nan.
        if record['truth_x'] == 100:
            self.assertFalse(record.get("ext_trailedSources_Naive_flag_edge"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag_off_image"))
            self.assertTrue(record.get("ext_trailedSources_Naive_flag_nan"))

        # Test case with one end of trail containing edge pixels, but nan is
        # set so edge does not end up set.
        elif record['truth_x'] == 20:

            self.assertFalse(record.get("ext_trailedSources_Naive_flag_edge"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag_off_image"))
            self.assertTrue(record.get("ext_trailedSources_Naive_flag_nan"))

        # Test case trail fully contained, but contains one nan. Only nan flag
        # is set.
        elif record["truth_x"] == 90:

            self.assertFalse(record.get("ext_trailedSources_Naive_flag_edge"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag_off_image"))
            self.assertTrue(record.get("ext_trailedSources_Naive_flag_nan"))

        # Test case with trailed source extending off chip. One coordinate
        # is off image the other is nan, so edge, off_image, and nan should
        # be set.
        else:
            self.assertEqual(record['truth_x'], -20)
            self.assertTrue(record.get("ext_trailedSources_Naive_flag_edge"))
            self.assertFalse(record.get("ext_trailedSources_Naive_flag"))
            self.assertTrue(record.get("ext_trailedSources_Naive_flag_off_image"))
            self.assertTrue(record.get("ext_trailedSources_Naive_flag_nan"))

        patcher.stop()


@classParameters(length=[10], theta=[5], xc=[-20], yc=[-30])
class TrailedEdgeSourcesOffImageTest(AlgorithmTestCase, lsst.utils.tests.TestCase):
    """ Test if ext_trailedSources_Naive_flag_edge is set correctly.

    Given a `TrailedSource`, test if the edge flag is set correctly in the
    source catalog after the
    'lsst.meas.extensions.trailedSources.Naive.Plugin.makeTrailedSourceMeasurementTask'
    has been run on the source catalog.
    """

    def setUp(self):
        self.center = Point2D(50.1, 49.8)
        self.bbox = Box2I(lsst.geom.Point2I(-20, -30), Extent2I(140, 160))
        self.dataset = TrailedTestDataset(self.bbox)

        # Trail which extends into edge pixels
        self.trail = TrailedEdgeSource(100000.0, self.length, self.theta,
                                       self.xc, self.yc)
        self.dataset.addTrailedSource(self.trail, edge=False)

    def tearDown(self):
        del self.center
        del self.bbox
        del self.trail
        del self.dataset

    def testOffImageEdgeFlag(self):
        """Test if edge flags are correctly set in NaivePlugin.py when source
        extends off the the image.

        Given a `TrailedTestDataset`, run the NaivePlugin measurement and
        check that the edge flag set when a source extends off the chip.
        Edge pixels are not set in this test.
        """
        # Set up and run Naive measurement.
        task = TrailedTaskSetup.makeTrailedSourceMeasurementTask(self,
                                                                 plugin="ext_trailedSources_Naive",
                                                                 dependencies=("base_SdssCentroid",
                                                                               "base_SdssShape")
                                                                 )
        exposure, catalog = self.dataset.realize(5.0, task.schema, randomSeed=0)
        task.run(catalog, exposure)
        record = catalog[0]

        self.assertTrue(record.get("ext_trailedSources_Naive_flag_edge"))
        self.assertFalse(record.get("ext_trailedSources_Naive_flag"))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
