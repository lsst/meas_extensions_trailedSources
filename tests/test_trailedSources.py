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
from scipy.optimize import check_grad
import lsst.afw.table as afwTable
from lsst.meas.base.tests import AlgorithmTestCase
from lsst.meas.extensions.trailedSources import SingleFrameNaiveTrailPlugin
from lsst.meas.extensions.trailedSources import VeresModel
from lsst.meas.extensions.trailedSources.utils import getMeasurementCutout
from lsst.utils.tests import classParameters


# Trailed-source length, angle, and centroid.
rng = np.random.default_rng(432)
nTrails = 50
Ls = rng.uniform(2, 20, nTrails)
thetas = rng.uniform(0, 2*np.pi, nTrails)
xcs = rng.uniform(0, 100, nTrails)
ycs = rng.uniform(0, 100, nTrails)


class TrailedSource:
    """Holds a set of true trail parameters.
    """

    def __init__(self, instFlux, length, angle, xc, yc):
        self.instFlux = instFlux
        self.length = length
        self.angle = angle
        self.center = lsst.geom.Point2D(xc, yc)
        self.x0 = xc - length/2 * np.cos(angle)
        self.y0 = yc - length/2 * np.sin(angle)
        self.x1 = xc + length/2 * np.cos(angle)
        self.y1 = yc + length/2 * np.sin(angle)


# "Extend" meas.base.tests.TestDataset
class TrailedTestDataset(lsst.meas.base.tests.TestDataset):
    """A dataset for testing trailed source measurements.
    Given a `TrailedSource`, construct a record of the true values and an
    Exposure.
    """

    def __init__(self, bbox, threshold=10.0, exposure=None, **kwds):
        super().__init__(bbox, threshold, exposure, **kwds)

    def addTrailedSource(self, trail):
        """Add a trailed source to the simulation.
        'Re-implemented' version of
        `lsst.meas.base.tests.TestDataset.addSource`. Numerically integrates a
        Gaussian PSF over a line to obtain am image of a trailed source.
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
        numIter = int(10*trail.length)
        xp = np.linspace(trail.x0, trail.x1, num=numIter)
        yp = np.linspace(trail.y0, trail.y1, num=numIter)
        for (x, y) in zip(xp, yp):
            pt = lsst.geom.Point2D(x, y)
            im = self.drawGaussian(self.exposure.getBBox(), trail.instFlux,
                                   lsst.afw.geom.Ellipse(self.psfShape, pt))
            self.exposure.getMaskedImage().getImage().getArray()[:, :] += im.getArray()

        totFlux = self.exposure.image.array.sum()
        self.exposure.image.array /= totFlux
        self.exposure.image.array *= trail.instFlux

        record.set(self.keys["instFlux"], trail.instFlux)
        self._installFootprint(record, self.exposure.getImage())

        return record, self.exposure.getImage()


# Following from meas_base/test_NaiveCentroid.py
# Taken from NaiveCentroidTestCase
@classParameters(length=Ls, theta=thetas, xc=xcs, yc=ycs)
class TrailedSourcesTestCase(AlgorithmTestCase, lsst.utils.tests.TestCase):

    def setUp(self):
        self.center = lsst.geom.Point2D(50.1, 49.8)
        self.bbox = lsst.geom.Box2I(lsst.geom.Point2I(-20, -30),
                                    lsst.geom.Extent2I(140, 160))
        self.dataset = TrailedTestDataset(self.bbox)

        self.trail = TrailedSource(100000.0, self.length, self.theta, self.xc, self.yc)
        self.dataset.addTrailedSource(self.trail)

    @staticmethod
    def transformMoments(Ixx, Iyy, Ixy):
        """Transform second-moments to semi-major and minor axis.
        """
        xmy = Ixx - Iyy
        xpy = Ixx + Iyy
        xmy2 = xmy*xmy
        xy2 = Ixy*Ixy
        a2 = 0.5 * (xpy + np.sqrt(xmy2 + 4.0*xy2))
        b2 = 0.5 * (xpy - np.sqrt(xmy2 + 4.0*xy2))
        return a2, b2

    @staticmethod
    def f_length(x):
        return SingleFrameNaiveTrailPlugin.findLength(*x)[0]

    @staticmethod
    def g_length(x):
        return SingleFrameNaiveTrailPlugin.findLength(*x)[1]

    @staticmethod
    def f_flux(x, model):
        return model.computeFluxWithGradient(x)[0]

    @staticmethod
    def g_flux(x, model):
        return model.computeFluxWithGradient(x)[1]

    @staticmethod
    def central_difference(func, x, *args, h=1e-8):
        result = np.zeros(len(x))
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += h
            fp = func(xp, *args)

            xm = x.copy()
            xm[i] -= h
            fm = func(xm, *args)
            result[i] = (fp - fm) / (2*h)

        return result

    def makeTrailedSourceMeasurementTask(self, plugin=None, dependencies=(),
                                         config=None, schema=None, algMetadata=None):
        """Set up a measurement task for a trailed source plugin.
        """

        config = self.makeSingleFrameMeasurementConfig(plugin=plugin,
                                                       dependencies=dependencies)

        # Make sure the shape slot is base_SdssShape
        config.slots.shape = "base_SdssShape"
        return self.makeSingleFrameMeasurementTask(plugin=plugin,
                                                   dependencies=dependencies,
                                                   config=config, schema=schema,
                                                   algMetadata=algMetadata)

    def testNaivePlugin(self):
        """Test the NaivePlugin measurements.
        Given a `TrailedTestDataset`, run the NaivePlugin measurement and
        compare the measured parameters to the true values.
        """

        # Set up and run Naive measurement.
        task = self.makeTrailedSourceMeasurementTask(
            plugin="ext_trailedSources_Naive",
            dependencies=("base_SdssCentroid", "base_SdssShape")
        )
        exposure, catalog = self.dataset.realize(10.0, task.schema, randomSeed=0)
        task.run(catalog, exposure)
        record = catalog[0]

        # Check the RA and Dec measurements
        wcs = exposure.getWcs()
        spt = wcs.pixelToSky(self.center)
        ra_true = spt.getRa().asDegrees()
        dec_true = spt.getDec().asDegrees()
        ra_meas = record.get("ext_trailedSources_Naive_ra")
        dec_meas = record.get("ext_trailedSources_Naive_dec")
        self.assertFloatsAlmostEqual(ra_true, ra_meas, atol=None, rtol=0.01)
        self.assertFloatsAlmostEqual(dec_true, dec_meas, atol=None, rtol=0.01)

        # Check that root finder converged
        converged = record.get("ext_trailedSources_Naive_flag_noConverge")
        self.assertFalse(converged)

        # Compare true with measured length, angle, and flux.
        # Accuracy is dependent on the second-moments measurements, so the
        # rtol values are simply rough upper bounds.
        length = record.get("ext_trailedSources_Naive_length")
        theta = record.get("ext_trailedSources_Naive_angle")
        flux = record.get("ext_trailedSources_Naive_flux")
        self.assertFloatsAlmostEqual(length, self.trail.length, atol=None, rtol=0.1)
        self.assertFloatsAlmostEqual(theta % np.pi, self.trail.angle % np.pi,
                                     atol=np.arctan(1/length), rtol=None)
        self.assertFloatsAlmostEqual(flux, self.trail.instFlux, atol=None, rtol=0.1)

        # Test function gradients versus finite difference derivatives
        # Do length first
        Ixx, Iyy, Ixy = record.getShape().getParameterVector()
        a2, b2 = self.transformMoments(Ixx, Iyy, Ixy)
        self.assertLessEqual(check_grad(self.f_length, self.g_length, [a2, b2]), 1e-6)

        # Now flux gradient
        xc = record.get("base_SdssShape_x")
        yc = record.get("base_SdssShape_y")
        params = np.array([xc, yc, 1.0, length, theta])
        cutout = getMeasurementCutout(record, exposure)
        model = VeresModel(cutout)
        gradNum = self.central_difference(self.f_flux, params, model, h=9e-5)
        gradMax = np.max(np.abs(gradNum - self.g_flux(params, model)))
        self.assertLessEqual(gradMax, 1e-5)

        # Check test setup
        self.assertNotEqual(length, self.trail.length)
        self.assertNotEqual(theta, self.trail.angle)

        # Make sure measurement flag is False
        self.assertFalse(record.get("ext_trailedSources_Naive_flag"))

    def testVeresPlugin(self):
        """Test the VeresPlugin measurements.
        Given a `TrailedTestDataset`, run the VeresPlugin measurement and
        compare the measured parameters to the true values.
        """

        # Set up and run Veres measurement.
        task = self.makeTrailedSourceMeasurementTask(
            plugin="ext_trailedSources_Veres",
            dependencies=(
                "base_SdssCentroid",
                "base_SdssShape",
                "ext_trailedSources_Naive")
        )
        exposure, catalog = self.dataset.realize(10.0, task.schema, randomSeed=0)
        task.run(catalog, exposure)
        record = catalog[0]

        # Make sure optmizer converged
        converged = record.get("ext_trailedSources_Veres_flag_nonConvergence")
        self.assertFalse(converged)

        # Compare measured trail length, angle, and flux to true values
        # These measurements should perform at least as well as NaivePlugin
        length = record.get("ext_trailedSources_Veres_length")
        theta = record.get("ext_trailedSources_Veres_angle")
        flux = record.get("ext_trailedSources_Veres_flux")
        self.assertFloatsAlmostEqual(length, self.trail.length, atol=None, rtol=0.1)
        self.assertFloatsAlmostEqual(theta % np.pi, self.trail.angle % np.pi,
                                     atol=np.arctan(1/length), rtol=None)
        self.assertFloatsAlmostEqual(flux, self.trail.instFlux, atol=None, rtol=0.1)

        xc = record.get("ext_trailedSources_Veres_centroid_x")
        yc = record.get("ext_trailedSources_Veres_centroid_y")
        params = np.array([xc, yc, flux, length, theta])
        cutout = getMeasurementCutout(record, exposure)
        model = VeresModel(cutout)
        gradNum = self.central_difference(model, params, h=1e-6)
        gradMax = np.max(np.abs(gradNum - model.gradient(params)))
        self.assertLessEqual(gradMax, 1e-5)

        # Make sure test setup is working as expected
        self.assertNotEqual(length, self.trail.length)
        self.assertNotEqual(theta, self.trail.angle)

        # Test that reduced chi-squared is reasonable
        rChiSq = record.get("ext_trailedSources_Veres_rChiSq")
        self.assertGreater(rChiSq, 0.8)
        self.assertLess(rChiSq, 1.3)

        # Make sure measurement flag is False
        self.assertFalse(record.get("ext_trailedSources_Veres_flag"))

    def testMonteCarlo(self):
        """Test the uncertainties in trail measurements from NaivePlugin
        """
        # Adapted from lsst.meas.base

        # Set up Naive measurement and dependencies.
        task = self.makeTrailedSourceMeasurementTask(
            plugin="ext_trailedSources_Naive",
            dependencies=("base_SdssCentroid", "base_SdssShape")
        )

        nSamples = 2000
        catalog = afwTable.SourceCatalog(task.schema)
        sample = 0
        seed = 0
        while sample < nSamples:
            seed += 1
            exp, cat = self.dataset.realize(100.0, task.schema, randomSeed=seed)
            rec = cat[0]
            task.run(cat, exp)

            # Accuracy of this measurement is entirely dependent on shape and
            # centroiding. Skip when shape measurement fails.
            if rec['base_SdssShape_flag']:
                continue
            catalog.append(rec)
            sample += 1

        catalog = catalog.copy(deep=True)
        nameBase = "ext_trailedSources_Naive_"

        # Currently, the errors don't include covariances, so just make sure
        # we're close or at least over estimate
        length = catalog[nameBase+"length"]
        lengthErr = catalog[nameBase+"lengthErr"]
        lengthStd = np.nanstd(length)
        lengthErrMean = np.nanmean(lengthErr)
        diff = (lengthErrMean - lengthStd) / lengthErrMean
        self.assertGreater(diff, -0.1)
        self.assertLess(diff, 0.5)

        angle = catalog[nameBase+"angle"]
        if (np.max(angle) - np.min(angle)) > np.pi/2:
            angle = angle % np.pi  # Wrap if bimodal
        angleErr = catalog[nameBase+"angleErr"]
        angleStd = np.nanstd(angle)
        angleErrMean = np.nanmean(angleErr)
        diff = (angleErrMean - angleStd) / angleErrMean
        self.assertGreater(diff, -0.1)
        self.assertLess(diff, 0.6)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
