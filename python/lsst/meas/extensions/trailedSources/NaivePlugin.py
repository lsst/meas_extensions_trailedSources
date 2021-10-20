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
import scipy.optimize as sciOpt
from scipy.special import erf

import lsst.log
from lsst.geom import Point2D
from lsst.meas.base.pluginRegistry import register
from lsst.meas.base import SingleFramePlugin, SingleFramePluginConfig
from lsst.meas.base import FlagHandler, FlagDefinitionList, SafeCentroidExtractor
from lsst.meas.base import MeasurementError

from ._trailedSources import VeresModel
from .utils import getMeasurementCutout

__all__ = ("SingleFrameNaiveTrailConfig", "SingleFrameNaiveTrailPlugin")


class SingleFrameNaiveTrailConfig(SingleFramePluginConfig):
    """Config class for SingleFrameNaiveTrailPlugin.
    """
    pass


@register("ext_trailedSources_Naive")
class SingleFrameNaiveTrailPlugin(SingleFramePlugin):
    """Naive trailed source measurement plugin

    Measures the length, angle from +x-axis, and end points of an extended
    source using the second moments.

    Parameters
    ----------
    config: `SingleFrameNaiveTrailConfig`
        Plugin configuration.
    name: `str`
        Plugin name.
    schema: `lsst.afw.table.Schema`
        Schema for the output catalog.
    metadata: `lsst.daf.base.PropertySet`
        Metadata to be attached to output catalog.

    Notes
    -----
    This measurement plugin aims to utilize the already measured adaptive
    second moments to naively estimate the length and angle, and thus
    end-points, of a fast-moving, trailed source. The length is solved for via
    finding the root of the difference between the numerical (stack computed)
    and the analytic adaptive second moments. The angle, theta, from the x-axis
    is also computed via adaptive moments: theta = arctan(2*Ixy/(Ixx - Iyy))/2.
    The end points of the trail are then given by (xc +/- (length/2)*cos(theta)
    and yc +/- (length/2)*sin(theta)), with xc and yc being the centroid
    coordinates.

    See also
    --------
    lsst.meas.base.SingleFramePlugin
    """

    ConfigClass = SingleFrameNaiveTrailConfig

    @classmethod
    def getExecutionOrder(cls):
        # Needs centroids, shape, and flux measurements.
        # VeresPlugin is run after, which requires image data.
        return cls.APCORR_ORDER + 0.1

    def __init__(self, config, name, schema, metadata):
        super().__init__(config, name, schema, metadata)

        # Measurement Keys
        self.keyRa = schema.addField(name + "_ra", type="D", doc="Trail centroid right ascension.")
        self.keyDec = schema.addField(name + "_dec", type="D", doc="Trail centroid declination.")
        self.keyX0 = schema.addField(name + "_x0", type="D", doc="Trail head X coordinate.", units="pixel")
        self.keyY0 = schema.addField(name + "_y0", type="D", doc="Trail head Y coordinate.", units="pixel")
        self.keyX1 = schema.addField(name + "_x1", type="D", doc="Trail tail X coordinate.", units="pixel")
        self.keyY1 = schema.addField(name + "_y1", type="D", doc="Trail tail Y coordinate.", units="pixel")
        self.keyFlux = schema.addField(name + "_flux", type="D", doc="Trailed source flux.", units="count")
        self.keyLength = schema.addField(name + "_length", type="D", doc="Trail length.", units="pixel")
        self.keyAngle = schema.addField(name + "_angle", type="D", doc="Angle measured from +x-axis.")

        # Measurement Error Keys
        self.keyX0Err = schema.addField(name + "_x0Err", type="D",
                                        doc="Trail head X coordinate error.", units="pixel")
        self.keyY0Err = schema.addField(name + "_y0Err", type="D",
                                        doc="Trail head Y coordinate error.", units="pixel")
        self.keyX1Err = schema.addField(name + "_x1Err", type="D",
                                        doc="Trail tail X coordinate error.", units="pixel")
        self.keyY1Err = schema.addField(name + "_y1Err", type="D",
                                        doc="Trail tail Y coordinate error.", units="pixel")

        flagDefs = FlagDefinitionList()
        flagDefs.addFailureFlag("No trailed-source measured")
        self.NO_FLUX = flagDefs.add("flag_noFlux", "No suitable prior flux measurement")
        self.NO_CONVERGE = flagDefs.add("flag_noConverge", "The root finder did not converge")
        self.NO_SIGMA = flagDefs.add("flag_noSigma", "No PSF width (sigma)")
        self.flagHandler = FlagHandler.addFields(schema, name, flagDefs)

        self.centriodExtractor = SafeCentroidExtractor(schema, name)

    def measure(self, measRecord, exposure):
        """Run the Naive trailed source measurement algorithm.

        Parameters
        ----------
        measRecord : `lsst.afw.table.SourceRecord`
            Record describing the object being measured.
        exposure : `lsst.afw.image.Exposure`
            Pixel data to be measured.

        See also
        --------
        lsst.meas.base.SingleFramePlugin.measure
        """

        # Get the SdssShape centroid or fall back to slot
        xc = measRecord.get("base_SdssShape_x")
        yc = measRecord.get("base_SdssShape_y")
        if not np.isfinite(xc) or not np.isfinite(yc):
            xc, yc = self.centriodExtractor(measRecord, self.flagHandler)

        ra, dec = self.computeRaDec(exposure, xc, yc)

        Ixx, Iyy, Ixy = measRecord.getShape().getParameterVector()
        xmy = Ixx - Iyy
        xpy = Ixx + Iyy
        xmy2 = xmy*xmy
        xy2 = Ixy*Ixy
        a2 = 0.5 * (xpy + np.sqrt(xmy2 + 4.0*xy2))

        # Get the width of the PSF at the center of the trail
        center = Point2D(xc, yc)
        sigma = exposure.getPsf().computeShape(center).getTraceRadius()
        if not np.isfinite(sigma):
            raise MeasurementError(self.NO_SIGMA, self.NO_SIGMA.number)

        # Check if moments are wieghted
        if measRecord.get("base_SdssShape_flag_unweighted"):
            lsst.log.info("Unweighed")
            length = np.sqrt(6.0*(a2 - 2*sigma*sigma))
        else:
            lsst.log.info("Weighted")
            length, results = self.findLength(a2, sigma*sigma)
            if not results.converged:
                lsst.log.info(results.flag)
                raise MeasurementError(self.NO_CONVERGE.doc, self.NO_CONVERGE.number)

        theta = 0.5 * np.arctan2(2.0 * Ixy, xmy)
        a = length/2.0
        dydt = a*np.cos(theta)
        dxdt = a*np.sin(theta)
        x0 = xc - dydt
        y0 = yc - dxdt
        x1 = xc + dydt
        y1 = yc + dxdt

        # Get a cutout of the object from the exposure
        # cutout = getMeasurementCutout(exposure, xc, yc, L, sigma)
        cutout = getMeasurementCutout(measRecord, exposure)

        # Compute flux assuming fixed parameters for VeresModel
        params = np.array([xc, yc, 1.0, length, theta])  # Flux = 1.0
        model = VeresModel(cutout)
        modelArray = model.computeModelImage(params).array.flatten()
        dataArray = cutout.image.array.flatten()
        flux = np.dot(dataArray, modelArray) / np.dot(modelArray, modelArray)

        # Fall back to aperture flux
        if not np.isfinite(flux):
            if np.isfinite(measRecord.getApInstFlux()):
                flux = measRecord.getApInstFlux()
            else:
                raise MeasurementError(self.NO_FLUX.doc, self.NO_FLUX.number)

        # Propagate errors from second moments
        xcErr2, ycErr2 = np.diag(measRecord.getCentroidErr())
        IxxErr2, IyyErr2, IxyErr2 = np.diag(measRecord.getShapeErr())
        desc = np.sqrt(xmy2 + 4.0*xy2)  # Descriminant^1/2 of EV equation
        denom = 2*np.sqrt(2.0*(Ixx + np.sqrt(4.0*xy2 + xmy2 + Iyy)))  # Denominator for dadIxx and dadIyy
        dadIxx = (1.0 + (xmy/desc)) / denom
        dadIyy = (1.0 - (xmy/desc)) / denom
        dadIxy = (4.0*Ixy) / (desc * denom)
        aErr2 = IxxErr2*dadIxx*dadIxx + IyyErr2*dadIyy*dadIyy + IxyErr2*dadIxy*dadIxy
        thetaErr2 = ((IxxErr2 + IyyErr2)*xy2 + xmy2*IxyErr2) / (desc*desc*desc*desc)

        dxda = np.cos(theta)
        dyda = np.sin(theta)
        xErr2 = aErr2*dxda*dxda + thetaErr2*dxdt*dxdt
        yErr2 = aErr2*dyda*dyda + thetaErr2*dydt*dydt
        x0Err = np.sqrt(xErr2 + xcErr2)  # Same for x1
        y0Err = np.sqrt(yErr2 + ycErr2)  # Same for y1

        # Set flags
        measRecord.set(self.keyRa, ra)
        measRecord.set(self.keyDec, dec)
        measRecord.set(self.keyX0, x0)
        measRecord.set(self.keyY0, y0)
        measRecord.set(self.keyX1, x1)
        measRecord.set(self.keyY1, y1)
        measRecord.set(self.keyFlux, flux)
        measRecord.set(self.keyLength, length)
        measRecord.set(self.keyAngle, theta)
        measRecord.set(self.keyX0Err, x0Err)
        measRecord.set(self.keyY0Err, y0Err)
        measRecord.set(self.keyX1Err, x0Err)
        measRecord.set(self.keyY1Err, y0Err)

    def fail(self, measRecord, error=None):
        """Record failure

        See also
        --------
        lsst.meas.base.SingleFramePlugin.fail
        """
        if error is None:
            self.flagHandler.handleFailure(measRecord)
        else:
            self.flagHandler.handleFailure(measRecord, error.cpp)

    def _computeSecondMomentDiff(self, z, c):
        """Compute difference of the numerical and analytic second moments.

        Parameters
        ----------
        z : `float`
            Proportional to the length of the trail. (see notes)
        c : `float`
            Constant (see notes)

        Returns
        -------
        diff : `float`
            Difference in numerical and analytic second moments.

        Notes
        -----
        This is a simplified expression for the difference between the stack
        computed adaptive second-moment and the analytic solution. The variable
        z is proportional to the length such that length=2*z*sqrt(2*(Ixx+Iyy)),
        and c is a constant (c = 4*Ixx/((Ixx+Iyy)*sqrt(pi))). Both have been
        defined to avoid unnecessary floating-point operations in the root
        finder.
        """

        diff = erf(z) - c*z*np.exp(-z*z)
        return diff

    def findLength(self, Ixx, Iyy):
        """Find the length of a trail, given adaptive second-moments.

        Uses a root finder to compute the length of a trail corresponding to
        the adaptive second-moments computed by previous measurements
        (ie. SdssShape).

        Parameters
        ----------
        Ixx : `float`
            Adaptive second-moment along x-axis.
        Iyy : `float`
            Adaptive second-moment along y-axis.

        Returns
        -------
        length : `float`
            Length of the trail.
        results : `scipy.optimize.RootResults`
            Contains messages about convergence from the root finder.
        """

        xpy = Ixx + Iyy
        c = 4.0*Ixx/(xpy*np.sqrt(np.pi))

        # Given a 'c' in (c_min, c_max], the root is contained in (0,1].
        # c_min is given by the case: Ixx == Iyy, ie. a point source.
        # c_max is given by the limit Ixx >> Iyy.
        # Emperically, 0.001 is a suitable lower bound, assuming Ixx > Iyy.
        z, results = sciOpt.brentq(lambda z: self._computeSecondMomentDiff(z, c),
                                   0.001, 1.0, full_output=True)

        length = 2.0*z*np.sqrt(2.0*xpy)
        return length, results

    def computeRaDec(self, exposure, x, y):
        """Convert pixel coordinates to RA and Dec.

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
            Exposure object containing the WCS.
        x : `float`
            x coordinate of the trail centroid
        y : `float`
            y coodinate of the trail centroid

        Returns
        -------
        ra : `float`
            Right ascension.
        dec : `float`
            Declination.
        """

        wcs = exposure.getWcs()
        center = wcs.pixelToSky(Point2D(x, y))
        ra = center.getRa().asDegrees()
        dec = center.getDec().asDegrees()
        return ra, dec
