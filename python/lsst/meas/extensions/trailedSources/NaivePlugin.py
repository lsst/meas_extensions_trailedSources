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

import logging
import numpy as np
import scipy.optimize as sciOpt
from scipy.special import erf
from math import sqrt

from lsst.geom import Point2D, Point2I
from lsst.meas.base.pluginRegistry import register
from lsst.meas.base import SingleFramePlugin, SingleFramePluginConfig
from lsst.meas.base import FlagHandler, FlagDefinitionList
import lsst.pex.config

from ._trailedSources import VeresModel
from .utils import getMeasurementCutout

__all__ = ("SingleFrameNaiveTrailConfig", "SingleFrameNaiveTrailPlugin")


class SingleFrameNaiveTrailConfig(SingleFramePluginConfig):
    """Config class for SingleFrameNaiveTrailPlugin.
    """
    maxFlux = lsst.pex.config.Field(
        dtype=float,
        default=1e10,
        doc="Maximum calculated model flux before falling back on aperture flux."
    )


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

    def __init__(self, config, name, schema, metadata, logName=None):
        if logName is None:
            logName = __name__
        super().__init__(config, name, schema, metadata, logName=logName)

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
        self.keyFluxErr = schema.addField(name + "_fluxErr", type="D",
                                          doc="Trail flux error.", units="count")
        self.keyLengthErr = schema.addField(name + "_lengthErr", type="D",
                                            doc="Trail length error.", units="pixel")
        self.keyAngleErr = schema.addField(name + "_angleErr", type="D", doc="Trail angle error.")

        flagDefs = FlagDefinitionList()
        self.FAILURE = flagDefs.addFailureFlag("No trailed-source measured")
        self.NO_FLUX = flagDefs.add("flag_noFlux", "No suitable prior flux measurement")
        self.NO_CONVERGE = flagDefs.add("flag_noConverge", "The root finder did not converge")
        self.NO_SIGMA = flagDefs.add("flag_noSigma", "No PSF width (sigma)")
        self.EDGE = flagDefs.add("flag_edge", "Trail contains edge pixels")
        self.OFFIMAGE = flagDefs.add("flag_off_image", "Trail extends off image")
        self.NAN = flagDefs.add("flag_nan", "One or more trail coordinates are missing")
        self.SUSPECT_LONG_TRAIL = flagDefs.add("flag_suspect_long_trail",
                                               "Trail length is greater than three times the psf radius")
        self.SHAPE = flagDefs.add("flag_shape", "Shape flag is set, trail length not calculated")
        self.flagHandler = FlagHandler.addFields(schema, name, flagDefs)

        self.log = logging.getLogger(self.logName)

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
        if measRecord.getShapeFlag():
            self.log.debug("Shape flag is set for measRecord: %s. Trail measurement "
                           "will not be made. All trail values will be set to nan.", measRecord.getId())
            self.flagHandler.setValue(measRecord, self.FAILURE.number, True)
            self.flagHandler.setValue(measRecord, self.SHAPE.number, True)
            return

        xc = measRecord["slot_Shape_x"]
        yc = measRecord["slot_Shape_y"]
        if not np.isfinite(xc) or not np.isfinite(yc):
            self.flagHandler.setValue(measRecord, self.SAFE_CENTROID.number, True)
            self.flagHandler.setValue(measRecord, self.FAILURE.number, True)
            return
        ra, dec = self.computeRaDec(exposure, xc, yc)

        # Transform the second-moments to semi-major and minor axes
        Ixx, Iyy, Ixy = measRecord.getShape().getParameterVector()
        xmy = Ixx - Iyy
        xpy = Ixx + Iyy
        xmy2 = xmy*xmy
        xy2 = Ixy*Ixy
        a2 = 0.5 * (xpy + sqrt(xmy2 + 4.0*xy2))
        b2 = 0.5 * (xpy - sqrt(xmy2 + 4.0*xy2))

        # Measure the trail length
        length, gradLength, results = self.findLength(a2, b2)
        if not results.converged:
            self.log.info("Results not converged: %s", results.flag)
            self.flagHandler.setValue(measRecord, self.NO_CONVERGE.number, True)
            self.flagHandler.setValue(measRecord, self.FAILURE.number, True)
            return

        # Compute the angle of the trail from the x-axis
        theta = 0.5 * np.arctan2(2.0 * Ixy, xmy)

        # Get end-points of the trail (there is a degeneracy here)
        radius = length/2.0  # Trail 'radius'
        dydtheta = radius*np.cos(theta)
        dxdtheta = radius*np.sin(theta)
        x0 = xc - dydtheta
        y0 = yc - dxdtheta
        x1 = xc + dydtheta
        y1 = yc + dxdtheta

        self.check_trail(measRecord, exposure, x0, y0, x1, y1, length)

        # Get a cutout of the object from the exposure
        cutout = getMeasurementCutout(measRecord, exposure)

        # Compute flux assuming fixed parameters for VeresModel
        params = np.array([xc, yc, 1.0, length, theta])  # Flux = 1.0
        model = VeresModel(cutout)
        flux, gradFlux = model.computeFluxWithGradient(params)

        # Fall back to aperture flux
        if (not np.isfinite(flux)) | (np.abs(flux) > self.config.maxFlux):
            if np.isfinite(measRecord.getApInstFlux()):
                flux = measRecord.getApInstFlux()
            else:
                self.flagHandler.setValue(measRecord, self.NO_FLUX.number, True)
                self.flagHandler.setValue(measRecord, self.FAILURE.number, True)
                return

        # Propogate errors from second moments and centroid
        IxxErr2, IyyErr2, IxyErr2 = np.diag(measRecord.getShapeErr())

        # SdssShape does not produce centroid errors. The
        # Slot centroid errors will suffice for now.
        xcErr2, ycErr2 = np.diag(measRecord.getCentroidErr())

        # Error in length
        desc = sqrt(xmy2 + 4.0*xy2)  # Descriminant^1/2 of EV equation
        da2dIxx = 0.5*(1.0 + (xmy/desc))
        da2dIyy = 0.5*(1.0 - (xmy/desc))
        da2dIxy = 2.0*Ixy / desc
        a2Err2 = IxxErr2*da2dIxx*da2dIxx + IyyErr2*da2dIyy*da2dIyy + IxyErr2*da2dIxy*da2dIxy
        b2Err2 = IxxErr2*da2dIyy*da2dIyy + IyyErr2*da2dIxx*da2dIxx + IxyErr2*da2dIxy*da2dIxy
        dLda2, dLdb2 = gradLength
        lengthErr = np.sqrt(dLda2*dLda2*a2Err2 + dLdb2*dLdb2*b2Err2)

        # Error in theta
        dThetadIxx = -Ixy / (xmy2 + 4.0*xy2)  # dThetadIxx = -dThetadIyy
        dThetadIxy = xmy / (xmy2 + 4.0*xy2)
        thetaErr = sqrt(dThetadIxx*dThetadIxx*(IxxErr2 + IyyErr2) + dThetadIxy*dThetadIxy*IxyErr2)

        # Error in flux
        dFdxc, dFdyc, _, dFdL, dFdTheta = gradFlux
        fluxErr = sqrt(dFdL*dFdL*lengthErr*lengthErr + dFdTheta*dFdTheta*thetaErr*thetaErr
                       + dFdxc*dFdxc*xcErr2 + dFdyc*dFdyc*ycErr2)

        # Errors in end-points
        dxdradius = np.cos(theta)
        dydradius = np.sin(theta)
        radiusErr2 = lengthErr*lengthErr/4.0
        xErr2 = sqrt(xcErr2 + radiusErr2*dxdradius*dxdradius + thetaErr*thetaErr*dxdtheta*dxdtheta)
        yErr2 = sqrt(ycErr2 + radiusErr2*dydradius*dydradius + thetaErr*thetaErr*dydtheta*dydtheta)
        x0Err = sqrt(xErr2)  # Same for x1
        y0Err = sqrt(yErr2)  # Same for y1

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
        measRecord.set(self.keyFluxErr, fluxErr)
        measRecord.set(self.keyLengthErr, lengthErr)
        measRecord.set(self.keyAngleErr, thetaErr)

    def check_trail(self, measRecord, exposure, x0, y0, x1, y1, length):
        """ Set flags for edge pixels, off chip, and nan trail coordinates and
        flag if trail length is three times larger than psf.

        Check if the coordinates of the beginning and ending of the trail fall
        inside the exposures bounding box. If not, set the off_chip flag.
        If the beginning or ending falls within a pixel marked as edge, set the
        edge flag. If any of the coordinates happens to fall on a nan, then
        set the nan flag.
        Additionally, check if the trail is three times larger than the psf. If
        so, set the suspect trail flag.

        Parameters
        ----------
        measRecord: `lsst.afw.MeasurementRecord`
            Record describing the object being measured.
        exposure: `lsst.afw.Exposure`
            Pixel data to be measured.

        x0: `float`
            x coordinate of the beginning of the trail.
        y0: `float`
            y coordinate of the beginning of the trail.
        x1: `float`
            x coordinate of the end of the trail.
        y1: `float`
            y coordinate of the end of the trail.
        """
        x_coords = [x0, x1]
        y_coords = [y0, y1]

        # Check if one of the end points of the trail sources is nan. If so,
        # set the trailed source nan flag.
        if np.isnan(x_coords).any() or np.isnan(y_coords).any():
            self.flagHandler.setValue(measRecord, self.NAN.number, True)
            x_coords = [x for x in x_coords if not np.isnan(x)]
            y_coords = [y for y in y_coords if not np.isnan(y)]

            # Check if the non-nan coordinates are within the bounding box
            if not (all(exposure.getBBox().beginX <= x <= exposure.getBBox().endX for x in x_coords)
                    and all(exposure.getBBox().beginY <= y <= exposure.getBBox().endY for y in y_coords)):
                self.flagHandler.setValue(measRecord, self.EDGE.number, True)
                self.flagHandler.setValue(measRecord, self.OFFIMAGE.number, True)
            else:
                # Check if edge is set for any of the pixel pairs. Do not
                # check any that have a nan.
                for (x_val, y_val) in zip(x_coords, y_coords):
                    if x_val is not np.nan and y_val is not np.nan:
                        if exposure.mask[Point2I(int(x_val),
                                                 int(y_val))] & exposure.mask.getPlaneBitMask('EDGE') != 0:
                            self.flagHandler.setValue(measRecord, self.EDGE.number, True)
        # Check whether trail extends off the edge of the exposure. Allows nans
        # as their location
        elif not (all(exposure.getBBox().beginX <= x <= exposure.getBBox().endX for x in x_coords)
                  and all(exposure.getBBox().beginY <= y <= exposure.getBBox().endY for y in y_coords)):
            self.flagHandler.setValue(measRecord, self.EDGE.number, True)
            self.flagHandler.setValue(measRecord, self.OFFIMAGE.number, True)
        else:
            # Check whether the beginning or end point of the trail has the
            # edge flag set. The end points are not whole pixel values, so
            # the pixel value must be rounded.
            if exposure.mask[Point2I(int(x0), int(y0))] and exposure.mask[Point2I(int(x1), int(y1))]:
                if ((exposure.mask[Point2I(int(x0), int(y0))] & exposure.mask.getPlaneBitMask('EDGE') != 0)
                        or (exposure.mask[Point2I(int(x1), int(y1))]
                            & exposure.mask.getPlaneBitMask('EDGE') != 0)):
                    self.flagHandler.setValue(measRecord, self.EDGE.number, True)

        psfShape = exposure.psf.computeShape(exposure.getBBox().getCenter())
        psfRadius = psfShape.getDeterminantRadius()

        if length > psfRadius*3.0:
            self.flagHandler.setValue(measRecord, self.SUSPECT_LONG_TRAIL.number, True)

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

    @staticmethod
    def _computeSecondMomentDiff(z, c):
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

    @classmethod
    def findLength(cls, Ixx, Iyy):
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
        # Empirically, 0.001 is a suitable lower bound, assuming Ixx > Iyy.
        z, results = sciOpt.brentq(lambda z: cls._computeSecondMomentDiff(z, c),
                                   0.001, 1.0, full_output=True)

        length = 2.0*z*np.sqrt(2.0*xpy)
        gradLength = cls._gradFindLength(Ixx, Iyy, z, c)
        return length, gradLength, results

    @staticmethod
    def _gradFindLength(Ixx, Iyy, z, c):
        """Compute the gradient of the findLength function.
        """
        spi = np.sqrt(np.pi)
        xpy = Ixx+Iyy
        xpy2 = xpy*xpy
        enz2 = np.exp(-z*z)
        sxpy = np.sqrt(xpy)

        fac = 4.0 / (spi*xpy2)
        dcdIxx = Iyy*fac
        dcdIyy = -Ixx*fac

        # Derivatives of the _computeMomentsDiff function
        dfdc = z*enz2
        dzdf = spi / (enz2*(spi*c*(2.0*z*z - 1.0) + 2.0))  # inverse of dfdz

        dLdz = 2.0*np.sqrt(2.0)*sxpy
        pLpIxx = np.sqrt(2.0)*z / sxpy  # Same as pLpIyy

        dLdc = dLdz*dzdf*dfdc
        dLdIxx = dLdc*dcdIxx + pLpIxx
        dLdIyy = dLdc*dcdIyy + pLpIxx
        return dLdIxx, dLdIyy

    @staticmethod
    def computeLength(Ixx, Iyy):
        """Compute the length of a trail, given unweighted second-moments.
        """
        denom = np.sqrt(Ixx - 2.0*Iyy)

        length = np.sqrt(6.0)*denom

        dLdIxx = np.sqrt(1.5) / denom
        dLdIyy = -np.sqrt(6.0) / denom
        return length, (dLdIxx, dLdIyy)

    @staticmethod
    def computeRaDec(exposure, x, y):
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
