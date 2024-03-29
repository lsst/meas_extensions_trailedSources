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

from lsst.pex.config import Field

from lsst.meas.base.pluginRegistry import register
from lsst.meas.base import SingleFramePlugin, SingleFramePluginConfig
from lsst.meas.base import FlagHandler, FlagDefinitionList, SafeCentroidExtractor

from ._trailedSources import VeresModel
from .NaivePlugin import SingleFrameNaiveTrailPlugin
from .utils import getMeasurementCutout

__all__ = ("SingleFrameVeresTrailConfig", "SingleFrameVeresTrailPlugin")


class SingleFrameVeresTrailConfig(SingleFramePluginConfig):
    """Config class for SingleFrameVeresTrailPlugin
    """

    optimizerMethod = Field(
        doc="Optimizer method for scipy.optimize.minimize",
        dtype=str,
        default="L-BFGS-B"
    )


@register("ext_trailedSources_Veres")
class SingleFrameVeresTrailPlugin(SingleFramePlugin):
    """Veres trailed source characterization plugin.

    Measures the length, angle, flux, centroid, and end points of a trailed
    source using the Veres et al. 2012 model [1]_.

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
    This plugin is designed to refine the measurements of trail length,
    angle, and end points from `NaivePlugin`, and of flux and centroid from
    previous measurement algorithms. Vereš et al. 2012 [1]_ derive a model for
    the flux in a given image pixel by convolving an axisymmetric Gaussian with
    a line. The model is parameterized by the total flux, trail length, angle
    from the x-axis, and the centroid. The best estimates are computed using a
    chi-squared minimization.

    References
    ----------
    .. [1] Vereš, P., et al. "Improved Asteroid Astrometry and Photometry with
       Trail Fitting" PASP, vol. 124, 2012.

    See also
    --------
    lsst.meas.base.SingleFramePlugin
    """

    ConfigClass = SingleFrameVeresTrailConfig

    @classmethod
    def getExecutionOrder(cls):
        # Needs centroids, shape, flux, and NaivePlugin measurements.
        # Make sure this always runs after NaivePlugin.
        return SingleFrameNaiveTrailPlugin.getExecutionOrder() + 0.1

    def __init__(self, config, name, schema, metadata, logName=None):
        super().__init__(config, name, schema, metadata, logName=logName)

        self.keyXC = schema.addField(
            name + "_centroid_x", type="D", doc="Trail centroid X coordinate.", units="pixel")
        self.keyYC = schema.addField(
            name + "_centroid_y", type="D", doc="Trail centroid Y coordinate.", units="pixel")
        self.keyX0 = schema.addField(name + "_x0", type="D", doc="Trail head X coordinate.", units="pixel")
        self.keyY0 = schema.addField(name + "_y0", type="D", doc="Trail head Y coordinate.", units="pixel")
        self.keyX1 = schema.addField(name + "_x1", type="D", doc="Trail tail X coordinate.", units="pixel")
        self.keyY1 = schema.addField(name + "_y1", type="D", doc="Trail tail Y coordinate.", units="pixel")
        self.keyLength = schema.addField(name + "_length", type="D", doc="Length of trail.", units="pixel")
        self.keyTheta = schema.addField(name + "_angle", type="D", doc="Angle of trail from +x-axis.")
        self.keyFlux = schema.addField(name + "_flux", type="D", doc="Trailed source flux.", units="count")
        self.keyRChiSq = schema.addField(name + "_rChiSq", type="D", doc="Reduced chi-squared of fit")

        flagDefs = FlagDefinitionList()
        self.FAILURE = flagDefs.addFailureFlag("No trailed-sources measured")
        self.NON_CONVERGE = flagDefs.add("flag_nonConvergence", "Optimizer did not converge")
        self.NO_NAIVE = flagDefs.add("flag_noNaive", "Naive measurement contains NaNs")
        self.flagHandler = FlagHandler.addFields(schema, name, flagDefs)

        self.centroidExtractor = SafeCentroidExtractor(schema, name)

    def measure(self, measRecord, exposure):
        """Run the Veres trailed source measurement plugin.

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
        xc, yc = self.centroidExtractor(measRecord, self.flagHandler)

        # Look at measRecord for Naive measurements
        # ASSUMES NAIVE ALREADY RAN
        flux = measRecord.get("ext_trailedSources_Naive_flux")
        length = measRecord.get("ext_trailedSources_Naive_length")
        theta = measRecord.get("ext_trailedSources_Naive_angle")
        if not np.isfinite(flux) or not np.isfinite(length) or not np.isfinite(theta):
            self.flagHandler.setValue(measRecord, self.NO_NAIVE.number)
            self.flagHandler.setValue(measRecord, self.FAILURE.number)
            return

        # Get exposure cutout
        # sigma = exposure.getPsf().getSigma()
        # cutout = getMeasurementCutout(exposure, xc, yc, length, sigma)
        cutout = getMeasurementCutout(measRecord, exposure)

        # Make VeresModel
        model = VeresModel(cutout)

        # Do optimization with scipy
        params = np.array([xc, yc, flux, length, theta])
        results = sciOpt.minimize(
            model, params, method=self.config.optimizerMethod, jac=model.gradient)

        # Check if optimizer converged
        if not results.success:
            self.flagHandler.setValue(measRecord, self.NON_CONVERGE.number)
            self.flagHandler.setValue(measRecord, self.FAILURE.number)
            return

        # Calculate end points and reduced chi-squared
        xc_fit, yc_fit, flux_fit, length_fit, theta_fit = results.x
        a = length_fit/2
        x0_fit = xc_fit - a * np.cos(theta_fit)
        y0_fit = yc_fit - a * np.sin(theta_fit)
        x1_fit = xc_fit + a * np.cos(theta_fit)
        y1_fit = yc_fit + a * np.sin(theta_fit)
        rChiSq = results.fun / (cutout.image.array.size - 6)

        # Set keys
        measRecord.set(self.keyXC, xc_fit)
        measRecord.set(self.keyYC, yc_fit)
        measRecord.set(self.keyX0, x0_fit)
        measRecord.set(self.keyY0, y0_fit)
        measRecord.set(self.keyX1, x1_fit)
        measRecord.set(self.keyY1, y1_fit)
        measRecord.set(self.keyFlux, flux_fit)
        measRecord.set(self.keyLength, length_fit)
        measRecord.set(self.keyTheta, theta_fit)
        measRecord.set(self.keyRChiSq, rChiSq)

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
