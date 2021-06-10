// -*- lsst-c++ -*-
/*
 * This file is part of meas_extensions_trailedSources.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef LSST_MEAS_EXTENSIONS_TRAILEDSOURCES_VERES_MODEL_H
#define LSST_MEAS_EXTENSIONS_TRAILEDSOURCES_VERES_MODEL_H

#include "ndarray.h"

#include "lsst/geom.h"
#include "lsst/pex/config.h"
#include "lsst/afw/image.h"
#include "lsst/afw/geom.h"

namespace lsst {
namespace meas {
namespace extensions {
namespace trailedSources {

/**
 * Implementation of an axisymmetric 2D Gaussian convolved with a line -- a model
 * for a fast-moving, trailed-source (Veres et al. 2012).
 *
 * VeresModel is designed to compute the chi-squared of the model given an
 * lsst::afw::image::Exposure.
 */
class VeresModel final {
public:
    using ImageF = afw::image::Image<float>;
    using ExposureF = afw::image::Exposure<float>;

    /**
     * Constructor for VeresModel.
     *
     * @param data Exposure passed from the measurement task.
     */
    explicit VeresModel(ExposureF const& data);

    VeresModel(VeresModel const &) = default;
    VeresModel(VeresModel &&) = default;
    VeresModel & operator=(VeresModel const &) = default;
    VeresModel & operator=(VeresModel &&) = default;
    ~VeresModel() = default;

    /**
     * Compute chi-squared of the model given the data.
     *
     * @param params Model parameters.
     *
     * @return The chi-squared of the model.
     *
     * @note The params vector contains the following model parameters:
     *     - Centroid x: X-coordinate of the centroid in given image [pixels].
     *     - Centroid y: Y-coordinate of the centroid in given image [pixels].
     *     - Flux: Total flux in the trail [count].
     *     - Length: Length of the trail [pixels].
     *     - Angle: Angle from the +x-axis [radians].
     */
    double operator()(std::vector<double> const& params) const;

    /**
     * Compute the gradient of chi-squared of the model given the data.
     *
     * @param params Model parameters.
     *
     * @return The gradient of chi-squared of the model with respect to the model
     *     parameters.
     *
     * @note The params vector contains the following model parameters:
     *     - Centroid x: X-coordinate of the centroid in given image [pixels].
     *     - Centroid y: Y-coordinate of the centroid in given image [pixels].
     *     - Flux: Total flux in the trail [count].
     *     - Length: Length of the trail [pixels].
     *     - Angle: Angle from the +x-axis [radians].
     */
    std::vector<double> gradient(std::vector<double> const& params) const;

    /// Compute an image for a trail generated from the Veres model.
    std::shared_ptr<ImageF> computeModelImage(std::vector<double> const& params) const;

    /// Return the PSF sigma.
    double getSigma() const noexcept { return _sigma; }

private:
    /**
     * Computes the Veres et al. 2012 model for a given pixel located at (x,y).
     *
     * @param[in] x x-coordinate of a pixel in the image [pixels].
     * @param[in] y y-coordinate of a pixel in the image [pixels].
     * @param[in] xc x-coordinate of the trail centroid [pixels].
     * @param[in] yc y-coordinate of the trail centroid [pixels].
     * @param[in] flux Total flux in the trail [count].
     * @param[in] length Length of the trail [pixels].
     * @param[in] theta Angle of the trail from the +x-axis [radians].
     *
     * @return Value of the model for the given parameters.
     */
    double _computeModel(double x, double y, double xc, double yc,
                         double flux, double length, double theta) const noexcept;

    /**
     * Computes the gradient of the Veres et al. 2012 model for a given pixel located at (x,y).
     *
     * @param[in] x x-coordinate of a pixel in the image [pixels].
     * @param[in] y y-coordinate of a pixel in the image [pixels].
     * @param[in] xc x-coordinate of the trail centroid [pixels].
     * @param[in] yc y-coordinate of the trail centroid [pixels].
     * @param[in] flux Total flux in the trail [count].
     * @param[in] length Length of the trail [pixels].
     * @param[in] theta Angle of the trail from the +x-axis [radians].
     *
     * @return Gradient of the model with respect to each of the input parameters.
     */
    std::array<double, 5> _computeGradient(double x, double y, double xc, double yc,
                         double flux, double length, double theta) const noexcept;

    double _sigma;           // PSF sigma
    lsst::geom::Box2I _bbox; // Data bounding box
    ImageF::Array _data;     // Image array of exposure
    ImageF::Array _variance; // Variance array of exposure
};

}}}} // lsst::meas::extensions::trailedSources

#endif // LSST_MEAS_EXTENSIONS_TRAILEDSOURCES_VERES_MODEL_H