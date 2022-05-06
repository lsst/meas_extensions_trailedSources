// -*- LSST-C++ -*-
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

#include "lsst/geom.h"
#include "lsst/afw/image.h"
#include "lsst/afw/detection.h"
#include "lsst/meas/extensions/trailedSources/VeresModel.h"

namespace lsst {
namespace meas {
namespace extensions {
namespace trailedSources {

using ImageF = afw::image::Image<float>;
using ExposureF = afw::image::Exposure<float>;
using std::sin, std::cos, std::sqrt, std::exp, std::erf;

VeresModel::VeresModel(
    ExposureF const& data
) : _sigma(data.getPsf()->computeShape().getTraceRadius()),
    _bbox(data.getBBox()),
    _data(data.getMaskedImage().getImage()->getArray()),
    _variance(data.getMaskedImage().getVariance()->getArray()) {}

double VeresModel::operator()(std::vector<double> const& params) const {

    double xc = params[0];     // Centroid x
    double yc = params[1];     // Centroid y
    double flux = params[2];   // Flux
    double length = params[3]; // Trail length
    double theta = params[4];  // Angle from +x-axis

    // Compute model image and chi-squared
    double chiSq = 0.0;
    // Loop is adapted from lsst::afw::detection::Psf::computeKernelImage()
    for (int yIndex = 0, yp = _bbox.getBeginY(); yIndex < _bbox.getHeight(); ++yIndex, ++yp) {
        ImageF::Array::Reference dataRow = _data[yIndex];
        ImageF::Array::Reference varRow = _variance[yIndex];
        for (int xIndex = 0, xp = _bbox.getBeginX(); xIndex < _bbox.getWidth(); ++xIndex, ++xp) {
            double model = _computeModel(xp,yp,xc,yc,flux,length,theta);
            double diff = dataRow[xIndex] - model;
            chiSq += diff*diff/varRow[xIndex];
        }
    }

    return chiSq;
}

std::vector<double> VeresModel::gradient(std::vector<double> const& params) const {

    double xc = params[0];     // Centroid x
    double yc = params[1];     // Centroid y
    double flux = params[2];   // Flux
    double length = params[3]; // Trail length
    double theta = params[4];  // Angle from +x-axis

    // Compute gradients of the model and of chi-squared
    std::vector<double> gradChiSq = {0.0,0.0,0.0,0.0,0.0};
    for (int yIndex = 0, yp = _bbox.getBeginY(); yIndex < _bbox.getHeight(); ++yIndex, ++yp) {
        ImageF::Array::Reference dataRow = _data[yIndex];
        ImageF::Array::Reference varRow = _variance[yIndex];
        for (int xIndex = 0, xp = _bbox.getBeginX(); xIndex < _bbox.getWidth(); ++xIndex, ++xp) {
            double model = _computeModel(xp,yp,xc,yc,flux,length,theta);
            double gradDiff = -2.0 * (dataRow[xIndex] - model) / varRow[xIndex];
            std::array<double, 5> gradModel = _computeGradient(xp,yp,xc,yc,flux,length,theta);
            for (int k=0; k<5; ++k) {
                gradChiSq[k] += gradModel[k] * gradDiff;
            }
        }
    }
    return gradChiSq;
}

std::tuple<double, std::vector<double>> VeresModel::computeFluxWithGradient(std::vector<double> const& params) const {

    double xc = params[0];     // Centroid x
    double yc = params[1];     // Centroid y
    // double flux = params[2];   // Flux (in this case, always 1)
    double length = params[3]; // Trail length
    double theta = params[4];  // Angle from +x-axis

    // Compute the flux and gradient wrt the other model parameters
    double m2 = 0.0;  // sum_i model_i*model_i
    double md = 0.0;  // sum_i model_i*data_i
    std::vector<double> gradmd = {0.0, 0.0, 0.0, 0.0, 0.0};  // sum_i (gradModel_(i,k)*data_i)
    std::vector<double> gradmm = {0.0, 0.0, 0.0, 0.0, 0.0};  // sum_i (gradModel_(i,k)*model_i)
    for (int yIndex = 0, yp = _bbox.getBeginY(); yIndex < _bbox.getHeight(); ++yIndex, ++yp) {
        ImageF::Array::Reference dataRow = _data[yIndex];
        for (int xIndex = 0, xp = _bbox.getBeginX(); xIndex < _bbox.getWidth(); ++xIndex, ++xp) {
            double data = dataRow[xIndex];
            double model = _computeModel(xp, yp, xc, yc, 1.0, length, theta);
            std::array<double, 5> gradModel = _computeGradient(xp, yp, xc, yc, 1.0, length, theta);
            m2 += model*model;
            md += model*dataRow[xIndex];
            for (int k=0; k<5; ++k) {
                gradmd[k] += gradModel[k] * data;
                gradmm[k] += gradModel[k] * model;
            }
        }
    }
    double flux = md / m2;
    std::vector<double> gradFlux = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (int k=0; k<5; ++k) {
        gradFlux[k] = (gradmd[k] - 2.0*flux*gradmm[k]) / m2;
    }
    gradFlux[2] = 0.0;  // Make dfluxdflux = 0
    std::tuple<double, std::vector<double>> results = std::make_tuple(flux, gradFlux);
    return results;
}

std::shared_ptr<ImageF> VeresModel::computeModelImage(std::vector<double> const& params) const {
    double xc = params[0];     // Centroid x
    double yc = params[1];     // Centroid y
    double flux = params[2];   // Flux
    double length = params[3]; // Trail length
    double theta = params[4];  // Angle from +x-axis

    // Loop is adapted from lsst::afw::detection::GaussianPsf::doComputeKernelImage()
    std::shared_ptr<ImageF> image(new ImageF(_bbox));
    ImageF::Array array = image->getArray();
    for (int yIndex = 0, yp = _bbox.getBeginY(); yIndex < _bbox.getHeight(); ++yIndex, ++yp) {
        ImageF::Array::Reference row = array[yIndex];
        for (int xIndex = 0, xp = _bbox.getBeginX(); xIndex < _bbox.getWidth(); ++xIndex, ++xp) {
            row[xIndex] = _computeModel(xp,yp,xc,yc,flux,length,theta);
        }
    }
    return image;
}

double VeresModel::_computeModel(double x, double y, double xc, double yc,
                                 double flux, double length, double theta) const noexcept {
    double xp = (x-xc)*cos(theta) + (y-yc)*sin(theta);
    double yp = (x-xc)*sin(theta) - (y-yc)*cos(theta);
    double A = exp(-0.5 * yp*yp / (_sigma*_sigma));
    double B = erf((xp+length/2) / (sqrt(2.0) * _sigma));
    double C = erf((xp-length/2) / (sqrt(2.0) * _sigma));
    return flux * A * (B - C) / (length * 2 * sqrt(2.0 * geom::PI) * _sigma);
}

std::array<double, 5> VeresModel::_computeGradient(double x, double y, double xc, double yc,
                                  double flux, double length, double theta) const noexcept {
    double xp = (x-xc)*cos(theta) + (y-yc)*sin(theta);
    double yp = (x-xc)*sin(theta) - (y-yc)*cos(theta);

    // Duplicated quantities
    double flux2L = flux/(2.0*length);
    double ypSq = yp*yp;
    double sqrt2 = sqrt(2.0);
    double sqrt2Pi = sqrt(2.0*geom::PI);
    double sigmaSq = _sigma*_sigma;
    double sigmaSq8 = sigmaSq * 8.0;
    double eypSq =  exp(-ypSq/(2.0*sigmaSq));
    double lengthPlus = length+2.0*xp;
    double lengthMinus= length-2.0*xp;
    double erfPlus = erf(lengthPlus/(2.0*sqrt2*_sigma));
    double erfMinus = erf(lengthMinus/(2.0*sqrt2*_sigma));
    double expPlus = exp(-lengthPlus*lengthPlus/sigmaSq8);

    // Compute partials wrt the transformed coordinates
    double dfdxp = flux2L/(geom::PI*sigmaSq)*exp(-4.0*ypSq/sigmaSq8)*expPlus*
        (1.0 - exp(length*xp/sigmaSq));
    double dfdyp = -flux2L*yp/(sqrt2Pi*_sigma*sigmaSq)*eypSq*(erfMinus+erfPlus);

    // Use the chain rule to get partials wrt the centroid and rotation angle
    double dxpdxc = -cos(theta);
    double dxpdyc = -sin(theta);
    double dxpdTheta = -yp;
    double dypdxc = -sin(theta);
    double dypdyc = cos(theta);
    double dypdTheta = xp;
    double dfdxc = dfdxp*dxpdxc + dfdyp*dypdxc;
    double dfdyc = dfdxp*dxpdyc + dfdyp*dypdyc;
    double dfdTheta = dfdxp*dxpdTheta + dfdyp*dypdTheta;

    double dfdFlux = _computeModel(x,y,xc,yc,1.0,length,theta); // dfdFlux = f / flux

    double dfdLength = flux2L/(length*sqrt2Pi*_sigma)*eypSq*(length/(sqrt2Pi*_sigma)*
        (exp(-lengthMinus*lengthMinus/sigmaSq8)+expPlus) - erfMinus - erfPlus);

    std::array<double, 5> gradModel = {dfdxc, dfdyc, dfdFlux, dfdLength, dfdTheta};
    return gradModel;
}

}}}} // lsst::meas::extensions::trailedSources
