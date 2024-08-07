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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "lsst/geom.h"
#include "lsst/afw/image.h"
#include "lsst/meas/extensions/trailedSources/VeresModel.h"
#include "lsst/cpputils/python.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace meas {
namespace extensions {
namespace trailedSources {

void wrapVeresModel(cpputils::python::WrapperCollection& wrappers) {

    wrappers.addSignatureDependency("lsst.geom");
    wrappers.addSignatureDependency("lsst.afw.image");

    wrappers.wrapType(
        py::class_<VeresModel, std::shared_ptr<VeresModel>>(wrappers.module, "VeresModel"),
        [](auto & mod, auto & cls) {
            cls.def(py::init<afw::image::Exposure<float> const&>(), "data"_a);
            cls.def("__call__", &VeresModel::operator(), py::is_operator(), "params"_a);
            cls.def("gradient", &VeresModel::gradient, "params"_a);
            cls.def("computeFluxWithGradient", &VeresModel::computeFluxWithGradient, "params"_a);
            cls.def("computeModelImage", &VeresModel::computeModelImage, "params"_a);
            cls.def_property_readonly("sigma", &VeresModel::getSigma);
    });
}
}}}} // lsst::meas::extensions::trailedSources