#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include "model.h"
#include <random>
#include <gen/utils/randutils.h>
#include <gen/still/particle_filter.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<Trace*>);

PYBIND11_MODULE(tracker, m) {
    m.doc() = "pybind11 tracker plugin"; // optional module docstring

    // TODO add name tags & keyword args
    py::class_<State>(m, "State")
            .def(py::init<double, double, double, double, double>())
            .def_readwrite("x", &State::x)
            .def_readwrite("y", &State::y)
            .def_readwrite("vx", &State::vx)
            .def_readwrite("vy", &State::vy)
            .def_readwrite("measured_bearing", &State::measured_bearing);

    py::class_<NextTimeStepObservations>(m, "NextTimeStepObservations")
            .def(py::init<double>())
            .def_readwrite("measured_bearing", &NextTimeStepObservations::measured_bearing);

    py::class_<EmptyConstraints>(m, "EmptyConstraints")
            .def(py::init<>());

    py::class_<ExtendByOneTimeStep>(m, "ExtendByOneTimeStep")
            .def(py::init<>());

    // TODO add name tags & keyword args
    py::class_<Parameters>(m, "Parameters")
            .def(py::init<double, double, double, double, double, double, double, double, double, double>())
            .def_readwrite("measurement_noise", &Parameters::measurement_noise)
            .def_readwrite("velocity_stdev", &Parameters::velocity_stdev)
            .def_readwrite("init_x_prior_mean", &Parameters::init_x_prior_mean)
            .def_readwrite("init_x_prior_stdev", &Parameters::init_x_prior_stdev)
            .def_readwrite("init_y_prior_mean", &Parameters::init_y_prior_mean)
            .def_readwrite("init_y_prior_stdev", &Parameters::init_y_prior_stdev)
            .def_readwrite("init_vx_prior_mean", &Parameters::init_vx_prior_mean)
            .def_readwrite("init_vx_prior_stdev", &Parameters::init_vx_prior_stdev)
            .def_readwrite("init_vy_prior_mean", &Parameters::init_vy_prior_mean)
            .def_readwrite("init_vy_prior_stdev", &Parameters::init_vy_prior_stdev);

    py::class_<RetDiff>(m, "RetDiff")
            .def(py::init<>());

    py::class_<Trace>(m, "Trace")
            .def("state", &Trace::state);

    py::class_<std::mt19937>(m, "mt19937")
            .def(py::init<size_t>());

    py::class_<Model>(m, "Model")
            .def(py::init<>())
            .def(py::init<size_t>())
            .def("simulate", &Model::simulate<std::mt19937>)
            .def("generate", &Model::generate<std::mt19937>);

    typedef gen::still::smc::ParticleSystem<Trace,std::mt19937> MyParticleSystem;

    py::bind_vector<std::vector<Trace*>>(m, "TraceVector");

    py::class_<MyParticleSystem>(m, "MyParticleSystem")
            .def(py::init<size_t, std::mt19937&>())
            .def("init_step", &MyParticleSystem::init_step<Model,Parameters,NextTimeStepObservations>)
            .def("step", &MyParticleSystem::step<ExtendByOneTimeStep, NextTimeStepObservations&>)
            .def("effective_sample_size", &MyParticleSystem::effective_sample_size)
            .def("resample", &MyParticleSystem::resample)
            .def("traces", &MyParticleSystem::traces);
