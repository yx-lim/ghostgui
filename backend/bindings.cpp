#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "robot_sim.hpp"

#include <stdexcept>
#include <string>
#include <vector>

/*
bindings.cpp

Purpose:
    Exposes the pelvis-target to base-pose backend to Python.

Python usage:
    backend = robot_backend.RobotBackend()
    solution = backend.solve_trajectory(trajectory)
    backend.export_last_solution_csv("output.csv")
*/

namespace py = pybind11;


// ============================================================
// Helper functions
// ============================================================

namespace {

template <typename T>
T get_dict_value_or(
    const py::dict& dict,
    const char* key,
    const T& default_value
) {
    py::str py_key(key);

    if (dict.contains(py_key)) {
        return dict[py_key].cast<T>();
    }

    return default_value;
}


template <typename T>
T get_attr_value_or(
    const py::object& obj,
    const char* key,
    const T& default_value
) {
    if (py::hasattr(obj, key)) {
        return py::getattr(obj, key).cast<T>();
    }

    return default_value;
}


TargetFrame target_frame_from_dict(const py::dict& dict) {
    TargetFrame frame;

    frame.time = get_dict_value_or<double>(dict, "time", 0.0);

    frame.phase = get_dict_value_or<std::string>(dict, "phase", "default");
    frame.frame_name = get_dict_value_or<std::string>(
        dict,
        "frame_name",
        "pelvis"
    );

    frame.x = get_dict_value_or<double>(dict, "x", 0.0);
    frame.y = get_dict_value_or<double>(dict, "y", 0.0);
    frame.z = get_dict_value_or<double>(dict, "z", 0.9);

    frame.roll = get_dict_value_or<double>(dict, "roll", 0.0);
    frame.pitch = get_dict_value_or<double>(dict, "pitch", 0.0);
    frame.yaw = get_dict_value_or<double>(dict, "yaw", 0.0);

    return frame;
}


TargetFrame target_frame_from_python_object(const py::object& obj) {
    /*
    Supports:
        - C++ robot_backend.TargetFrame
        - Python dataclass TargetFrame
        - dictionary from trajectory.as_list()
    */

    if (py::isinstance<py::dict>(obj)) {
        return target_frame_from_dict(obj.cast<py::dict>());
    }

    try {
        return obj.cast<TargetFrame>();
    } catch (const py::cast_error&) {
        // Fall through to attribute conversion.
    }

    TargetFrame frame;

    frame.time = get_attr_value_or<double>(obj, "time", 0.0);

    frame.phase = get_attr_value_or<std::string>(obj, "phase", "default");
    frame.frame_name = get_attr_value_or<std::string>(
        obj,
        "frame_name",
        "pelvis"
    );

    frame.x = get_attr_value_or<double>(obj, "x", 0.0);
    frame.y = get_attr_value_or<double>(obj, "y", 0.0);
    frame.z = get_attr_value_or<double>(obj, "z", 0.9);

    frame.roll = get_attr_value_or<double>(obj, "roll", 0.0);
    frame.pitch = get_attr_value_or<double>(obj, "pitch", 0.0);
    frame.yaw = get_attr_value_or<double>(obj, "yaw", 0.0);

    return frame;
}


std::vector<TargetFrame> trajectory_from_python_object(
    const py::object& obj
) {
    /*
    Supports:
        trajectory
        trajectory.frames
        trajectory.as_list()
        list[TargetFrame]
        list[dict]
    */

    py::object sequence_obj = obj;

    if (py::hasattr(obj, "as_list")) {
        sequence_obj = py::getattr(obj, "as_list")();
    } else if (py::hasattr(obj, "frames")) {
        sequence_obj = py::getattr(obj, "frames");
    }

    if (
        !py::isinstance<py::list>(sequence_obj) &&
        !py::isinstance<py::tuple>(sequence_obj)
    ) {
        throw std::runtime_error(
            "solve_trajectory expected a Trajectory object, "
            "a list of TargetFrame objects, or a list of dictionaries."
        );
    }

    std::vector<TargetFrame> trajectory;

    for (py::handle item : sequence_obj) {
        py::object item_obj = py::reinterpret_borrow<py::object>(item);
        trajectory.push_back(target_frame_from_python_object(item_obj));
    }

    return trajectory;
}

}  // namespace


// ============================================================
// Module definition
// ============================================================

PYBIND11_MODULE(robot_backend, m) {
    m.doc() = "Pelvis-target to base-pose backend for robot trajectory GUI";

    // ------------------------------------------------------------
    // TargetFrame
    // ------------------------------------------------------------

    py::class_<TargetFrame>(m, "TargetFrame")
        .def(py::init<>())

        .def_readwrite("time", &TargetFrame::time)

        .def_readwrite("phase", &TargetFrame::phase)
        .def_readwrite("frame_name", &TargetFrame::frame_name)

        .def_readwrite("x", &TargetFrame::x)
        .def_readwrite("y", &TargetFrame::y)
        .def_readwrite("z", &TargetFrame::z)

        .def_readwrite("roll", &TargetFrame::roll)
        .def_readwrite("pitch", &TargetFrame::pitch)
        .def_readwrite("yaw", &TargetFrame::yaw);

    // ------------------------------------------------------------
    // RobotConfiguration
    // ------------------------------------------------------------

    py::class_<RobotConfiguration>(m, "RobotConfiguration")
        .def(py::init<>())

        .def_readwrite("time", &RobotConfiguration::time)

        .def_readwrite("base_x", &RobotConfiguration::base_x)
        .def_readwrite("base_y", &RobotConfiguration::base_y)
        .def_readwrite("base_z", &RobotConfiguration::base_z)

        .def_readwrite("base_qw", &RobotConfiguration::base_qw)
        .def_readwrite("base_qx", &RobotConfiguration::base_qx)
        .def_readwrite("base_qy", &RobotConfiguration::base_qy)
        .def_readwrite("base_qz", &RobotConfiguration::base_qz)

        .def_readwrite("joint_names", &RobotConfiguration::joint_names)
        .def_readwrite("joint_positions", &RobotConfiguration::joint_positions)

        .def_readwrite("ik_error", &RobotConfiguration::ik_error)
        .def_readwrite("success", &RobotConfiguration::success)
        .def_readwrite("status", &RobotConfiguration::status);

    // ------------------------------------------------------------
    // RobotBackend
    // ------------------------------------------------------------

    py::class_<RobotBackend>(m, "RobotBackend")
        .def(py::init<>())

        .def("reset", &RobotBackend::reset)

        .def(
            "set_joint_names",
            &RobotBackend::set_joint_names,
            py::arg("joint_names")
        )

        .def(
            "set_default_joint_positions",
            &RobotBackend::set_default_joint_positions,
            py::arg("joint_positions")
        )

        .def(
            "solve_frame",
            &RobotBackend::solve_frame,
            py::arg("target"),
            py::arg("q_initial")
        )

        .def(
            "solve_trajectory",
            [](RobotBackend& backend, const py::object& trajectory_obj) {
                std::vector<TargetFrame> trajectory =
                    trajectory_from_python_object(trajectory_obj);

                return backend.solve_trajectory(trajectory);
            },
            py::arg("trajectory")
        )

        .def(
            "export_last_solution_csv",
            &RobotBackend::export_last_solution_csv,
            py::arg("csv_path")
        )

        .def(
            "last_solution",
            &RobotBackend::last_solution
        );
}