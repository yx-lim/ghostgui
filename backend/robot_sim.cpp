#include "robot_sim.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <sstream>

/*
robot_sim.cpp

Current behavior:
    GUI pelvis TargetFrame
        -> base_x, base_y, base_z
        -> base quaternion
        -> fixed joint values
        -> CSV row

This is intentionally simple.
The later IK implementation should replace solve_frame().
*/


// ============================================================
// Helper quaternion struct
// ============================================================

struct Quaternion {
    double w = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};


// ============================================================
// Convert roll, pitch, yaw to quaternion
// ============================================================

static Quaternion rpy_to_quaternion(
    double roll,
    double pitch,
    double yaw
) {
    /*
    Assumes:
        roll  = rotation about x-axis
        pitch = rotation about y-axis
        yaw   = rotation about z-axis

    Returns:
        quaternion in qw, qx, qy, qz order
    */

    const double cr = std::cos(roll * 0.5);
    const double sr = std::sin(roll * 0.5);

    const double cp = std::cos(pitch * 0.5);
    const double sp = std::sin(pitch * 0.5);

    const double cy = std::cos(yaw * 0.5);
    const double sy = std::sin(yaw * 0.5);

    Quaternion q;

    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q;
}


// ============================================================
// Constructor / reset
// ============================================================

RobotBackend::RobotBackend() {
    /*
    Replace this list with your lab's exact joint names and order.

    The order here should match the order expected by the lab CSV reader.
    */

    joint_names_ = {
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",

        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",

        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",

        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",

        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint"
    };

    default_joint_positions_.assign(joint_names_.size(), 0.0);

    reset();
}


void RobotBackend::reset() {
    last_solution_.clear();
}


// ============================================================
// Set joint names / default joint values
// ============================================================

void RobotBackend::set_joint_names(
    const std::vector<std::string>& joint_names
) {
    joint_names_ = joint_names;

    if (default_joint_positions_.size() != joint_names_.size()) {
        default_joint_positions_.assign(joint_names_.size(), 0.0);
    }
}


void RobotBackend::set_default_joint_positions(
    const std::vector<double>& joint_positions
) {
    if (joint_positions.size() != joint_names_.size()) {
        throw std::runtime_error(
            "set_default_joint_positions: number of joint positions "
            "must match number of joint names."
        );
    }

    default_joint_positions_ = joint_positions;
}


// ============================================================
// Make default q
// ============================================================

RobotConfiguration RobotBackend::make_default_configuration() const {
    RobotConfiguration q;

    q.time = 0.0;

    q.base_x = 0.0;
    q.base_y = 0.0;
    q.base_z = 0.9;

    q.base_qw = 1.0;
    q.base_qx = 0.0;
    q.base_qy = 0.0;
    q.base_qz = 0.0;

    q.joint_names = joint_names_;
    q.joint_positions = default_joint_positions_;

    q.ik_error = 0.0;
    q.success = true;
    q.status = "Default configuration";

    return q;
}


// ============================================================
// Solve one frame
// ============================================================

RobotConfiguration RobotBackend::solve_frame(
    const TargetFrame& target,
    const RobotConfiguration& q_initial
) {
    /*
    First version:
        Only pelvis/base target is mapped.

    Assumption:
        pelvis frame == floating base frame

    If your robot model has a fixed transform between base and pelvis,
    replace the direct assignment with:

        T_world_base = T_world_pelvis_target * inverse(T_base_pelvis)
    */

    RobotConfiguration q = q_initial;

    q.time = target.time;
    q.joint_names = joint_names_;

    if (q.joint_positions.size() != joint_names_.size()) {
        q.joint_positions = default_joint_positions_;
    }

    // ------------------------------------------------------------
    // Basic validation
    // ------------------------------------------------------------

    if (target.time < 0.0) {
        q.success = false;
        q.ik_error = std::abs(target.time);
        q.status = "Failed: target time is negative";
        return q;
    }

    if (target.z < 0.0) {
        q.success = false;
        q.ik_error = std::abs(target.z);
        q.status = "Failed: pelvis target is below ground";
        return q;
    }

    // ------------------------------------------------------------
    // Pelvis target -> base pose
    // ------------------------------------------------------------

    if (
        target.frame_name == "pelvis" ||
        target.frame_name == "base" ||
        target.frame_name == "root"
    ) {
        q.base_x = target.x;
        q.base_y = target.y;
        q.base_z = target.z;

        Quaternion quat = rpy_to_quaternion(
            target.roll,
            target.pitch,
            target.yaw
        );

        q.base_qw = quat.w;
        q.base_qx = quat.x;
        q.base_qy = quat.y;
        q.base_qz = quat.z;

        q.ik_error = 0.0;
        q.success = true;

        std::ostringstream oss;
        oss << "Mapped "
            << target.frame_name
            << " target directly to floating base pose"
            << " | phase=" << target.phase
            << " | t=" << target.time << "s";

        q.status = oss.str();

        return q;
    }

    // ------------------------------------------------------------
    // Other target frames are ignored for this first version
    // ------------------------------------------------------------

    q.success = true;
    q.ik_error = 0.0;

    std::ostringstream oss;
    oss << "Ignored non-pelvis target frame '"
        << target.frame_name
        << "' in pelvis-only mapper";

    q.status = oss.str();

    return q;
}


// ============================================================
// Solve full trajectory
// ============================================================

std::vector<RobotConfiguration> RobotBackend::solve_trajectory(
    const std::vector<TargetFrame>& trajectory
) {
    last_solution_.clear();

    std::vector<TargetFrame> sorted_trajectory = trajectory;

    std::sort(
        sorted_trajectory.begin(),
        sorted_trajectory.end(),
        [](const TargetFrame& a, const TargetFrame& b) {
            return a.time < b.time;
        }
    );

    RobotConfiguration q_prev = make_default_configuration();

    for (const TargetFrame& target : sorted_trajectory) {
        RobotConfiguration q = solve_frame(target, q_prev);

        /*
        Joints stay fixed because solve_frame copies q_initial.joint_positions.

        Later:
            replace this with IK-generated joint positions.
        */

        last_solution_.push_back(q);
        q_prev = q;
    }

    return last_solution_;
}


// ============================================================
// Return last solution
// ============================================================

std::vector<RobotConfiguration> RobotBackend::last_solution() const {
    return last_solution_;
}


// ============================================================
// Export CSV
// ============================================================

void RobotBackend::export_last_solution_csv(
    const std::string& csv_path
) const {
    if (last_solution_.empty()) {
        throw std::runtime_error(
            "export_last_solution_csv: no solved trajectory to export."
        );
    }

    std::ofstream file(csv_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "export_last_solution_csv: failed to open output CSV file."
        );
    }

    /*
    Header format:

        time,
        base_x, base_y, base_z,
        base_qw, base_qx, base_qy, base_qz,
        joint_1, joint_2, ..., joint_N

    This should be close to your lab's q(t) format.
    */

    file << "time";
    file << ",base_x,base_y,base_z";
    file << ",base_qw,base_qx,base_qy,base_qz";

    for (const std::string& name : joint_names_) {
        file << "," << name;
    }

    file << "\n";

    file << std::fixed << std::setprecision(8);

    for (const RobotConfiguration& q : last_solution_) {
        file << q.time;

        file << "," << q.base_x;
        file << "," << q.base_y;
        file << "," << q.base_z;

        file << "," << q.base_qw;
        file << "," << q.base_qx;
        file << "," << q.base_qy;
        file << "," << q.base_qz;

        for (double joint_value : q.joint_positions) {
            file << "," << joint_value;
        }

        file << "\n";
    }
}