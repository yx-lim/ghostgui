#pragma once

#include <string>
#include <vector>

/*
robot_sim.hpp

Purpose:
    Backend for first task-space to joint-space conversion.

Current version:
    - Takes pelvis TargetFrame from GUI
    - Maps pelvis pose directly to floating base pose
    - Keeps all joints fixed
    - Exports one CSV row per timestep

This is the first version of:
    x(t) -> q(t)

where:
    x(t) = GUI task-space pelvis trajectory
    q(t) = base pose + joint angles
*/


// ============================================================
// Target frame from GUI
// ============================================================

struct TargetFrame {
    double time = 0.0;

    std::string phase = "default";
    std::string frame_name = "pelvis";

    double x = 0.0;
    double y = 0.0;
    double z = 0.9;

    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
};


// ============================================================
// Robot configuration q(t)
// ============================================================

struct RobotConfiguration {
    double time = 0.0;

    double base_x = 0.0;
    double base_y = 0.0;
    double base_z = 0.9;

    double base_qw = 1.0;
    double base_qx = 0.0;
    double base_qy = 0.0;
    double base_qz = 0.0;

    std::vector<std::string> joint_names;
    std::vector<double> joint_positions;

    double ik_error = 0.0;
    bool success = true;

    std::string status = "Not solved yet";
};


// ============================================================
// Robot backend
// ============================================================

class RobotBackend {
public:
    RobotBackend();

    void reset();

    /*
    Set joint names used in CSV export.

    Use this to match your lab's exact joint-name order.
    */
    void set_joint_names(const std::vector<std::string>& joint_names);

    /*
    Set fixed default joint positions.

    For the first version, these joint values are copied into every
    timestep. Later, IK will replace these values.
    */
    void set_default_joint_positions(
        const std::vector<double>& joint_positions
    );

    /*
    Solve one TargetFrame into one RobotConfiguration.
    Currently only maps pelvis/base target to floating base pose.
    */
    RobotConfiguration solve_frame(
        const TargetFrame& target,
        const RobotConfiguration& q_initial
    );

    /*
    Solve a full task-space trajectory into joint-space trajectory.
    */
    std::vector<RobotConfiguration> solve_trajectory(
        const std::vector<TargetFrame>& trajectory
    );

    /*
    Export last solved trajectory to CSV.
    */
    void export_last_solution_csv(const std::string& csv_path) const;

    /*
    Return last solved joint-space trajectory.
    */
    std::vector<RobotConfiguration> last_solution() const;

private:
    std::vector<std::string> joint_names_;
    std::vector<double> default_joint_positions_;
    std::vector<RobotConfiguration> last_solution_;

    RobotConfiguration make_default_configuration() const;
};