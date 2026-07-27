"""Tutorial step definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TutorialStep:
    id: str
    title: str
    body: str
    target: str | None = None
    before_show: str | None = None


FIRST_MOTION_TUTORIAL = [
    TutorialStep(
        id="welcome",
        title="First Motion Walkthrough",
        body=(
            "This guided tour follows the main GhostGUI workflow: select a "
            "frame, move it into an orange preview, commit a keyframe, generate a "
            "trajectory, and export the result."
        ),
        before_show="show_3d_view",
    ),
    TutorialStep(
        id="choose_model",
        title="Choose A Robot",
        body=(
            "The robot model controls the available joints, frames, geometry, "
            "and home pose. Open the Robot menu to choose the active model."
        ),
        target="appMenuBar",
        before_show="expand_setup",
    ),
    TutorialStep(
        id="select_frame",
        title="Select A Target Frame",
        body=(
            "Choose the body, site, or logical frame you want to edit. For a "
            "first motion, a hand target such as left_hand is usually easiest "
            "to see."
        ),
        target="targetFrameCombo",
        before_show="expand_target_pose",
    ),
    TutorialStep(
        id="move_target",
        title="Move The Target",
        body=(
            "Drag the 3D transform gizmo or adjust the pose controls. The "
            "orange robot shows the temporary IK preview before it is saved. "
            "Press T for translate, R for rotate, or E/Esc to cancel a drag."
        ),
        target="workflowToolbar",
        before_show="expand_end_effector_editor",
    ),
    TutorialStep(
        id="preview_path",
        title="Preview Path Checks Motion",
        body=(
            "Preview Path validates the path from the committed pose to the "
            "orange preview. It does not save the pose by itself."
        ),
        target="planPreviewButton",
        before_show="show_3d_view",
    ),
    TutorialStep(
        id="slice_pose",
        title="Commit Keyframe Saves The Pose",
        body=(
            "Commit Keyframe records the preview at the active time and advances "
            "the timeline by the keyframe interval."
        ),
        target="sliceButton",
        before_show="show_3d_view",
    ),
    TutorialStep(
        id="generate",
        title="Generate The Trajectory",
        body=(
            "After you have saved one or more keyframes, Generate samples the "
            "timeline into a robot trajectory for playback and export."
        ),
        target="quickGenerateButton",
        before_show="show_3d_view",
    ),
    TutorialStep(
        id="export",
        title="Export The Result",
        body=(
            "Use File > Export to save the current committed qpos or timed "
            "trajectory rows. Unsaved orange previews are not exported."
        ),
        target="appMenuBar",
        before_show="expand_setup",
    ),
]
