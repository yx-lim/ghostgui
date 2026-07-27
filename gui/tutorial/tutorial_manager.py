"""Tutorial manager that coordinates steps, highlighting, and navigation."""

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QTimer
from PySide6.QtWidgets import QWidget

from .tutorial_card import TutorialCard
from .tutorial_overlay import TutorialOverlay
from .tutorial_steps import FIRST_MOTION_TUTORIAL


class TutorialManager(QObject):
    """Owns the first-motion guided tutorial for a main window."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.steps = []
        self.current_index = 0
        self.overlay = None
        self.card = None
        self.active = False

    def start_first_motion(self):
        self.start(FIRST_MOTION_TUTORIAL)

    def start(self, steps):
        if not steps:
            return
        self.steps = list(steps)
        self.current_index = 0
        self.active = True
        self._ensure_widgets()
        self.main_window.installEventFilter(self)
        self.overlay.setGeometry(self.main_window.rect())
        self.overlay.show()
        self.card.show()
        self.show_current_step()

    def stop(self):
        if not self.active:
            return
        self.active = False
        self.main_window.removeEventFilter(self)
        if self.overlay is not None:
            self.overlay.hide()
            self.overlay.set_target_rect(QRect())
        if self.card is not None:
            self.card.hide()

    def next_step(self):
        if not self.active:
            return
        if self.current_index >= len(self.steps) - 1:
            self.stop()
            return
        self.current_index += 1
        self.show_current_step()

    def previous_step(self):
        if not self.active or self.current_index <= 0:
            return
        self.current_index -= 1
        self.show_current_step()

    def show_current_step(self):
        if not self.active:
            return
        step = self.steps[self.current_index]
        self._run_before_show(step.before_show)
        self.card.set_step(step, self.current_index, len(self.steps))
        QTimer.singleShot(0, self.reposition)

    def reposition(self):
        if not self.active:
            return
        self.overlay.setGeometry(self.main_window.rect())
        target_rect = self._target_rect(self.steps[self.current_index].target)
        self.overlay.set_target_rect(target_rect)
        self._position_card(target_rect)
        self.overlay.raise_()
        self.overlay.stackUnder(self.card)
        self.card.raise_()

    def eventFilter(self, watched, event):
        if watched is self.main_window and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
        ):
            QTimer.singleShot(0, self.reposition)
        return super().eventFilter(watched, event)

    def _ensure_widgets(self):
        if self.overlay is None:
            self.overlay = TutorialOverlay(self.main_window)
        if self.card is None:
            self.card = TutorialCard(self.main_window)
            self.card.back_requested.connect(self.previous_step)
            self.card.next_requested.connect(self.next_step)
            self.card.skip_requested.connect(self.stop)

    def _target_rect(self, object_name):
        if not object_name:
            return QRect()
        target = self.main_window.findChild(QWidget, object_name)
        if target is None or not target.isVisible():
            return QRect()
        top_left = target.mapTo(self.main_window, QPoint(0, 0))
        return QRect(top_left, target.size())

    def _position_card(self, target_rect):
        self.card.adjustSize()
        margin = 14
        window_rect = self.main_window.rect()
        card_size = self.card.size()

        if target_rect.isNull():
            x = (window_rect.width() - card_size.width()) // 2
            y = (window_rect.height() - card_size.height()) // 2
        else:
            right_x = target_rect.right() + margin
            left_x = target_rect.left() - card_size.width() - margin
            below_y = target_rect.bottom() + margin
            above_y = target_rect.top() - card_size.height() - margin

            if right_x + card_size.width() <= window_rect.right() - margin:
                x = right_x
                y = target_rect.top()
            elif left_x >= margin:
                x = left_x
                y = target_rect.top()
            else:
                x = target_rect.left()
                y = below_y if below_y + card_size.height() <= window_rect.bottom() else above_y

        x = max(margin, min(x, window_rect.width() - card_size.width() - margin))
        y = max(margin, min(y, window_rect.height() - card_size.height() - margin))
        self.card.move(x, y)

    def _run_before_show(self, hook):
        if hook == "show_3d_view":
            self.main_window.viewer_tabs.setCurrentWidget(
                self.main_window.viewer_3d_stack
            )
            self.main_window.update_editor_context()
        elif hook == "expand_setup":
            self._run_before_show("show_3d_view")
            toolbar = getattr(self.main_window, "app_toolbar", None)
            if toolbar is not None:
                toolbar.show()
                toolbar.raise_()
        elif hook == "expand_target_pose":
            self._run_before_show("show_3d_view")
            self._set_section_expanded("Target", True)
        elif hook == "expand_end_effector_editor":
            self._run_before_show("expand_target_pose")
            self._set_section_expanded("Editing Mode", True)
            self.main_window.controls.set_editing_mode("end_effector")
        elif hook == "expand_time_slices":
            self._run_before_show("show_3d_view")
            self._set_section_expanded("Planning", True)

    def _set_section_expanded(self, title, expanded):
        for sidebar in (self.main_window.left_sidebar_content, self.main_window.right_sidebar_content):
            for section in sidebar.sections:
                if section.title == title:
                    section.set_expanded(expanded)
                    return
