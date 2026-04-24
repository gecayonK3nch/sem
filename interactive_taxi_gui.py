from __future__ import annotations

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from autonomous_taxi_model import (
    ModelParams,
    calculate_minimum_safe_interval,
    simulate_emergency_braking,
)


class TaxiQtWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Autonomous Taxi Safety Interval - Qt Simulator")
        self.resize(1500, 900)

        self.trace = None
        self.frame_idx = 0
        self.playing = True
        self.play_time_s = 0.0
        self.last_tick_time_s: float | None = None

        self._build_ui()
        self._connect_events()
        self._recalculate()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._sync_timer_interval()
        self.timer.start()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        controls = self._build_controls_panel()
        layout.addWidget(controls, 0)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        plt.style.use("seaborn-v0_8-whitegrid")
        self.figure = Figure(figsize=(10, 7), facecolor="#f7f9fc")
        self.canvas = FigureCanvas(self.figure)

        gs = self.figure.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0], hspace=0.35, wspace=0.28)
        self.ax_speed = self.figure.add_subplot(gs[0, 0])
        self.ax_gap = self.figure.add_subplot(gs[1, 0])
        self.ax_scene = self.figure.add_subplot(gs[:, 1])

        self.speed_line_lead, = self.ax_speed.plot([], [], color="#0b84f3", linewidth=2.3, label="Лидер")
        self.speed_line_follow, = self.ax_speed.plot([], [], color="#f39c12", linewidth=2.3, label="Следующий")
        self.speed_cursor = self.ax_speed.axvline(0, color="#34495e", linestyle="--", linewidth=1.4)

        self.gap_line, = self.ax_gap.plot([], [], color="#2c3e50", linewidth=2.3, label="Интервал")
        self.gap_cursor = self.ax_gap.axvline(0, color="#34495e", linestyle="--", linewidth=1.4)
        self.ax_gap.axhline(0, color="#e74c3c", linestyle="--", linewidth=1.2, label="Столкновение")

        self.car_len = 4.6
        self.car_width = 1.9
        self.lead_car = self._build_car_artist("#0b84f3", "#083d77")
        self.follow_car = self._build_car_artist("#f39c12", "#9a5e00")
        self.ax_scene.axvspan(-1.8, 5.4, color="#e5e7eb", alpha=0.45)
        self.ax_scene.axvline(1.8, color="#475569", linewidth=1.0, linestyle=(0, (6, 5)))
        self.ax_scene.axvline(-1.8, color="#334155", linewidth=1.2)
        self.ax_scene.axvline(5.4, color="#334155", linewidth=1.2)

        right_layout.addWidget(self.canvas, 1)

        self.scene_status = QLabel()
        self.scene_status.setWordWrap(True)
        self.scene_status.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scene_status.setStyleSheet("font-size: 12px; color: #1f2937; background: #f8fafc; border: 1px solid #dbe2ea; border-radius: 8px; padding: 6px;")
        right_layout.addWidget(self.scene_status)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.summary.setStyleSheet("font-size: 13px; color: #1f2937; background: #ffffff; border: 1px solid #dbe2ea; border-radius: 8px; padding: 8px;")
        right_layout.addWidget(self.summary)

        layout.addWidget(right, 1)

        self.setStyleSheet(
            """
            QWidget { font-family: Segoe UI; font-size: 11pt; color: #1f2937; background: #f2f4f8; }
            QLabel { color: #1f2937; background: transparent; }
            QGroupBox {
                border: 1px solid #dbe2ea;
                border-radius: 8px;
                margin-top: 8px;
                font-weight: 600;
                color: #111827;
                background: #ffffff;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #111827; }
            QDoubleSpinBox, QSpinBox {
                background: #ffffff;
                color: #111827;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                min-width: 140px;
                padding: 3px 6px;
            }
            QDoubleSpinBox:focus, QSpinBox:focus { border: 1px solid #0d6efd; }
            QPushButton { background: #0d6efd; color: white; border-radius: 6px; padding: 6px 10px; }
            QPushButton:hover { background: #0b5ed7; }
            QPushButton#secondary { background: #198754; }
            QPushButton#secondary:hover { background: #157347; }
            QPushButton#warning { background: #ffc107; color: #222; }
            QPushButton#warning:hover { background: #ffca2c; }
            """
        )

    def _build_car_artist(self, body_color: str, roof_color: str) -> dict[str, FancyBboxPatch | Circle | Polygon]:
        body = FancyBboxPatch(
            (0.0, 0.0),
            self.car_width,
            self.car_len,
            boxstyle="round,pad=0.03,rounding_size=0.28",
            facecolor=body_color,
            edgecolor="#1f2937",
            linewidth=1.0,
            alpha=0.97,
        )
        roof = FancyBboxPatch(
            (0.0, 0.0),
            self.car_width * 0.58,
            self.car_len * 0.5,
            boxstyle="round,pad=0.01,rounding_size=0.18",
            facecolor=roof_color,
            edgecolor="none",
            alpha=0.9,
        )
        windshield = FancyBboxPatch(
            (0.0, 0.0),
            self.car_width * 0.42,
            self.car_len * 0.22,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            facecolor="#e0f2fe",
            edgecolor="none",
            alpha=0.85,
        )
        light_front = Circle((0.0, 0.0), radius=0.08, facecolor="#fef3c7", edgecolor="none", alpha=0.95)
        light_rear = Circle((0.0, 0.0), radius=0.08, facecolor="#fecaca", edgecolor="none", alpha=0.95)
        direction_marker = Polygon([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)], closed=True, facecolor="#ffffff", edgecolor="none", alpha=0.85)

        self.ax_scene.add_patch(body)
        self.ax_scene.add_patch(roof)
        self.ax_scene.add_patch(windshield)
        self.ax_scene.add_patch(light_front)
        self.ax_scene.add_patch(light_rear)
        self.ax_scene.add_patch(direction_marker)
        return {
            "body": body,
            "roof": roof,
            "windshield": windshield,
            "light_front": light_front,
            "light_rear": light_rear,
            "direction_marker": direction_marker,
        }

    def _set_car_pose(self, car: dict[str, FancyBboxPatch | Circle | Polygon], x_center: float, y_rear: float) -> None:
        x_left = x_center - self.car_width / 2.0
        y_bottom = y_rear

        body = car["body"]
        roof = car["roof"]
        windshield = car["windshield"]
        light_front = car["light_front"]
        light_rear = car["light_rear"]
        direction_marker = car["direction_marker"]

        body.set_x(x_left)
        body.set_y(y_bottom)

        roof_w = self.car_width * 0.58
        roof_h = self.car_len * 0.5
        roof_x = x_center - roof_w / 2.0
        roof_y = y_bottom + (self.car_len - roof_h) * 0.5
        roof.set_x(roof_x)
        roof.set_y(roof_y)

        ws_w = self.car_width * 0.42
        ws_h = self.car_len * 0.22
        ws_x = x_center - ws_w / 2.0
        ws_y = y_bottom + self.car_len - ws_h - 0.28
        windshield.set_x(ws_x)
        windshield.set_y(ws_y)

        light_front.center = (x_center, y_bottom + self.car_len - 0.22)
        light_rear.center = (x_center, y_bottom + 0.22)
        direction_marker.set_xy(
            [
                (x_center, y_bottom + self.car_len - 0.36),
                (x_center + 0.16, y_bottom + self.car_len - 0.86),
                (x_center - 0.16, y_bottom + self.car_len - 0.86),
            ]
        )

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(420)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(10)

        p_group = QGroupBox("Параметры модели")
        form = QFormLayout(p_group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.speed = self._double(40.0, 10.0, 120.0, 1.0)
        self.brake_dist = self._double(10.0, 3.0, 40.0, 0.2)
        self.leader_speed = self._double(0.0, 0.0, 120.0, 1.0)
        self.leader_brake_dist = self._double(10.0, 0.1, 40.0, 0.2)
        self.react = self._double(0.2, 0.01, 2.0, 0.01)
        self.delay = self._double(0.5, 0.0, 3.0, 0.01)
        self.loss = self._double(0.01, 0.0, 0.5, 0.001)
        self.start_gap = self._double(18.0, 0.5, 120.0, 0.1)

        form.addRow("Скорость следующего (км/ч)", self.speed)
        form.addRow("Тормозной путь следующего (м)", self.brake_dist)
        form.addRow("Скорость лидера (км/ч)", self.leader_speed)
        form.addRow("Тормозной путь лидера (м)", self.leader_brake_dist)
        form.addRow("Время реакции (с)", self.react)
        form.addRow("Задержка связи (с)", self.delay)
        form.addRow("Потери пакетов (доля)", self.loss)
        form.addRow("Начальный интервал (м)", self.start_gap)

        sim_group = QGroupBox("Симуляция")
        sim_form = QFormLayout(sim_group)
        sim_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.duration = self._double(3.5, 1.0, 20.0, 0.5)
        self.dt = self._double(0.02, 0.005, 0.2, 0.005)
        self.playback = self._double(1.0, 0.5, 20.0, 0.5)

        sim_form.addRow("Длительность (с)", self.duration)
        sim_form.addRow("Шаг dt (с)", self.dt)
        sim_form.addRow("Скорость анимации x", self.playback)

        buttons = QGroupBox("Управление")
        b_layout = QGridLayout(buttons)

        self.btn_recalc = QPushButton("Пересчитать")
        self.btn_analytic = QPushButton("Взять расчетный S_min")
        self.btn_analytic.setObjectName("secondary")
        self.btn_play = QPushButton("Пауза")
        self.btn_play.setObjectName("warning")
        self.btn_restart = QPushButton("С начала")

        b_layout.addWidget(self.btn_recalc, 0, 0)
        b_layout.addWidget(self.btn_analytic, 0, 1)
        b_layout.addWidget(self.btn_play, 1, 0)
        b_layout.addWidget(self.btn_restart, 1, 1)

        panel_layout.addWidget(p_group)
        panel_layout.addWidget(sim_group)
        panel_layout.addWidget(buttons)
        panel_layout.addStretch(1)
        return panel

    @staticmethod
    def _double(value: float, minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setSingleStep(step)
        box.setValue(value)
        box.setDecimals(3)
        box.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        box.setKeyboardTracking(False)
        return box

    def _connect_events(self) -> None:
        self.btn_recalc.clicked.connect(self._recalculate)
        self.btn_analytic.clicked.connect(self._set_analytical_gap)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_restart.clicked.connect(self._restart)

        for w in [
            self.speed,
            self.brake_dist,
            self.leader_speed,
            self.leader_brake_dist,
            self.react,
            self.delay,
            self.loss,
            self.start_gap,
            self.duration,
            self.dt,
            self.playback,
        ]:
            w.editingFinished.connect(self._recalculate)

        self.playback.valueChanged.connect(self._sync_timer_interval)
        self.dt.valueChanged.connect(self._sync_timer_interval)

    def _params(self) -> ModelParams:
        return ModelParams(
            speed_kmh=self.speed.value(),
            braking_distance_m=self.brake_dist.value(),
            leader_speed_kmh=self.leader_speed.value(),
            leader_braking_distance_m=self.leader_brake_dist.value(),
            reaction_time_s=self.react.value(),
            communication_delay_s=self.delay.value(),
            packet_loss_probability=self.loss.value(),
        )

    def _set_analytical_gap(self) -> None:
        try:
            p = self._params()
            s_min = calculate_minimum_safe_interval(p).minimum_interval_m
            self.start_gap.setValue(max(self.start_gap.minimum(), min(self.start_gap.maximum(), s_min)))
        except ValueError as exc:
            self.summary.setText(f"Ошибка параметров: {exc}")

    def _sync_timer_interval(self) -> None:
        # Fixed 60 FPS timer for smooth rendering; simulation time advances by playback factor.
        interval_ms = 16
        if hasattr(self, "timer"):
            self.timer.setInterval(interval_ms)

    def _recalculate(self) -> None:
        self._sync_timer_interval()
        params = self._params()
        rng = np.random.default_rng()
        try:
            self.trace = simulate_emergency_braking(
                params,
                initial_gap_m=self.start_gap.value(),
                duration_s=self.duration.value(),
                dt_s=self.dt.value(),
                rng=rng,
                sudden_stop_lead=False,
            )
        except ValueError as exc:
            self.summary.setText(f"Ошибка параметров: {exc}")
            return
        self.frame_idx = 0
        self.play_time_s = 0.0
        self.last_tick_time_s = None
        self._redraw_static(params)
        self._update_at_time(0.0)
        self.canvas.draw_idle()

    def _redraw_static(self, params: ModelParams) -> None:
        result = calculate_minimum_safe_interval(params)
        t = self.trace.time_s

        self.speed_line_lead.set_data(t, self.trace.lead_speed_mps * 3.6)
        self.speed_line_follow.set_data(t, self.trace.follower_speed_mps * 3.6)
        self.gap_line.set_data(t, self.trace.gap_m)

        self.ax_speed.set_title("Скорости автомобилей", fontsize=12, fontweight="bold")
        self.ax_speed.set_xlabel("Время, с")
        self.ax_speed.set_ylabel("Скорость, км/ч")
        self.ax_speed.set_xlim(0, float(np.max(t)))
        self.ax_speed.set_ylim(0, max(8.0, params.speed_kmh * 1.2))
        self.ax_speed.grid(alpha=0.3)
        self.ax_speed.legend(loc="upper right")

        self.ax_gap.set_title("Интервал между машинами", fontsize=12, fontweight="bold")
        self.ax_gap.set_xlabel("Время, с")
        self.ax_gap.set_ylabel("Интервал, м")
        self.ax_gap.set_xlim(0, float(np.max(t)))
        self.ax_gap.set_ylim(float(np.min(self.trace.gap_m) - 1.0), float(np.max(self.trace.gap_m) + 1.0))
        self.ax_gap.grid(alpha=0.3)
        self.ax_gap.legend(loc="upper right")

        self.ax_scene.set_title("Анимация движения", fontsize=12, fontweight="bold")
        self.ax_scene.set_xlabel("Координата Y, м")
        self.ax_scene.set_ylabel("Координата X, м")
        self.ax_scene.set_xticks([-1.8, 0.0, 1.8, 3.6])
        self.ax_scene.set_xlim(-2.6, 6.2)
        self.ax_scene.set_aspect("equal", adjustable="box")
        self.ax_scene.set_ylim(
            float(np.min(self.trace.follower_position_m) - 2.0),
            float(np.max(self.trace.lead_position_m) + self.car_len + 2.0),
        )
        self.ax_scene.grid(alpha=0.15)

        status = "ОПАСНО: столкновение" if self.trace.collision else "Без столкновения"
        self.summary.setStyleSheet(
            f"font-size: 13px; color: #1f2937; background: #ffffff; border: 1px solid #dbe2ea; border-radius: 8px; padding: 8px;"
        )
        self.summary.setText(
            "\n".join(
                [
                    "Итоговый расчет:",
                    f"S_min = {result.minimum_interval_m:.2f} м ({result.minimum_interval_s:.2f} с)",
                    (
                        f"a_след = {result.max_deceleration_mps2:.2f} м/с^2, "
                        f"a_лид = {result.leader_max_deceleration_mps2:.2f} м/с^2, "
                        f"k_rel = {result.reliability_factor:.3f}"
                    ),
                    f"Начальный интервал = {self.start_gap.value():.2f} м, задержка оповещения = {self.trace.alert_delay_s:.2f} с",
                    f"Маневр объезда: {'АКТИВИРОВАН' if bool(np.any(self.trace.avoidance_active)) else 'не требуется'}",
                    f"Статус сценария: {status}",
                    f"Скорость проигрывания: x{self.playback.value():.1f}",
                ]
            )
        )

    def _update_at_time(self, t_now: float) -> None:
        t = self.trace.time_s
        t_now = max(0.0, min(t_now, float(t[-1])))

        lead_x = float(np.interp(t_now, t, self.trace.lead_position_m))
        follow_x = float(np.interp(t_now, t, self.trace.follower_position_m))
        lead_y = float(np.interp(t_now, t, self.trace.lead_lateral_m))
        follow_y = float(np.interp(t_now, t, self.trace.follower_lateral_m))
        gap = float(np.interp(t_now, t, self.trace.gap_m))
        lead_v_kmh = float(np.interp(t_now, t, self.trace.lead_speed_mps) * 3.6)
        follow_v_kmh = float(np.interp(t_now, t, self.trace.follower_speed_mps) * 3.6)
        avoid_now = bool(np.interp(t_now, t, self.trace.avoidance_active.astype(float)) > 0.5)

        self._set_car_pose(self.lead_car, lead_y, lead_x)
        self._set_car_pose(self.follow_car, follow_y, follow_x)
        self.speed_cursor.set_xdata([t_now, t_now])
        self.gap_cursor.set_xdata([t_now, t_now])

        overlap_x = abs(lead_x - follow_x) < self.car_len
        overlap_y = abs(lead_y - follow_y) < self.car_width
        collision_now = "ДА" if (overlap_x and overlap_y) else "НЕТ"
        self.scene_status.setText(
            f"t = {t_now:.2f} с\n"
            f"v_лид = {lead_v_kmh:.1f} км/ч\n"
            f"v_след = {follow_v_kmh:.1f} км/ч\n"
            f"интервал = {gap:.2f} м\n"
            f"объезд: {'да' if avoid_now else 'нет'}\n"
            f"столкновение сейчас: {collision_now}"
        )

    def _tick(self) -> None:
        if self.trace is None or not self.playing:
            return

        now = time.perf_counter()
        if self.last_tick_time_s is None:
            self.last_tick_time_s = now
            return

        dt_real = max(0.0, now - self.last_tick_time_s)
        self.last_tick_time_s = now

        self.play_time_s += dt_real * self.playback.value()
        t_end = float(self.trace.time_s[-1])
        if self.play_time_s > t_end:
            self.play_time_s = self.play_time_s % t_end

        self._update_at_time(self.play_time_s)
        self.canvas.draw_idle()

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.last_tick_time_s = None
        self.btn_play.setText("Старт" if not self.playing else "Пауза")

    def _restart(self) -> None:
        self.frame_idx = 0
        self.play_time_s = 0.0
        self.last_tick_time_s = None
        if self.trace is not None:
            self._update_at_time(0.0)
            self.canvas.draw_idle()


def main() -> None:
    app = QApplication(sys.argv)
    win = TaxiQtWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()