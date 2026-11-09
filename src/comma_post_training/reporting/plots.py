"""Export measured curves with Matplotlib's noninteractive backend."""

from pathlib import Path

import numpy as np

COLORS = {
    "expert": "#18794e",
    "teacher": "#546ee5",
    "adapted": "#c24e10",
    "commanded": "#bc5a13",
    "executed": "#824fbd",
}


class Figures:
    def __init__(self, root: Path):
        try:
            import matplotlib
        except ImportError as error:
            raise ValueError("install the visualization extra or matplotlib>=3.10,<4") from error
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        self.plt = plt
        self.root, self.items = root, []
        root.mkdir()
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "svg.fonttype": "none",
                "text.parse_math": False,
            }
        )

    def save(self, figure, name, title, caption):
        figure.suptitle(title, fontsize=13)
        figure.tight_layout()
        paths = []
        for extension in ("png", "svg"):
            path = self.root / f"{name}.{extension}"
            figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            paths.append(path)
        self.plt.close(figure)
        self.items.append({"name": name, "title": title, "caption": caption, "paths": paths})

    @staticmethod
    def axes_style(axes):
        for axis in np.asarray(axes).reshape(-1):
            axis.grid(alpha=0.2)

    def benchmark(self, report, name):
        rows = report["cases"]
        labels = [row["case"]["id"] for row in rows]
        x = np.arange(len(rows))
        fig, axes = self.plt.subplots(2, 2, figsize=(max(12, len(rows) * 0.6), 8))
        self.axes_style(axes)
        metrics = (
            ("controlled_return", "Controlled return", False),
            ("cross_track_mae_m", "Cross-track MAE (m)", True),
            ("speed_mae_mps", "Speed MAE (m/s)", True),
            ("recovery_time_s", "Recovery confirmation (s); red = not recovered", True),
        )
        for axis, (key, title, reached) in zip(axes.flat, metrics, strict=True):
            values = [(row["reached_state"] if reached else row).get(key) for row in rows]
            axis.bar(x, [v if v is not None else np.nan for v in values], color="#175d85")
            failed = [
                i
                for i, row in enumerate(rows)
                if row["terminated"]
                or key == "recovery_time_s"
                and row["case"]["group"] == "recovery"
                and row["reached_state"]["recovered"] is not True
            ]
            if failed:
                axis.scatter(failed, [0] * len(failed), marker="x", color="#bc2115", zorder=5)
            axis.set(title=title, xticks=x, xticklabels=labels)
            axis.tick_params(axis="x", rotation=65, labelsize=8)
        self.save(
            fig,
            name,
            f"{report['policy']} · {report['split']} · fixed recovery suite",
            "Measured episodes including failures; controlled metrics exclude expert warmup. "
            "Synthetic engineering scenes do not establish real-driving performance.",
        )

    def episode(self, values, name, title):
        fig, ax = self.plt.subplots(2, 2, figsize=(12, 7))
        self.axes_style(ax)
        times = np.asarray(values["time_s"])
        if values.get("xy") is not None:
            xy = np.asarray(values["xy"])
            ax[0, 0].plot(xy[:, 0], xy[:, 1], color="#175d85")
            ax[0, 0].scatter(*xy[0], marker="o", label="start", color="#18794e")
            ax[0, 0].scatter(*xy[-1], marker="x", label="last observation", color="#c24e10")
            ax[0, 0].set_aspect("equal", adjustable="datalim")
            ax[0, 0].legend()
        else:
            ax[0, 0].text(0.5, 0.5, "No pose recorded", ha="center")
        ax[0, 0].set(xlabel="World x (m)", ylabel="World y (m)", title="Observed trajectory")
        state = np.asarray(values["state"])
        ax[0, 1].plot(times, state[:, 0], color="#175d85")
        state_names = values.get("state_features", [])
        ax[0, 1].set(
            xlabel="Time (s)",
            ylabel=state_names[0] if state_names else "state[0]",
            title="Vehicle state",
        )
        for dimension, axis in enumerate(ax[1]):
            for key, color in COLORS.items():
                if key in values:
                    axis.plot(
                        times,
                        np.asarray(values[key])[:, dimension],
                        label=key,
                        color=color,
                        linewidth=1.3,
                    )
            valid = values.get("teacher_valid")
            if valid is not None:
                axis.fill_between(
                    times,
                    0,
                    1,
                    where=~np.asarray(valid, dtype=bool),
                    transform=axis.get_xaxis_transform(),
                    color="#ccc",
                    alpha=0.25,
                    label="Teacher warmup/invalid",
                )
            axis.set(
                xlabel="Time (s)",
                ylabel="Acceleration (m/s²)",
                title=("Lateral" if dimension == 0 else "Longitudinal"),
            )
            axis.legend(fontsize=8)
        self.save(
            fig, name, title, "观测和动作按源时间戳展示；灰色区域为 Teacher warmup 或无效标签。"
        )

    def training(self, run, name):
        metrics, events = run["metrics"], run["events"]
        fig, ax = self.plt.subplots(2, 1, figsize=(11, 7))
        self.axes_style(ax)
        if "algorithm" not in metrics:
            if events:
                ax[0].plot(
                    [row["step"] for row in events],
                    [row.get("training_loss", np.nan) for row in events],
                    color="#546ee5",
                    label="train minibatch objective",
                )
                rows = [row for row in events if "validation" in row]
                ax[0].plot(
                    [row["step"] for row in rows],
                    [row["validation"]["loss"] for row in rows],
                    marker="o",
                    color="#c24e10",
                    label="validation objective",
                )
                for key in ("expert_lat_mae", "expert_long_mae"):
                    ax[1].plot(
                        [row["step"] for row in rows],
                        [row["validation"]["metrics"].get(key, np.nan) for row in rows],
                        marker="o",
                        label=key,
                    )
            ax[0].set(ylabel="Objective", title="Measured training and validation events")
            ax[1].set(ylabel="MAE (m/s²)", title="Validation action error")
        else:
            for key in ("supervised_loss", "critic_loss", "actor_loss"):
                rows = [row for row in events if key in row]
                if rows:
                    ax[0].plot([row["step"] for row in rows], [row[key] for row in rows], label=key)
            ax[0].set(ylabel="Objective (different losses have different scales)")
            if events:
                ax[1].plot(
                    [row["step"] for row in events],
                    [row["reward"] for row in events],
                    linewidth=1,
                    label="training reward per step",
                )
                for key, color in (("terminated", "#b42318"), ("truncated", "#956e02")):
                    rows = [row for row in events if row.get(key)]
                    ax[1].scatter(
                        [row["step"] for row in rows],
                        [row["reward"] for row in rows],
                        color=color,
                        marker="x",
                        label=key,
                    )
            ax[1].set(ylabel="Reward", title="Training rollouts; termination markers are explicit")
        for axis in ax:
            axis.set_xlabel("Learner/environment step (see run type)")
            if axis.lines or axis.collections:
                axis.legend(fontsize=8)
        if not events:
            ax[0].text(
                0.5,
                0.5,
                "No events.jsonl; no curve reconstructed from endpoints",
                ha="center",
                transform=ax[0].transAxes,
            )
        self.save(
            fig,
            name,
            run["summary"]["name"],
            "训练曲线只使用 events.jsonl 实测点；缺失日志不会用首尾数值伪造曲线。",
        )

    def prediction(self, prediction, name):
        routes = [
            (route, data)
            for route, data in prediction["per_route"].items()
            if data["expert_frames"]
        ]
        if not routes:
            return
        fig, axes = self.plt.subplots(1, 2, figsize=(12, max(4, len(routes) * 0.4)))
        for dim, axis in enumerate(axes):
            positions = np.arange(len(routes))
            for offset, key, label, color in (
                (-0.18, "teacher_mae_mps2", "Teacher", COLORS["teacher"]),
                (
                    0.18,
                    "paired_adapted_mae_mps2",
                    "Policy (paired when available)",
                    COLORS["adapted"],
                ),
            ):
                if key == "teacher_mae_mps2" and not any(
                    data.get(key) is not None for _, data in routes
                ):
                    continue
                values = []
                for _, data in routes:
                    values_for_route = data.get(key)
                    if values_for_route is None and key == "paired_adapted_mae_mps2":
                        values_for_route = data["adapted_mae_mps2"]
                    values.append(values_for_route[dim] if values_for_route is not None else np.nan)
                axis.barh(
                    positions + offset,
                    values,
                    height=0.35,
                    label=label,
                    color=color,
                )
            axis.set_yticks(positions, [route for route, _ in routes], fontsize=8)
            axis.set(xlabel="MAE (m/s²)", title=("Lateral" if dim == 0 else "Longitudinal"))
            axis.legend()
            axis.grid(axis="x", alpha=0.2)
        self.save(
            fig,
            name,
            prediction["name"],
            "由逐帧预测重算。Teacher 与策略柱使用共同专家标签；"
            "没有 Teacher 时显示策略全部标签，详见配对样本数。",
        )

    def closed_loop(self, run, name):
        items = list(run["summary"]["evaluations"].items())
        fig, ax = self.plt.subplots(1, 3, figsize=(13, 4))
        phases = {
            "official_baseline": "Official",
            "initial_validation": "Offline anchor",
            "final_validation": "After online",
        }
        palette = {
            "official_baseline": "#546ee5",
            "initial_validation": "#18794e",
            "final_validation": "#c24e10",
        }
        labels = [phases[key] for key, _ in items]
        for axis, key, title in zip(
            ax,
            ("mean_controlled_return", "controlled_return_per_step", "terminated"),
            ("Mean controlled return", "Controlled return / step", "True terminal episodes"),
            strict=True,
        ):
            values = [item[key] if item[key] is not None else np.nan for _, item in items]
            axis.bar(labels, values, color=[palette[key] for key, _ in items])
            axis.tick_params(axis="x", rotation=20)
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.2)
        self.save(
            fig,
            name,
            run["summary"]["name"] + " · same-run validation",
            "同时看总控制回报、每步回报和真实终止。"
            "不同控制步数和终止惩罚会影响回报；单 seed 不能证明泛化。",
        )

    def online_actions(self, run, name):
        events = run["events"]
        if not events or "commanded_action" not in events[0]:
            return
        fig, axes = self.plt.subplots(2, 1, figsize=(11, 6))
        steps = [row["step"] for row in events]
        keys = {
            "expert_action": "#18794e",
            "official_action": "#546ee5",
            "anchor_action": "#8c6d31",
            "commanded_action": "#c24e10",
            "achieved_action": "#824fbd",
        }
        for dim, axis in enumerate(axes):
            for key, color in keys.items():
                if all(key in row for row in events):
                    axis.plot(
                        steps,
                        [row[key][dim] for row in events],
                        color=color,
                        label=key,
                        linewidth=1,
                    )
            axis.fill_between(
                steps,
                0,
                1,
                where=[not row.get("teacher_warmed", True) for row in events],
                transform=axis.get_xaxis_transform(),
                color="#ddd",
                alpha=0.3,
            )
            for previous, row in zip(events[:-1], events[1:], strict=True):
                if row["episode"] != previous["episode"]:
                    axis.axvline(row["step"], color="#888", linestyle=":", linewidth=0.8)
            axis.set(
                xlabel="Environment step",
                ylabel="Acceleration (m/s²)",
                title="Lateral" if dim == 0 else "Longitudinal",
            )
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
        self.save(
            fig,
            name,
            run["summary"]["name"] + " · visited-state actions",
            "比较专家、基座、anchor、最终命令和实际加速度；灰区为 warmup，竖线标记 episode reset。",
        )

    def trace_diagnostics(self, values, name):
        if "cross_track_after_m" not in values:
            return
        fig, axes = self.plt.subplots(2, 1, figsize=(11, 6))
        times = values["time_s"] + values["interval_s"]
        axes[0].plot(times, values["cross_track_after_m"], label="cross-track after step")
        if "road_half_width_m" in values:
            edge = values["road_half_width_m"]
            axes[0].plot(times, edge, "r--", label="road boundary")
            axes[0].plot(times, -edge, "r--")
        axes[0].set(ylabel="Cross-track (m)")
        axes[1].plot(times, values["rewards"], label="reward")
        for flag, color in (("terminated", "#b42318"), ("truncated", "#956e02")):
            mask = np.asarray(values[flag], dtype=bool)
            axes[1].scatter(times[mask], values["rewards"][mask], color=color, label=flag)
        axes[1].set(ylabel="Reward", xlabel="Reached-state time (s)")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        self.save(
            fig,
            name,
            "Reached-state diagnostics",
            "偏差与奖励对应动作执行后的状态。是否触发道路越界或真实终止，以 trace 的实际记录为准。",
        )
