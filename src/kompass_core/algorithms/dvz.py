import logging
from typing import Any

from ..utils.common import base_validators, BaseAttrs
from ..utils.geometry import convert_to_0_2pi, convert_to_plus_minus_pi
import matplotlib.pyplot as plt
import numpy as np
from attrs import define, field
from matplotlib.patches import Ellipse, Polygon

from ..models import Robot, RobotState, RobotCtrlLimits

# Lapierre, Lionel & Zapata, Rene & Lépinay, Pascal. (2007). Simultaneous Path Following and Obstacle Avoidance Control of a Unicycle-type Robot. 2617 - 2622. 10.1109/ROBOT.2007.363860.

EPSILON_ANG = 0.01


@define
class DeformableVirtualZoneParams(BaseAttrs):
    """
        Deformable Virtual Zone Parameters

        ```{list-table}
    :widths: 10 10 10 70
    :header-rows: 1

    * - Name
      - Type
      - Default
      - Description

    * - min_front_margin
      - `float`
      - `1.0`
      - Minimum front margin distance. Must be between `0.0` and `1e2`.
    * - K_linear
      - `float`
      - `1.0`
      - Proportional gain for linear control. Must be between `0.1` and `10.0`.

    * - K_angular
      - `float`
      - `1.0`
      - Proportional gain for angular control. Must be between `0.1` and `10.0`.

    * - K_I
      - `float`
      - `5.0`
      - Proportional deformation gain. Must be between `0.1` and `10.0`.

    * - side_margin_width_ratio
      - `float`
      - `1.0`
      - Width ratio between the deformation zone front and side (circle if 1.0). Must be between `1e-2` and `1e2`.
    ```
    """

    min_front_margin: float = field(
        default=1.0, validator=base_validators.in_range(min_value=0.0, max_value=1e2)
    )

    K_linear: float = field(
        default=1.0, validator=base_validators.in_range(min_value=0.1, max_value=10.0)
    )

    K_angular: float = field(
        default=1.0, validator=base_validators.in_range(min_value=0.1, max_value=10.0)
    )

    K_I: float = field(
        default=5.0, validator=base_validators.in_range(min_value=0.1, max_value=10.0)
    )

    side_margin_width_ratio: float = field(
        default=1.0, validator=base_validators.in_range(min_value=1e-2, max_value=1e2)
    )


class DeformableVirtualZone:
    """DeformableVirtualZone."""

    def __init__(
        self,
        robot: Robot,
        ctrl_limits: RobotCtrlLimits,
        config: DeformableVirtualZoneParams,
    ) -> None:
        """
        Deformable Virtual Zone Control Implementation

        :param robot: Robot using the controller
        :type robot: Robot
        :param config: DVZ parameters
        :type config: DiformableVirtualZoneParams
        """

        self.robot = robot
        self.config = config
        self.ctrl_limits = ctrl_limits

        # control regularization
        self._set_control_regularization()

        # shift of the center of the zone from the center of the robot
        self.update_zone_size(robot_speed=robot.state.speed if robot.state else 0.0)

        # Init zone constant size parameters
        self._init_constant_zone_parameters()

    def _init_constant_zone_parameters(self) -> None:
        """
        Setup the DVZ constant zone parameters
        """
        zone_side_margin = self.robot.radius / self.config.side_margin_width_ratio

        self.zone_minor_radius: float = self.robot.radius + zone_side_margin
        self.zone_minor_radius_diff = 0.0

        self.zone_center_shift_y: float = 0.0  # a_y in the paper
        self.zone_ori_shift: float = 0.0  # gamma in the paper
        self.zone_shift_y_diff: float = 0.0

    def set_from_yaml(self, path_to_file: str) -> None:
        """Setup the DVZ controller params from YAML.

        :param path_to_file:
        :type path_to_file: str
        """
        self.config.from_yaml(path_to_file, nested_root_name="DVZ")
        self._set_control_regularization()

    def _set_control_regularization(self) -> None:
        """
        Compute regularization coefficients to keep the linear and angular controls within limits
        """
        deformation_max_at_angle = 0.25  # 1/4 of total
        angle_max_angular = np.pi / 4  # angle resulting in max angular action

        # Set not to exceed max allowed value at max angular
        self.angular_regulation = self.ctrl_limits.omega_limits.max_acc / (
            angle_max_angular * deformation_max_at_angle
        )

        self.linear_regulation = self.ctrl_limits.vx_limits.max_acc / (
            deformation_max_at_angle
        )

    def update_zone_size(self, robot_speed: float) -> None:
        """
        Update zone size based on the robot speed

        :param robot_speed: Robot linear velocity (m/s)
        :type robot_speed: float
        """
        self.zone_major_radius = (
            1 + (abs(robot_speed) / self.ctrl_limits.vx_limits.max_vel)
        ) * self.config.min_front_margin  # zone size is proportional to robot speed

        self.zone_major_radius_diff = (
            self.config.min_front_margin / self.ctrl_limits.vx_limits.max_vel
        )

        zone_shift_const = 2 / 3  # only one third of the area is behind the robot
        sign_speed = 1 if robot_speed == 0 else np.sign(robot_speed)
        self.zone_center_shift_x: float = (
            -zone_shift_const * sign_speed * self.zone_major_radius
        )
        self.zone_shift_x_diff: float = -zone_shift_const * self.zone_major_radius_diff

    def set_scan_values(self, scan_values: np.ndarray, scan_angles: np.ndarray) -> None:
        """
        Set scan values

        :param scan_values: Values
        :type scan_values: np.ndarray
        :param scan_angles: Angle ranges
        :type scan_angles: np.ndarray
        """
        self.scan_values = scan_values
        self.scan_angles = scan_angles

    def _get_undeformed_radius(self, alpha: float) -> float:
        """
        Computes the undeformed DVZ radius at a given angle

        :param alpha: Point angle on the ellipse
        :type alpha: float
        """
        ang_cos = np.cos(alpha - self.zone_ori_shift)
        ang_sin = np.sin(alpha - self.zone_ori_shift)

        # radius parameters
        _A = (self.zone_minor_radius * ang_cos) ** 2 + (
            self.zone_major_radius * ang_sin
        ) ** 2
        _B = 2 * (
            self.zone_center_shift_x * ang_cos * self.zone_minor_radius**2
            + self.zone_center_shift_y * ang_sin * self.zone_major_radius**2
        )
        _C = (
            (self.zone_center_shift_x * self.zone_minor_radius) ** 2
            + (self.zone_center_shift_y * self.zone_major_radius) ** 2
            - (self.zone_minor_radius * self.zone_major_radius) ** 2
        )

        _deformation_num = np.sqrt(_B**2 - 4 * _A * _C)

        undeformed_radius: float = (-_B + _deformation_num) / (2 * _A)

        return undeformed_radius

    def _get_deformation_radius(
        self, scan_value: float, undeformed_radius: float
    ) -> float:
        """
        Compute the deformation radius at a given scan value

        :param scan_angle: Scan angle (rad)
        :type scan_angle: float
        :param scan_value: Scan value (m)
        :type scan_value: float
        """
        if undeformed_radius > scan_value:
            deformed_radius: float = scan_value
        else:
            deformed_radius = undeformed_radius
        return deformed_radius

    def _get_grad_A_linear(self, angle: float) -> float:
        """
        Get the gradient of term A of the deformation formula with respect to control - "J_A^u" in the paper

        :param angle: deformation angle
        :type angle: float
        """
        _term_1 = (
            self.zone_minor_radius * self.zone_minor_radius_diff * np.cos(angle) ** 2
        )

        _term_2 = (
            self.zone_major_radius * self.zone_major_radius_diff * np.sin(angle) ** 2
        )

        grad_A_u = 2 * (_term_1 + _term_2)
        return grad_A_u

    def _get_grad_A_angular(self, angle: float) -> float:
        """
        Get the gradient of term A of the deformation formula with respect to the shift angle - "J_A^gamma" in the paper

        :param angle: deformation angle
        :type angle: float
        """
        grad_A_ang = (
            2
            * np.cos(angle)
            * np.sin(angle)
            * (self.zone_minor_radius**2 - self.zone_major_radius**2)
        )
        return grad_A_ang

    def _get_grad_B_linear(self, angle: float) -> float:
        """
        Get the gradient of term B of the deformation formula with respect to control - "J_B^u" in the paper

        :param angle: deformation angle
        :type angle: float
        """
        _term_1 = np.cos(angle) * (
            self.zone_minor_radius**2 * self.zone_shift_x_diff
            + 2
            * self.zone_center_shift_x
            * self.zone_minor_radius
            * self.zone_minor_radius_diff
        )

        _term_2 = np.sin(angle) * (
            self.zone_major_radius**2 * self.zone_shift_y_diff
            + 2
            * self.zone_center_shift_y
            * self.zone_major_radius
            * self.zone_major_radius_diff
        )

        grad_B_u = 2 * (_term_1 + _term_2)
        return grad_B_u

    def _get_grad_B_angular(self, angle: float) -> float:
        """
        Get the gradient of term B of the deformation formula with respect to the shift angle - "J_B^gamma" in the paper

        :param angle: deformation angle
        :type angle: float
        """
        grad_B_ang = 2 * (
            self.zone_center_shift_x * self.zone_minor_radius**2 * np.sin(angle)
            - self.zone_center_shift_y * self.zone_major_radius**2 * np.cos(angle)
        )
        return grad_B_ang

    def _get_grad_C_linear(self) -> float:
        """
        Get the gradient of term C of the deformation formula with respect to control - "J_C^u" in the paper

        :param angle: deformation angle
        :type angle: float
        """
        _term_1 = (
            self.zone_center_shift_x
            * self.zone_minor_radius
            * (
                self.zone_minor_radius * self.zone_shift_x_diff
                + self.zone_center_shift_x * self.zone_minor_radius_diff
            )
        )

        _term_2 = (
            self.zone_center_shift_y
            * self.zone_major_radius
            * (
                self.zone_major_radius * self.zone_shift_y_diff
                + self.zone_center_shift_y * self.zone_major_radius_diff
            )
        )

        _term_3 = (
            self.zone_major_radius
            * self.zone_minor_radius
            * (
                self.zone_major_radius * self.zone_minor_radius_diff
                + self.zone_minor_radius * self.zone_major_radius_diff
            )
        )

        grad_C_u = 2 * (_term_1 + _term_2 - _term_3)
        return grad_C_u

    def _init_deformation(self) -> None:
        """
        Init total deformation and its gradients
        """
        self.total_deformation: float = 0.0  # in [0, 1]
        self.deformation_orientation: float = 0.0  # in [0, 2pi]

        # to vis deformation
        self.deformation_plot = []

    def _regulate_deformation(self) -> None:
        """
        Normalize the deformation , it's orientation and compute the deformation regulation term
        """
        self.deformation_orientation = (
            self.deformation_orientation / self.total_deformation
        )
        self.total_deformation = self.total_deformation / self.regularization_coeff

        self.deformation_regulation: float = 1 / (
            1 + self.config.K_I * self.total_deformation
        )

    def get_gradients(self, angle: float) -> None:
        """
        Get all deformation gradients values

        :param angle: Deformation angle (rad)
        :type angle: float
        """
        self.grad_A_ang: float = self._get_grad_A_angular(angle)
        self.grad_A_u: float = self._get_grad_A_linear(angle)
        self.grad_B_ang: float = self._get_grad_B_angular(angle)
        self.grad_B_u: float = self._get_grad_B_linear(angle)
        self.grad_C_u: float = self._get_grad_C_linear()

    def get_total_deformation(self, compute_deformation_plot: bool = False) -> None:
        """
        Get total deformationa nd total gradients using available scan values
        """
        self._init_deformation()

        self.regularization_coeff = len(self.scan_angles)

        for idx, angle in enumerate(self.scan_angles):
            # Get deformation at angle
            undeformed_radius: float = self._get_undeformed_radius(angle)

            deformed_radius: float = self._get_deformation_radius(
                scan_value=self.scan_values[idx], undeformed_radius=undeformed_radius
            )

            if compute_deformation_plot:
                self.deformation_plot.append((angle, deformed_radius))

            if deformed_radius < undeformed_radius:  # if the zone is deformed at angle
                new_deformation = (
                    undeformed_radius - deformed_radius
                ) / deformed_radius

                # Update total deformation
                self.total_deformation += new_deformation

                self.deformation_orientation += new_deformation * convert_to_0_2pi(
                    angle
                )

        if self.total_deformation > 0.0:
            self._regulate_deformation()

    def set_control_params(
        self, linear_gain: float, angular_gain: float, deformation_gain: float
    ) -> None:
        """
        Set control parameters

        :param linear_gain: Gain parameter applied to the linear control
        :type linear_gain: float
        :param angular_gain: Gain parameter applied to the angular control
        :type angular_gain: float
        :param deformation_gain: Gain parameter applied to the deformation regulation function
        :type deformation_gain: float
        """
        self.config.K_linear = linear_gain
        self.config.K_angular = angular_gain
        self.config.K_I = deformation_gain
        self._set_control_regularization()

    def compute_linear_control(
        self, ref_control_linear: float, old_control: float, time_step: float
    ) -> float:
        """
        Compute the DVZ modified linear control given a reference control

        :param ref_control_linear: Reference control value
        :type ref_control_linear: float
        :param old_control: Last control value
        :type old_control: float

        :return: Modified linear control
        :rtype: float
        """
        if self.total_deformation > 0.0:
            orientation_regulated = (
                convert_to_plus_minus_pi(self.deformation_orientation) + EPSILON_ANG
            )  # to avoid division by zero

            logging.debug(
                f"Total deformation {self.total_deformation}, deformation angle {self.deformation_orientation}"
            )

            _dvz_acc = (
                -self.config.K_linear
                * self.total_deformation
                * self.linear_regulation
                / orientation_regulated
            )

            _dzv_control = _dvz_acc * time_step + old_control
            logging.debug(
                f"DVZ acceleration: {_dvz_acc}, DVZ control {_dzv_control}, Deformation regulation {self.deformation_regulation}, reference control {ref_control_linear}, old control {old_control}"
            )
            linear_ctr: float = (
                1 - self.deformation_regulation
            ) * _dzv_control + self.deformation_regulation * ref_control_linear
        else:
            # If the zone is not deformed apply the reference control
            linear_ctr = ref_control_linear

        return min(linear_ctr, self.ctrl_limits.vx_limits.max_vel)

    def compute_angular_control(self, ref_control_angular: float) -> float:
        """
        Compute the DVZ modified angular control given a reference control

        :param ref_control_angular: Reference control value
        :type ref_control_angular: float

        :return: Modified angular control
        :rtype: float
        """
        if self.total_deformation > 0.0:
            inv_angle = convert_to_plus_minus_pi(np.pi - self.deformation_orientation)

            _dzv_control = (
                -self.config.K_angular
                * inv_angle
                * self.total_deformation
                * self.angular_regulation
            )

            angular_ctr: float = (
                1 - self.deformation_regulation
            ) * _dzv_control + self.deformation_regulation * ref_control_angular

            logging.debug(
                f"Total: {angular_ctr}, DVZ angular control: {_dzv_control}, Deformation regulation {self.deformation_regulation}"
            )
        else:
            # If the zone is not deformed apply the reference control
            angular_ctr = ref_control_angular

        return min(angular_ctr, self.ctrl_limits.omega_limits.max_vel)

    def plt_robot_zone(
        self, robot_state: RobotState, fig_ax: Any = None, display_now: bool = False
    ) -> None:
        """
        Plot the robot and the deformable zone

        :param robot_state: Current robot state
        :type robot_state: RobotState
        :param fig_ax: Plot figure, defaults to None
        :type fig_ax: Any, optional
        :param display_now: Display the figure, defaults to False
        :type display_now: bool, optional
        """
        # Create new figure if none is given
        if not fig_ax:
            fig_ax = plt.gca()

        self.robot.footprint.plt_robot(
            x=robot_state.x, y=robot_state.y, heading=robot_state.yaw, ax=fig_ax
        )
        fig_ax.set(aspect=1)

        zone_x = robot_state.x - self.zone_center_shift_x
        zone_y = robot_state.y + self.zone_center_shift_y
        zone_angle = robot_state.yaw + self.zone_ori_shift

        fig_ax.add_patch(
            Ellipse(
                xy=(zone_x, zone_y),
                width=self.zone_major_radius * 2,
                height=self.zone_minor_radius * 2,
                angle=zone_angle,
                color="red",
                alpha=0.5,
                fill=True,
            )
        )
        fig_ax.plot(zone_x, zone_y, "r+")

        # plot deformation from laser scan
        deformation_plot_xy = [
            (
                point[1] * np.cos(point[0]) + robot_state.x,
                point[1] * np.sin(point[0]) + robot_state.y,
            )
            for point in self.deformation_plot
        ]
        polygon = Polygon(deformation_plot_xy, fill=False)

        fig_ax.add_patch(polygon)
        fig_ax.autoscale()

        if display_now:
            plt.show()
