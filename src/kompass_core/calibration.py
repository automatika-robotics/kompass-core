import logging

from .utils.geometry import convert_to_plus_minus_pi
import matplotlib.pyplot as plt
import numpy as np
from .datatypes.path import MotionSample, PathSample
from scipy import optimize

from .models import MotionModel2D, Robot, RobotState
from .simulation import RobotSim


class ModelFitting(RobotSim):
    FITTING_METHODS = {"CURVE_FIT": 0}

    def __init__(
        self, params_file: str, fitting_method: int = FITTING_METHODS["CURVE_FIT"]
    ) -> None:
        super().__init__(params_file)
        self.fitting_method: int = fitting_method

    def prep_data(self, robot_data: MotionSample):
        """
        Prepare given motion data for model fitting

        :param robot_data: Recorded robot motion data
        :type robot_data: MotionSample
        """
        dt = np.diff(robot_data.time)
        robot_data.path_sample.heading_points = convert_to_plus_minus_pi(
            robot_data.path_sample.heading_points
        )
        self.linear_y_out = robot_data.path_sample.y_points[1:]
        self.linear_y_in = (
            robot_data.path_sample.y_points[:-1],
            robot_data.path_sample.heading_points[:-1],
            robot_data.control[:-1, 0],
            dt,
        )

        self.linear_x_out = robot_data.path_sample.x_points[1:]
        self.linear_x_in = (
            robot_data.path_sample.x_points[:-1],
            robot_data.path_sample.heading_points[:-1],
            robot_data.control[:-1, 0],
            dt,
        )

        self.angular_out = robot_data.path_sample.heading_points[1:]
        self.angular_in = (
            robot_data.path_sample.heading_points[:-1],
            robot_data.control[:-1, 1],
            dt,
        )

    def fit_data(self, log: bool = False) -> MotionModel2D:
        """
        Fit robot model parameters to a given data

        :param log: To log the result, defaults to False
        :type log: bool, optional

        :return: Calibrated robot model
        :rtype: MotionModel2D
        """
        motion_model = self.robot.state.model
        vx_opt, vx_cov = optimize.curve_fit(
            MotionModel2D.x_model,
            self.linear_x_in,
            self.linear_x_out,
            p0=[motion_model.params.x_dot_prop_vx, motion_model.params.x_dot_prop_vy],
        )

        motion_model.set_linear_x_params(vx_opt)

        vy_opt, vy_cov = optimize.curve_fit(
            MotionModel2D.y_model,
            self.linear_y_in,
            self.linear_y_out,
            p0=[motion_model.params.y_dot_prop_vx, motion_model.params.y_dot_prop_vy],
        )

        motion_model.set_linear_y_params(vy_opt)

        ang_opt, ang_cov = optimize.curve_fit(
            MotionModel2D.heading_model,
            self.angular_in,
            self.angular_out,
            p0=[motion_model.params.yaw_dot_prop],
        )

        motion_model.set_angular_params(ang_opt)

        if log:
            logging.info(f"Calibration Result: {motion_model}")
            logging.info(
                f"""
                         Optimal linear x-coordinates parameters: {vx_opt} with the covariance {np.sqrt(np.diag(vx_cov))}
                         Optimal linear y-coordinates parameters: {vy_opt} with the covariance {np.sqrt(np.diag(vy_cov))}
                         Optimal angular parameters: {ang_opt} with the covariance {np.sqrt(np.diag(ang_cov))}"""
            )

        return motion_model


class Calibration:
    @classmethod
    def simulate_calibrated_model_data(
        cls,
        calibrated_model: MotionModel2D,
        motion_sample: MotionSample,
        geometry_type: str,
        geometry_params: np.ndarray,
        robot_type: str,
    ) -> PathSample:
        """
        Simulates robot motion along a given set of commands using the calibrated model

        :param calibrated_model: Given robot model
        :type calibrated_model: MotionModel2D
        :param motion_sample: Robot motion data containing the control commands and actual path
        :type motion_sample: MotionSample
        :param footprint_type: Robot footprint type
        :type footprint_type: int
        :param robot_type: Robot type
        :type robot_type: str

        :return: Modeled robot path
        :rtype: PathSample
        """
        init_state = RobotState(
            x=motion_sample.path_sample.x_points[0],
            y=motion_sample.path_sample.y_points[0],
            yaw=motion_sample.path_sample.heading_points[0],
            motion_model=calibrated_model,
        )

        robot = Robot(
            robot_type=robot_type,
            state=init_state,
            geometry_params=geometry_params,
            geometry_type=geometry_type,
        )
        dt = motion_sample.time[1] - motion_sample.time[0]

        modeled_path: PathSample = RobotSim.simulate_motion(
            time_step=dt,
            number_of_steps=motion_sample.length,
            control_seq=motion_sample.control,
            robot=robot,
        )

        return modeled_path

    @classmethod
    def calibrate_data(
        cls, model_fitting: ModelFitting, robot_data: MotionSample
    ) -> MotionModel2D:
        """
        Calibrate robot model using given data

        :param model_fitting: Model fitting object
        :type model_fitting: ModelFitting
        :param robot_data: Recorded robot data for calibration
        :type robot_data: MotionSample

        :return: Calibrated model
        :rtype: MotionModel2D
        """
        model_fitting.prep_data(robot_data)
        calibrated_model = model_fitting.fit_data(log=True)
        return calibrated_model

    @classmethod
    def vis_calibration(cls, robot_data: MotionSample, modeled_path: PathSample):
        """
        Plot the robot calibration results (real data vs. model output)

        :param robot_data: Robot motion data
        :type robot_data: MotionSample
        :param modeled_path: Motion model output data
        :type modeled_path: PathSample
        """
        fig, [ax0, ax1, ax2, ax3] = plt.subplots(nrows=4, ncols=1, figsize=(8, 8))
        fig.suptitle("Calibration Results")
        fig.tight_layout(pad=3.0)

        # Plot steering control
        ax0.plot(
            robot_data.time,
            robot_data.path_sample.x_points,
            color="red",
            label="Real",
        )
        ax0.plot(
            robot_data.time,
            modeled_path.x_points,
            color="blue",
            label="Model",
        )
        ax0.legend()
        ax0.set_title("X-axis motion")
        ax0.set_xlabel("time (s)")
        ax0.set_ylabel("X (m)")

        # Plot y error
        ax1.plot(
            robot_data.time,
            robot_data.path_sample.y_points,
            color="red",
            label="Real",
        )
        ax1.plot(
            robot_data.time,
            modeled_path.y_points,
            color="blue",
            label="Model",
        )
        ax1.set_title("Y-axis motion")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("Y (m)")

        # Plot orientation error
        ax2.plot(
            robot_data.time,
            convert_to_plus_minus_pi(robot_data.path_sample.heading_points),
            color="red",
            label="Real",
        )
        ax2.plot(
            robot_data.time,
            convert_to_plus_minus_pi(modeled_path.heading_points),
            color="blue",
            label="Model",
        )
        ax2.set_title("Angular motion (absolute)")
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("Heading (rad)")

        # Plot control
        ax3.plot(robot_data.time, robot_data.control[:, 1], color="red")
        ax3.set_title("Angular control")
        ax3.set_xlabel("time (s)")
        ax3.set_ylabel("control (rad/s)")
        plt.show()
