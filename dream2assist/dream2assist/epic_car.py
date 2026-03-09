import math
import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Any, Dict

from gymnasium.envs.box2d.car_dynamics import Car


from epic_dynamics.dynamics.bike_dynamics import (
    default_state_dot_parameters,
)
from epic_dynamics.dynamics.load_transfer_bike_dynamics import (
    LoadTransferBikeDynamics,
)
from epic_dynamics.dynamics.load_transfer_bike_dynamics import (
    default_parameters as default_loadtransfer_bike_parameters,
)
from epic_dynamics.dynamics.load_transfer_bike_dynamics import (
    load_loadtransfer_bike_dynamics_vehicle_parameters,
)
from epic_dynamics.dynamics.vehicle_dynamics import VehicleDynamics
from epic_dynamics.dynamics.vehicle_with_pose_dynamics import (
    LoadTransferBikeWithPoseDynamics,
    VehicleWithPoseDynamics,
)
import epic_dynamics.parameters.params_selector as ps
from platform_interface.msg import UnifiedControl, UnifiedVehicleState
from vehicle_dynamics_sim.utils.load_params import (
    load_params_file,
)


class EpicCar(Car):
    """
    Class that implements the Epic Car simulator.
    It inherits from the Car class and implements the methods to simulate the dynamics of the car.

    Note: A portion of this implementation mirrors code in
    https://github.shared-services.aws.tri.global/driving/vehicle_dynamics_sim/blob/elliot/sims/scripts/sim_node.py
    """

    def __init__(
        self,
        *args,
        vehicle="loadtransfer_Leia",
        sim_discretization_time=0.1,
        euler_discretization_time=0.0005,
        **kwargs,
    ):
        super(EpicCar, self).__init__(*args, **kwargs)

        self.vehicle = vehicle

        # selecting vehicle dynamics model
        if self.vehicle == "loadtransfer_Leia":
            result = load_loadtransfer_bike_dynamics_vehicle_parameters("Leia")
            constant_parameters = result[0]
            self.dynamics_state_dot_params = result[1]
            default_loadtransfer_bike_parameters.update(constant_parameters)
            self.dynamics = LoadTransferBikeDynamics(
                default_loadtransfer_bike_parameters, self.dynamics_state_dot_params
            )
        else:
            raise ValueError("Vehicle model not compatible with sim_node.")

        self.params = load_params_file("sim_params.yaml")

        # get vehicle parameters to determine conversion to vehicle controls
        if self.vehicle == "Keisuke":
            vehicle_params = ps.select_vehicle_params("Keisuke")
        elif (
            self.vehicle == "Leia"
            or self.vehicle == "neuralnet_Leia"
            or self.vehicle == "loadtransfer_Leia"
            or self.vehicle == "loadtransfer_topography_Leia"
        ):
            vehicle_params = ps.select_vehicle_params("Leia")
        else:
            raise ValueError("Unknown joy vehicle model name")
        self.vehicle_params = vehicle_params

        # TODO(jon): Remove this
        self.controller_states = [
            "delta",
            "engine_torque",
            "front_brake_torque",
            "rear_brake_torque",
        ]
        self.driver_inputs = [
            0.0,
            0.0,
            0.0,
        ]  # driver steering, throttle, and brake (in that order)
        self.received_first_control_msg = False
        self._sim_time = 0.0
        self.printout = False

        self.init_sim(self.params, self.dynamics, self.dynamics_state_dot_params)
        self.reset_sim()

        self.state_names = self.dynamics.names_states

        # Overwrite parameters
        self.params["sim_discretization_time"] = sim_discretization_time
        self.params["euler_discretization_time"] = euler_discretization_time

    @property
    def params(self) -> Dict:
        """Return a dictionary of parameters."""
        return self._params

    @params.setter
    def params(self, value: Dict):
        """Set the dictionary of parameters."""
        self._params = value

    @property
    def dynamics(self) -> VehicleWithPoseDynamics:
        """Return the dynamics of the class."""
        return self._dynamics

    @dynamics.setter
    def dynamics(self, value: Dict):
        """Set the dynamics."""
        self._dynamics = value

    @property
    def dynamics_state_dot_params(self) -> Dict:
        """Return the parameters of the state_dot function of the dynamics."""
        return self._dynamics_state_dot_params

    @dynamics_state_dot_params.setter
    def dynamics_state_dot_params(self, value: Dict):
        """Set the parameters of the state_dot function of the dynamics."""
        self._dynamics_state_dot_params = value

    @property
    def map_data(self) -> Dict[str, jnp.array]:
        """Return the dictionary containing all the keys in the reference map."""
        return self._map_data

    def set_dynamics_state_dot_params(self, dynamics_state_dot_params: Dict[str, Any]):
        """
        Sets the parameters of the state_dot function of the dynamics.

        Args:
            dynamics_state_dot_params: parameters of the state_dot function of
                the dynamics.
                (str, Any) dictionary
        """
        self._dynamics_state_dot_params = dynamics_state_dot_params

    @property
    def sim_time(self) -> float:
        """Return the current simulation time."""
        return self._sim_time

    @property
    def sim_discretization_time(self) -> int:
        """Return the sim discretization time."""
        return self.params["sim_discretization_time"]

    @property
    def euler_discretization_time(self) -> int:
        """Return the euler discretization time."""
        return self.params["euler_discretization_time"]

    @property
    def initial_state(self) -> np.array:
        """Return the initial state."""
        states = self.params["initial_state"]
        states[-3] = self.hull.position[0]
        states[-2] = self.hull.position[1]
        states[-1] = self.hull.angle
        return states

    @property
    def initial_control(self) -> np.array:
        """Return the initial control."""
        return self.params["initial_control"]

    def init_sim(
        self,
        params: Dict[str, Any],
        dynamics: VehicleDynamics,
        dynamics_state_dot_params: Dict[str, Any] = default_state_dot_parameters,
    ):
        """
        Initializes the simulator.

        Args:
            params: dictionary of parameters for the node
                (dict)
            dynamics: dynamics of the vehicle
                (Dynamics)
            dynamics_state_dot_params: parameters of the state_dot function of
                the dynamics.
                (str, Any) dictionary
        """

        if not isinstance(dynamics, VehicleDynamics):
            err_msg = "[SimNode] dynamics should be of type "
            err_msg += str((type(dynamics))) + "."
            raise TypeError(err_msg)

        dynamics_state_dot_params["gear"] = params["gear"]
        dynamics_state_dot_params["speed_min"] = params["speed_min"]
        dynamics_state_dot_params["yaw_decay"] = params["yaw_decay"]
        dynamics_state_dot_params["brake_decay"] = params["brake_decay"]

        self._params = params

        if isinstance(dynamics, LoadTransferBikeDynamics):
            self._dynamics = LoadTransferBikeWithPoseDynamics(
                dynamics.params, dynamics_state_dot_params
            )
        else:
            self._dynamics = VehicleWithPoseDynamics(dynamics)
        self.set_dynamics_state_dot_params(dynamics_state_dot_params.copy())

    def reset_sim(self):
        """
        Resets the simulator.
        """

        self.state_current = self.initial_state.copy()
        self.state_dot = [0.0] * len(self.initial_state)
        self.control_from_msg = self.initial_control.copy()
        self.control_current = self.initial_control.copy()
        self.control_previous = self.initial_control.copy()

        self._sim_time = 0.0

    def apply_controls(self, action):
        """Apply actions for use in car model.  Actions are [steer, gas, brake]."""
        # TODO: Remove if not needed.  These might be still used for rendering.
        if action is None:
            return
        self.steer(action[0])
        self.gas(action[1])
        self.brake(action[2])

        # Apply unified control scaling.
        max_handwheel_angle = 450.0 * math.pi / 180.0
        driver_handwheel = action[0] * max_handwheel_angle  # handwheel is in range [-1, 1]
        driver_throttle = action[1]  # throttle is in range [0, 1]
        driver_brake = action[2]  # brake is in range [0, 1]

        # Convert driver inputs to vehicle inputs.
        steering = driver_handwheel / self.vehicle_params["Steering"]["steering_ratio"]
        engine_torque = driver_throttle * self.vehicle_params["Engine"]["engine_torque_max"]
        total_brake_torque = (
            0.01 * driver_brake * self.vehicle_params["Brakes"]["brake_torque_min"]
        )
        front_brake_torque = total_brake_torque * self.params["joy_brake_bias"]
        rear_brake_torque = total_brake_torque * (1 - self.params["joy_brake_bias"])

        self.control_current[self.controller_states.index("delta")] = steering
        self.control_current[self.controller_states.index("engine_torque")] = engine_torque
        self.control_current[self.controller_states.index("front_brake_torque")] = (
            front_brake_torque
        )
        self.control_current[self.controller_states.index("rear_brake_torque")] = rear_brake_torque

    def apply_control_input_limits(self):
        """
        Limits control inputs based on actuator constraints.
        """
        assert (
            self.controller_states
        ), "Controller states must be set before applying control input limits."
        for i in range(len(self.controller_states)):
            control_prev = self.control_previous[i]
            dt = self.sim_discretization_time
            if self.controller_states[i] == "delta":
                control_min = max(
                    self.dynamics_state_dot_params["delta_min"],
                    control_prev + dt * self.dynamics_state_dot_params["delta_dot_min"],
                )
                control_max = min(
                    self.dynamics_state_dot_params["delta_max"],
                    control_prev + dt * self.dynamics_state_dot_params["delta_dot_max"],
                )
            elif self.controller_states[i] == "engine_torque":
                control_min = max(
                    self.dynamics_state_dot_params["engine_torque_min"],
                    control_prev + dt * self.dynamics_state_dot_params["engine_torque_dot_min"],
                )
                control_max = min(
                    self.dynamics_state_dot_params["engine_torque_max"],
                    control_prev + dt * self.dynamics_state_dot_params["engine_torque_dot_max"],
                )
            elif (
                self.controller_states[i] == "front_brake_torque"
                or self.controller_states[i] == "rear_brake_torque"
            ):
                control_min = max(
                    self.dynamics_state_dot_params["brake_torque_min"],
                    control_prev + dt * self.dynamics_state_dot_params["brake_torque_dot_min"],
                )
                control_max = min(
                    self.dynamics_state_dot_params["brake_torque_max"],
                    control_prev + dt * self.dynamics_state_dot_params["brake_torque_dot_max"],
                )
            else:  # default to effectively no limits if state doesn't match above names
                control_min = -1.0e10
                control_max = 1.0e10

            self.control_current[i] = np.clip(self.control_current[i], control_min, control_max)

    def step(self):
        """
        Simulate the dynamics.  Overrides the step method in Car class.
        step() must be preceded by apply_controls(); otherwise we'll be reusing stale control values.
        """
        # Call the subclass step to update state, per the actions under its physics model, but then later correct the position / yaw.
        super(EpicCar, self).step(dt=self.sim_discretization_time)

        # set control_previous and control_current
        self.control_previous = self.control_current.copy()
        # self.control_current = self.control_from_msg.copy()

        # apply input limits
        self.apply_control_input_limits()

        east_index = self.state_names.index("position_east")
        north_index = self.state_names.index("position_north")
        heading_index = self.state_names.index("heading")

        if self.printout:
            print(
                f"{self.dynamics.names_states[east_index]}: {self.state_current[east_index]:.2f}"
            )
            print(
                f"{self.dynamics.names_states[north_index]}: {self.state_current[north_index]:.2f}"
            )
            print(
                f"{self.dynamics.names_states[heading_index]}: {self.state_current[heading_index]:.2f}"
            )
            for i in range(len(self.controller_states)):
                print(f"{self.controller_states[i]}: {self.control_current[i]:.2f}")

        # update vehicle state, state dot, and sim time

        num_euler_steps = round(self.sim_discretization_time / self.euler_discretization_time)

        # Correct the yaw angle.
        self.state_current[heading_index] += np.pi / 2

        for n in range(num_euler_steps):
            [self.state_current, self.state_dot] = self.integrate_dynamics(
                self.state_current, self.control_current, self.euler_discretization_time
            )
        self._sim_time += self.sim_discretization_time

        # Un-correct the yaw angle.
        self.state_current[heading_index] -= np.pi / 2

        # Update the underlying car's position and yaw, for reward calculation and rendering.
        # Note that we're relying on the state_dot ordering in VehicleWithPoseDynamics, which has E, N, heading as its last entries.
        if self.printout:
            print(
                f" state dot {float(self.state_dot[east_index])}, {float(self.state_dot[north_index])}, {float(self.state_dot[heading_index])}"
            )
            print(
                f" control states {float(self.control_current[0])}, {float(self.control_current[1])}, {float(self.control_current[2])}"
            )
            print(
                f" pose states {float(self.state_current[east_index])}, {float(self.state_current[north_index])}, {float(self.state_current[heading_index])}"
            )
            print("..............................................................................")
        self.hull.position = (
            float(self.state_current[east_index]),
            float(self.state_current[north_index]),
        )
        self.hull.angle = float(self.state_current[heading_index])

    @partial(jit, static_argnums=(0,))
    def integrate_dynamics(
        self,
        state: np.array,
        control: np.array,
        dt: float,
    ) -> jnp.array:
        """
        Predicts the next state from the current state

        Args:
            state: current state of the system with pose variables (see names_states)
                (_num_states, ) array
            control: control input applied to the system (see names_controls)
                (_num_controls, ) array
            dt: euler discretization time
                (float)

        Returns:
            next_state: next state of the system with pose variables
                (_num_states, ) array
            state_dot: time derivative of the state
                (_num_states, ) array
        """

        speed_min = self.dynamics_state_dot_params["speed_min"]
        yaw_decay = self.dynamics_state_dot_params["yaw_decay"]
        max_wr = self.dynamics_state_dot_params["max_wrs"][self.params["gear"] - 1]

        speed_idx = self.dynamics.names_states.index("lin_vel")
        speed_curr = state[speed_idx]

        # TODO(jon): I had to add the [0], since it seems state_dot was not properly flattened / concatenated in VehicleWithPoseDynamics.
        state_dot = self.dynamics_state_dot(state, control)[0]
        next_state = []

        # performing euler integration and adjusting states as needed
        # for low speeds / high rear wheelspeed
        for i, state_name in enumerate(self.dynamics.names_states):

            next_state_i = state[i] + state_dot[i] * dt

            # below minimum speed, decay yaw rate to 0
            # this results in reasonable spinout behavior
            if state_name == "yaw_rate":
                next_state_i = jnp.where(
                    speed_curr > speed_min,
                    next_state_i,
                    state[i] - (2 * (1 / (1 + jnp.exp(-yaw_decay * state[i]))) - 1) * dt,
                )

            # below minimum speed, reset sideslip to 0
            # this enables the driver to accelerate again after a spinout
            elif state_name == "sideslip":
                next_state_i = jnp.where(speed_curr > speed_min, next_state_i, 0)

            # making sure that V is never negative
            elif state_name == "lin_vel":
                next_state_i = jnp.maximum(next_state_i, 0)

            # limiting wr based on engine/motor power
            elif state_name == "rear_wheel_speed":
                next_state_i = jnp.minimum(next_state_i, max_wr)

            next_state.append(next_state_i)

        return [next_state, state_dot]

    @partial(jit, static_argnums=(0,))
    def dynamics_state_dot(
        self,
        state: jnp.array,
        control: jnp.array,
    ) -> jnp.array:
        """
        Computes the time derivative of the system state. Returns x_dot = f(x, u)
        where f describes the dynamics of the system.

        Args:
            state: state of the system with pose variables (see names_states)
                (_num_states, ) array
            control: control input applied to the system (see names_controls)
                (_num_controls, ) array

        Returns:
            state_dot: time derivative of the state
                (_num_states, ) array
        """
        return self.dynamics.state_dot(state, control, self.dynamics_state_dot_params)
