import pytest
import os
import numpy as np

from .kinematic_kf import KinematicKalman, ObservationKind, States

GENERATED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'generated'))

class TestKinematic:
  def setup_method(self):
    self.kf = KinematicKalman(GENERATED_DIR)

  def test_kinematic_kf(self):
    np.random.seed(0)

    # Simple simulation
    dt = 0.01
    ts = np.arange(0, 5, step=dt)
    vs = np.sin(ts * 5)

    x = 0.0
    xs = []

    xs_meas = []

    xs_kf = []
    vs_kf = []

    xs_kf_std = []
    vs_kf_std = []

    for t, v in zip(ts, vs):
      xs.append(x)

      # Update kf
      meas = np.random.normal(x, 0.1)
      xs_meas.append(meas)
      self.kf.predict_and_observe(t, ObservationKind.POSITION, [meas])

      # Retrieve kf values
      state = self.kf.x
      xs_kf.append(float(state[States.POSITION].item()))
      vs_kf.append(float(state[States.VELOCITY].item()))
      std = np.sqrt(self.kf.P)
      xs_kf_std.append(float(std[States.POSITION, States.POSITION].item()))
      vs_kf_std.append(float(std[States.VELOCITY, States.VELOCITY].item()))

      # Update simulation
      x += v * dt

    xs, xs_meas, xs_kf, vs_kf, xs_kf_std, vs_kf_std = (np.asarray(a) for a in (xs, xs_meas, xs_kf, vs_kf, xs_kf_std, vs_kf_std))

    assert xs_kf[-1] == pytest.approx(-0.010866289677966417)
    assert xs_kf_std[-1] == pytest.approx(0.04477103863330089)
    assert vs_kf[-1] == pytest.approx(-0.8553720537261753)
    assert vs_kf_std[-1] == pytest.approx(0.6695762270974388)

    if "PLOT" in os.environ:
      import matplotlib.pyplot as plt  # pylint: disable=import-error
      plt.figure()
      plt.subplot(2, 1, 1)
      plt.plot(ts, xs, 'k', label='Simulation')
      plt.plot(ts, xs_meas, 'k.', label='Measurements')
      plt.plot(ts, xs_kf, label='KF')
      ax = plt.gca()
      ax.fill_between(ts, xs_kf - xs_kf_std, xs_kf + xs_kf_std, alpha=.2, color='C0')

      plt.xlabel("Time [s]")
      plt.ylabel("Position [m]")
      plt.legend()

      plt.subplot(2, 1, 2)
      plt.plot(ts, vs, 'k', label='Simulation')
      plt.plot(ts, vs_kf, label='KF')

      ax = plt.gca()
      ax.fill_between(ts, vs_kf - vs_kf_std, vs_kf + vs_kf_std, alpha=.2, color='C0')

      plt.xlabel("Time [s]")
      plt.ylabel("Velocity [m/s]")
      plt.legend()

      plt.show()

  def test_init_state(self):
    init_x = self.kf.x

    dim_state_err = self.kf.initial_P_diag.shape[0]

    new_x = np.copy(init_x)
    new_x[States.POSITION] = 100.0
    new_x[States.VELOCITY] = 5.0

    new_P = np.eye(dim_state_err) * 0.5

    self.kf.init_state(new_x, covs=new_P, filter_time=1.0)

    assert np.allclose(self.kf.x, new_x)
    assert np.allclose(self.kf.P, new_P)
    assert self.kf.t == 1.0

  def test_set_filter_time(self):
    assert np.isnan(self.kf.t)

    self.kf.filter.set_filter_time(10.5)
    assert self.kf.t == 10.5

  def test_predict(self):
    dim_state = self.kf.initial_x.shape[0]

    x0 = np.zeros(dim_state)
    x0[States.VELOCITY] = 10.0
    self.kf.init_state(x0, filter_time=0.0)

    t0 = self.kf.t
    dt = 0.1

    self.kf.filter.predict(t0 + dt)

    assert self.kf.t == pytest.approx(t0 + dt)
    assert self.kf.x[States.POSITION].item() == pytest.approx(1.0)

  def test_rewind(self):
    try:
      self.kf.filter.reset_rewind()
    except Exception as e:
      pytest.fail(f"reset_rewind raised exception: {e}")
