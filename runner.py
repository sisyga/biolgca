from lgca import get_lgca
import numpy as np
import time
from os import environ as env
from sys import stderr

# util
# - config
def conf(name, default=None):
  try:
    return env[f"LGCA_{name}"]
  except KeyError:
    return default

str_conf   = lambda name, default="": str(conf(name, default))
int_conf   = lambda name, default=0:  int(conf(name, default))
float_conf = lambda name, default=.0: float(conf(name, default))

# - time
time_since    = lambda t: time.time() - t
minutes_since = lambda t: time_since(t) / 60

# - io
write = lambda msg: print(msg)
log   = lambda msg: print(f"biolgca: {msg}")

# run
log("loading configuration")
config = {
  "output": str_conf("OUTPUT", default="results.numpy"),
  "runs":   int_conf("RUNS", default=1),
  "lgca": {
    "dims": int_conf("DIMS", default=1),
    "density":   int_conf("DENSITY", default=1),
    "birthrate": float_conf("BIRTHRATE", default=0.5),
    "deathrate": float_conf("DEATHRATE", default=0.1),
    "restchannels": int_conf("RESTCHANNELS", default=1),
  },
}

log(str(config))

log("creating lgca builder")
new_lgca = lambda: get_lgca(
  ib = True,
  bc = 'reflecting',
  geometry     = 'lin',
  interaction  = 'inheritance',
  variation    = False,
  density      = config["lgca"]["density"],
  restchannels = config["lgca"]["restchannels"],
  dims = config["lgca"]["dims"],
  r_b  = config["lgca"]["birthrate"],
  r_d  = config["lgca"]["deathrate"]
)

log("preparing run data")
num_runs = config["runs"]
runs = ((i, new_lgca()) for i in range(1, num_runs + 1))
thom = np.zeros(num_runs)

log("entering run-loop")
t0 = time.time()

for run_no, lgca in runs:
  runlog = lambda msg: log(f"run {run_no}/{num_runs} - {msg}")

  runlog("started")
  run_start = time.time()

  lgca.timeevo_until_hom(record=True)
  run_timesteps = len(lgca.props_t)
  thom[run_no - 1] = run_timesteps

  ## write timesteps to stdout - deactivated for now cuz we log to stdout
  # write(run_timesteps)
  
  run_duration  = "{:5.3f} minutes".format(minutes_since(run_start))
  runlog(f"homogenous after {run_timesteps} timesteps")
  runlog(f"completed in {run_duration}") 

total_duration = "{:5.3f} minutes".format(minutes_since(t0))
log(f"completed in {total_duration}")

log("saving data to " + config["output"])
np.save(config["output"], thom)

log("done")