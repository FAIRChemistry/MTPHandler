import numpy as np

from mtphandler.model import Plate

times = np.linspace(0, 10, 11)


p = Plate(
    id="MTP_001",
    name="Enzyme Kinetics",
    temperatures=[25] * 11,
    temperature_unit="C",
    times=times.tolist(),
    time_unit="s",
)
