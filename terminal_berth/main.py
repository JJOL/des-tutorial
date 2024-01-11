import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class G:
    number_runs = 1
    warmup_period = 3*60
    results_collection_period = 8*60

    # resource parameters
    num_berths = 2
    num_cranes = 7

    # time parameters
    vessel_iat = 10

    pre_pcat_mt  = 2
    post_pcat_mt = 2

    truck_arrival_mt = 2
    operation_start_mt = 2
    pickup_mt = 2
    dropoff_mt = 2

    crane_move_mt = 2

    # data parameters
    vessels: list['Vessel'] = []
    
class Hatch:
    pass

class Vessel:
    pass

class Crane:
    pass

class Berth:
    pass

class Model:
    pass
