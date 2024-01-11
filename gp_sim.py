import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print('Booting up...')

    clinic_times: list[int] = []
    reception_waiting: list[int] = []
    gp_waiting: list[int] = []
    book_waiting: list[int] = []

    G.num_gps = 2

    for run in range(G.number_runs):
        if run % 10 == 0:
            print(f'Run: {run}')
        model = Model()
        model.run()
        df = Patient.to_df()
        clinic_times.append(df['clinic_waiting'].mean())
        reception_waiting.append(df['reception_waiting'].mean())
        gp_waiting.append(df['gp_waiting'].mean())
        book_waiting.append(df['book_waiting'].mean())

    print('Summarising Results...')
    clinic_times = list(filter(lambda x: not np.isnan(x), clinic_times))
    reception_waiting = list(filter(lambda x: not np.isnan(x), reception_waiting))
    gp_waiting = list(filter(lambda x: not np.isnan(x), gp_waiting))
    book_waiting = list(filter(lambda x: not np.isnan(x), book_waiting))

    mean_clinic_time = round(np.mean(clinic_times), 2)
    mean_registration_waiting_time = round(np.mean(reception_waiting), 2)
    mean_surgery_waiting_time = round(np.mean(gp_waiting), 2)
    mean_booking_waiting_time = round(np.mean(book_waiting), 2)


    print('Number of runs: ', G.number_runs)
    print('Warmup Period: ', G.warmup_period, ' mins')
    print('Results Collection Period: ', G.results_collection_period, ' mins')
    print('Number of GPs: ', G.num_gps)
    print(f'Clinic Waiting: {mean_clinic_time}', ' mins')
    print(f'Reception Waiting: {mean_registration_waiting_time}', ' mins')
    print(f'GP Waiting: {mean_surgery_waiting_time}', ' mins')
    print(f'Book Waiting: {mean_booking_waiting_time}', ' mins')

    print('Plotting Results...')
    fig, ax = plt.subplots()
    labels = ['Clinic Time', 'Reception time', 'GP time', 'Book time']
    times  = [mean_clinic_time, mean_registration_waiting_time, mean_surgery_waiting_time, mean_booking_waiting_time]
    
    ax.bar(labels, times)
    ax.set_ylabel('Time (mins)')
    ax.set_title(f'Patient Times w/ {G.num_gps} doctors')
    
    plt.savefig(f'./r_{G.num_gps}_doctors.png')

    print('Done!')

class G:
    number_runs = 100
    warmup_period = 3*60
    results_collection_period = 8*60

    booking_test_prob = 0.25

    # resource parameters
    num_gps = 2
    num_receptionists = 1

    # parameters
    patient_iat = 3
    registration_mt = 2
    gp_consult_mt = 8
    booking_mt = 4

    call_iat = 10
    call_mt = 4

class QueueRecorder:
    def __init__(self, env: simpy.Environment, name: str) -> None:
        self.env = env
        self.name = name

        self.enter_time = np.nan
        self.leave_time = np.nan
        self.waiting_time = np.nan

    def enter(self):
        self.enter_time = self.env.now

    def leave(self):
        self.leave_time = self.env.now
        self.waiting_time = self.leave_time - self.enter_time

class Patient:
    collection: list['Patient'] = []

    def __init__(self, env) -> None:
        self.id = len(Patient.collection)
        Patient.collection.append(self)

        self.q_reception = QueueRecorder(env, 'Reception')
        self.a_reception_time = np.nan
        self.q_gp = QueueRecorder(env, 'GP_Consultation')
        self.a_gp_time = np.nan
        self.q_book = QueueRecorder(env, 'Booking')
        self.a_book_time = np.nan

        self.clinic = QueueRecorder(env, 'Clinic_System')

    @staticmethod
    def to_df() -> pd.DataFrame:
        collected = list(filter(lambda p: p.clinic.enter_time >= G.warmup_period, Patient.collection))

        # collected is a list of patients where each has queues with enter, leave, and waiting times as well as activities times
        # we want to make a dataframe where each row is a patient. Index is patient id and columns are the queue and activity times
        data = {
            'id': [p.id for p in collected],
            'reception_enter': [p.q_reception.enter_time for p in collected],
            'reception_leave': [p.q_reception.leave_time for p in collected],
            'reception_waiting': [p.q_reception.waiting_time for p in collected],
            'reception_activity': [p.a_reception_time for p in collected],
            'gp_enter': [p.q_gp.enter_time for p in collected],
            'gp_leave': [p.q_gp.leave_time for p in collected],
            'gp_waiting': [p.q_gp.waiting_time for p in collected],
            'gp_activity': [p.a_gp_time for p in collected],
            'book_enter': [p.q_book.enter_time for p in collected],
            'book_leave': [p.q_book.leave_time for p in collected],
            'book_waiting': [p.q_book.waiting_time for p in collected],
            'book_activity': [p.a_book_time for p in collected],
            'clinic_enter': [p.clinic.enter_time for p in collected],
            'clinic_leave': [p.clinic.leave_time for p in collected],
            'clinic_waiting': [p.clinic.waiting_time for p in collected],
        }

        return pd.DataFrame(data).set_index('id')

class Model:
    def __init__(self) -> None:
        Patient.collection.clear()
        self.env = simpy.Environment()

        self.receptionist = simpy.Resource(self.env, capacity=G.num_receptionists)
        self.gps = simpy.Resource(self.env, G.num_gps)

    def run(self):
        self.env.process(self.patients_generator())
        self.env.process(self.receptionist_calls_generator())

        self.env.run(until=(G.warmup_period + G.results_collection_period))

    def receptionist_calls_generator(self):
        while True:
            t = random.expovariate(1.0 / G.call_iat)
            yield self.env.timeout(t)
            self.env.process(self.call_activity_generator())
            
    def call_activity_generator(self):
        with self.receptionist.request() as req:
            yield req
            t = random.expovariate(1.0 / G.call_mt)
            yield self.env.timeout(t)

    def patients_generator(self):
        while True:
            self.env.process(self.patient_activity_generator(Patient(self.env)))
            t = random.expovariate(1.0 / G.patient_iat)
            yield self.env.timeout(t)

    def patient_activity_generator(self, patient: Patient):
        patient.clinic.enter()

        # Reception Activity
        patient.q_reception.enter()
        with self.receptionist.request() as req:
            yield req
            patient.q_reception.leave()
            t = random.expovariate(1.0 / G.registration_mt)
            yield self.env.timeout(t)
            patient.a_reception_time = t
        
        # GP Activity
        patient.q_gp.enter()
        with self.gps.request() as req:
            yield req
            patient.q_gp.leave()
            t = random.expovariate(1.0 / G.gp_consult_mt)
            yield self.env.timeout(t)
            patient.a_gp_time = t

        # Test Booking Activity
        decide_booking_branch = random.uniform(0, 1)
        if decide_booking_branch < G.booking_test_prob:
            patient.q_book.enter()
            with self.receptionist.request() as req:
                yield req
                patient.q_book.leave()
                t = random.expovariate(1.0 / G.booking_mt)
                yield self.env.timeout(t)
                patient.a_book_time = t

        patient.clinic.leave()

if __name__ == "__main__":
    main()