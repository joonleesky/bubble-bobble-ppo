import subprocess
from multiprocessing import Pool

if __name__=='__main__':
    experiments = [
        {'--exp_name': 'baseline',
         '--param_name': 'BubbleBobble-Nes-Nature',
         '--gpu_device': '0'},
        {'--exp_name': 'shift',
         '--param_name': 'BubbleBobble-Nes-Nature-Shift',
         '--gpu_device': '1'},
        {'--exp_name': 'scale',
         '--param_name': 'BubbleBobble-Nes-Nature-Scale',
         '--gpu_device': '2'},
        {'--exp_name': 'shift_scale',
         '--param_name': 'BubbleBobble-Nes-Nature-Shift-Scale',
         '--gpu_device': '3'},
    ]
    def run_experiment(experiment):
        cmd = ['python', 'train.py']
        print(experiment)
        for key, value in experiment.items():
            cmd.append(key)
            cmd.append(value)
        return subprocess.call(cmd)

    pool = Pool(4)
    pool.map(run_experiment, experiments)
    pool.close()