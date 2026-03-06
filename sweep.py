import os
import time
import subprocess
from absl import app, flags

import numpy as np


def check_status(run_dir):
    run_status = None
    status_file = os.path.join(run_dir, 'status.txt')
    if os.path.exists(status_file):
        with open(status_file, 'r') as file:
            run_status = file.read().strip()
    return run_status

def slurm_job_count():
    return len(subprocess.check_output(['squeue', '-h']).decode().strip().split('\n'))


def indent_lines(lines):
    lines = '\n'.join(lines).split('\n')
    return '\n'.join([('  ' + line) for line in lines])


class Sweep:
    def __init__(self, config):
        self.config = config
        self.size = None

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        raise NotImplementedError

    def __iter__(self):
        for i in range(self.size):
            yield self[i]

    def __repr__(self):
        if isinstance(self.config, dict):
            config_repr = '\n'.join([
                '{',
                indent_lines([
                    repr(k) + ': ' + repr(v) + ','
                    for k, v in self.config.items()
                ]),
                '}',
            ])
        else:
            config_repr = '\n'.join([
                '[',
                indent_lines([
                    repr(c) + ','
                    for c in self.config
                ]),
                ']',
            ])
        return f'{self.__class__.__name__}({config_repr})'


class Singleton(Sweep):
    def __init__(self, config):
        super().__init__(config)
        self.size = 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        assert idx == 0
        return self.config

    def __iter__(self):
        for i in range(self.size):
            yield self[i]


class Parallel(Sweep):
    def __init__(self, config):
        super().__init__(config)
        sizes = [len(v) for v in config.values()]
        assert all([n == sizes[0] for n in sizes[1:]])
        self.size = sizes[0]
        self.config = config

    def __getitem__(self, idx):
        assert idx >= 0 and idx < self.size
        return {k: v[idx] for k, v in self.config.items()}


class Concat(Sweep):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if isinstance(self.config, dict):
            self.keys = list(self.config.keys())
            self.cumulative_size = np.cumsum([0] + [len(self.config[k]) for k in self.keys])
        else:
            self.cumulative_size = np.cumsum([0] + [len(c) for c in self.config])
        self.size = self.cumulative_size[-1]

    def __getitem__(self, idx):
        assert idx >= 0 and idx < self.size
        i1 = np.searchsorted(self.cumulative_size, idx, side='right') - 1
        i2 = idx - self.cumulative_size[i1]
        if isinstance(self.config, dict):
            res = self.config[self.keys[i1]][i2]
        else:
            res = self.config[i1][i2]
        return res


class Product(Sweep):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if isinstance(self.config, dict):
            self.keys = list(self.config.keys())
            self.sizes = np.array([len(self.config[k]) for k in self.keys])
        else:
            self.sizes = np.array([len(c) for c in self.config])
        self.size = np.prod(self.sizes)

    def __getitem__(self, idx):
        assert idx >= 0 and idx < self.size
        res = {}
        if isinstance(self.config, dict):
            for k, i in zip(self.keys, np.unravel_index(idx, self.sizes)):
                res[k] = self.config[k][i]
        else:
            for c, i in zip(self.config, np.unravel_index(idx, self.sizes)):
                res.update(c[i])
        return res


FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'sweep.txt', 'Sweep configuration file')
flags.DEFINE_string('template', 'slurm-template.txt', 'Template script file')
flags.DEFINE_string('command', 'sbatch', 'command to execute run scripts')
flags.DEFINE_string('base_dir', 'exp/sweep/', 'Base save directory')
flags.DEFINE_integer('max_jobs', 50, 'Maximum number of parallel jobs')


def main(_):
    print('Reading sweep configuration')
    with open(FLAGS.config, 'r') as file:
        sweep = eval(file.read())

    print('Reading template')
    with open(FLAGS.template, 'r') as file:
        template = file.read()

    print('Making sweep base directory')
    os.makedirs(FLAGS.base_dir, exist_ok=True)

    with open(os.path.join(FLAGS.base_dir, 'sweep.txt'), 'w') as file:
        file.write(repr(sweep) + '\n')

    git_log = subprocess.check_output(['git', 'log', '-1'])
    with open(os.path.join(FLAGS.base_dir, 'log.txt'), 'ab') as file:
        file.write(
            time.ctime().encode() + b'\n'
            + b'### git info ###\n'
            + git_log
            + b'###\n\n'
            + b'### sweep config ###\n'
            + repr(sweep).encode() + b'\n'
            + b'###\n\n'
        )

    for idx, config in enumerate(sweep):
        run_dir = os.path.abspath(os.path.join(FLAGS.base_dir, 'runs', str(idx)))
        os.makedirs(run_dir, exist_ok=True)

        if check_status(run_dir) in ('succeeded', 'running'):
            print(f'Skipping run {idx}')
            continue

        script = template.format(
            run_dir=run_dir,
            args=' \\\n\t'.join([
                f'--{k} {repr(v)}'
                for k, v in config.items()
            ])
        )

        script_file = os.path.join(run_dir, 'job.sh')
        with open(script_file, 'w') as file:
            file.write(script)

        while slurm_job_count() >= FLAGS.max_jobs:
            time.sleep(600)

        print(f'Submitting run {idx}')
        command_output = subprocess.check_output([FLAGS.command, script_file])
        with open(os.path.join(FLAGS.base_dir, 'log.txt'), 'ab') as file:
            file.write(
                f'Run {idx}\n'.encode()
                + command_output
                + b'\n'
            )


if __name__ == '__main__':
    app.run(main)
