import os
import subprocess
from absl import app, flags

import numpy as np


def indent_lines(lines):
    lines = '\n'.join(lines).split('\n')
    return '\n'.join([('  ' + line) for line in lines])

def format_list(l):
    return '\n'.join([
        '[',
        indent_lines([
            repr(f) + ','
            for f in l
        ]),
        ']',
    ])

def format_dict(d):
    return '\n'.join([
        '{',
        indent_lines([
            repr(k) + ': ' + repr(v) + ','
            for k, v in d.items()
        ]),
        '}',
    ])


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
        return (
            self.__class__.__name__
            + '('
            + (
                format_dict(self.config)
                if isinstance(self.config, dict) else
                format_list(self.config)
            )
            + ')')


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
            self.cumulative_size = np.cumsum([0] + [len(f) for f in self.config])
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
            self.sizes = np.array([len(f) for f in self.config])
        self.size = np.prod(self.sizes)

    def __getitem__(self, idx):
        assert idx >= 0 and idx < self.size
        res = {}
        if isinstance(self.config, dict):
            for k, i in zip(self.keys, np.unravel_index(idx, self.sizes)):
                res[k] = self.config[k][i]
        else:
            for f, i in zip(self.config, np.unravel_index(idx, self.sizes)):
                res.update(f[i])
        return res


FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'sweep.txt', 'Sweep configuration file')
flags.DEFINE_string('template', 'slurm-template.txt', 'Template script file')
flags.DEFINE_string('command', 'sbatch', 'command to execute run scripts')
flags.DEFINE_string('base_dir', 'exp/sweep/', 'Base save directory')


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

    date = subprocess.check_output(['date'])
    git_log = subprocess.check_output(['git', 'log', '-1'])
    with open(os.path.join(FLAGS.base_dir, 'log.txt'), 'ab') as file:
        file.write(
            date + b'\n'
            + b'### git info ###\n'
            + git_log
            + b'###\n\n'
            + b'### sweep config ###\n'
            + repr(sweep).encode() + b'\n'
            + b'###\n\n'
        )

    for idx, config in enumerate(sweep):
        run_dir = os.path.join(FLAGS.base_dir, 'runs', str(idx))
        os.makedirs(run_dir, exist_ok=True)

        status_file = os.path.join(run_dir, 'status.txt')
        if os.path.exists(status_file):
            with open(status_file, 'r') as file:
                run_status = file.read()
            if 'failed' not in run_status:
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
