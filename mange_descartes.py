import matplotlib
matplotlib.use('Agg')
import os


import neptune
import gin
import gin.tf
from absl import flags, app

import feature_classifier

@gin.configurable
def descartes_builder(name='out', params=[]):
    print(params)

    directory = os.path.join('grids', name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    i = 0
    for param in params:
        values = gin.query_parameter(param)
        for val in values:
            exp_directory = os.path.join(directory, str(i))
            if not os.path.exists(exp_directory):
                os.makedirs(exp_directory)

            with gin.unlock_config():
                gin.bind_parameter(param, val)
            config_str = gin.config_str()
            with open(os.path.join(exp_directory, 'config.gin'), 'w+') as f:
                f.write(config_str)
            i += 1
    pass


def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)
    descartes_builder()

if __name__ == '__main__':
    flags.DEFINE_multi_string('gin_file', None, 'List of paths to the config files.')
    flags.DEFINE_multi_string('gin_param', None, 'Newline separated list of Gin parameter bindings.')
    FLAGS = flags.FLAGS

    app.run(main)
