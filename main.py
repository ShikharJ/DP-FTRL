# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DP-FTRL training, based on paper
"Practical and Private (Deep) Learning without Sampling or Shuffling"
https://arxiv.org/abs/2103.00039.
"""

from absl import app
from absl import flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import trange
import numpy as np

import tensorflow as tf
import torch
from torch.utils.tensorboard import SummaryWriter
from opacus import PrivacyEngine

from optimizers import FTRLOptimizer
from ftrl_noise import CummuNoiseTorch, CummuNoiseEffTorch
from nn import get_nn
from data import get_data
import utils
from utils import EasyDict
import random
from evaluator import get_eps_audit


FLAGS = flags.FLAGS

flags.DEFINE_enum('data', 'mnist', ['mnist', 'cifar10', 'emnist_merge'], '')

flags.DEFINE_boolean('dp_ftrl', True, 'If True, train with DP-FTRL. If False, train with vanilla FTRL.')
flags.DEFINE_float('noise_multiplier', 4.0, 'Ratio of the standard deviation to the clipping norm.')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm.')

flags.DEFINE_integer('restart', 0, 'If > 0, restart the tree every this number of epoch(s).')
flags.DEFINE_boolean('effi_noise', False, 'If True, use tree aggregation proposed in https://privacytools.seas.harvard.edu/files/privacytools/files/honaker.pdf.')
flags.DEFINE_boolean('tree_completion', False, 'If true, generate until reaching a power of 2.')

flags.DEFINE_float('momentum', 0, 'Momentum for DP-FTRL.')
flags.DEFINE_float('learning_rate', 0.4, 'Learning rate.')
flags.DEFINE_integer('batch_size', 250, 'Batch size.')
flags.DEFINE_integer('epochs', 3, 'Number of epochs.')

flags.DEFINE_integer('report_nimg', -1, 'Write to tb every this number of samples. If -1, write every epoch.')

flags.DEFINE_integer('run', 1, '(run-1) will be used for random seed.')
flags.DEFINE_string('dir', '.', 'Directory to write the results.')
flags.DEFINE_integer('m', 10000, 'Number of canaries.')
flags.DEFINE_integer('kin', 100, 'Number of members guesses.')
flags.DEFINE_integer('kout', 100, 'Number of non-member guesses.')
flags.DEFINE_integer('limit_train', -1, 'If not -1, limit training set size to the number of canaries.')
flags.DEFINE_float('p', 0.1, 'Centum minus confidence.')
flags.DEFINE_float('delta', 0.00001, 'Delta for DP-guarantee.')
flags.DEFINE_boolean('black_box', True, 'If true, run with black box auditing process.')


def main(argv):
    tf.get_logger().setLevel('ERROR')
    tf.config.experimental.set_visible_devices([], "GPU")

    # Setup random seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(FLAGS.run - 1)
    np.random.seed(FLAGS.run - 1)

    # Data
    trainset, testset, ntrain, nclass = get_data(FLAGS.data)
    if FLAGS.limit_train != -1:
        ntrain = FLAGS.m
    print('Training set size', trainset.image.shape)

    # Hyperparameters for training.
    epochs = FLAGS.epochs
    batch = FLAGS.batch_size if FLAGS.batch_size > 0 else ntrain - (FLAGS.m // 2)
    num_batches = (ntrain - (FLAGS.m // 2)) // batch
    noise_multiplier = FLAGS.noise_multiplier if FLAGS.dp_ftrl else -1
    clip = FLAGS.l2_norm_clip if FLAGS.dp_ftrl else -1
    lr = FLAGS.learning_rate
    if not FLAGS.restart:
        FLAGS.tree_completion = False

    report_nimg = (ntrain - (FLAGS.m // 2)) if FLAGS.report_nimg == -1 else FLAGS.report_nimg
    assert report_nimg % batch == 0

    # Get the name of the output directory.
    log_dir = os.path.join(FLAGS.dir, FLAGS.data,
                           utils.get_fn(EasyDict(batch=batch),
                                        EasyDict(dpsgd=FLAGS.dp_ftrl, restart=FLAGS.restart, completion=FLAGS.tree_completion, noise=noise_multiplier, clip=clip, mb=1),
                                        [EasyDict({'lr': lr}),
                                         EasyDict(m=FLAGS.momentum if FLAGS.momentum > 0 else None,
                                                  effi=FLAGS.effi_noise),
                                         EasyDict(sd=FLAGS.run)]
                                        )
                           )
    print('Model dir', log_dir)
    zero_level_losses = []
    final_level_losses = []

    # Class to output batches of data
    class DataStream:
        def __init__(self, choices):
            self.choices = choices
            self.shuffle(0)

        def shuffle(self, shuffle_counter):
            self.perm = np.random.permutation(ntrain)
            self.i = 0
            if shuffle_counter == 0:
                self.include = []
                self.exclude = []
                self.inclusion = [1] * FLAGS.m
                for i in range(FLAGS.m):
                    self.inclusion[i] = random.choice(self.choices)
                for i, indexed_value in enumerate(self.perm):
                    if i == FLAGS.m:
                        break
                    if self.inclusion[i] == 0:
                        self.exclude.append(indexed_value)
                    else:
                        self.include.append(indexed_value)
            self.batch_idxs = [x for x in self.perm if x not in self.exclude]

        def __call__(self):
            if self.i == num_batches:
                self.i = 0
            batch_idx = self.batch_idxs[self.i * batch:(self.i + 1) * batch]
            self.i += 1
            return trainset.image[batch_idx], trainset.label[batch_idx]
    
        def get_inclusions_exclusions(self):
            return self.exclude, self.include

        def get_data_at_index(self, index):
            return [trainset.image[index]], [trainset.label[index]]

    data_stream = DataStream(choices = [0, 1])

    # Function to conduct training for one epoch
    def train_loop(model, device, optimizer, cumm_noise, epoch, writer, counter):
        model.train()
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        losses = []
        loop = trange(0, num_batches * batch, batch,
                      leave=False, unit='img', unit_scale=batch,
                      desc='Epoch %d/%d' % (1 + epoch, epochs))
        step = epoch * num_batches
        for it in loop:
            step += 1
            data, target = data_stream()
            data = torch.Tensor(data).to(device)
            target = torch.LongTensor(target).to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step((lr, cumm_noise()))
            losses.append(loss.item())

            if (step * batch) % report_nimg == 0:
                acc_train, acc_test = test(model, device)
                writer.add_scalar('eval/accuracy_test', 100 * acc_test, step)
                writer.add_scalar('eval/accuracy_train', 100 * acc_train, step)
                if counter == 0:
                    print("First Level")
                    test_under_audit(model, device, criterion, zero_level_losses)
                if counter == 1:
                    print("Final Level")
                    test_under_audit(model, device, criterion, final_level_losses)
                model.train()
                print('Step %04d Accuracy %.2f' % (step, 100 * acc_test))

        writer.add_scalar('eval/loss_train', np.mean(losses), epoch + 1)
        print('Epoch %04d Loss %.2f' % (epoch + 1, np.mean(losses)))

    # Function for evaluating the model to get training and test accuracies
    def test(model, device, desc='Evaluating'):
        model.eval()
        b = 1000
        with torch.no_grad():
            accs = [0, 0]
            for i, dataset in enumerate([trainset, testset]):
                for it in trange(0, dataset.image.shape[0], b, leave=False, desc=desc):
                    data, target = dataset.image[it: it + b], dataset.label[it: it + b]
                    data, target = torch.Tensor(data).to(device), torch.LongTensor(target).to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    accs[i] += pred.eq(target.view_as(pred)).sum().item()
                accs[i] /= dataset.image.shape[0]
        return accs
    
    # Function for evaluating the model to get training and test accuracies
    def test_under_audit(model, device, criterion, loss_vector, desc='Evaluating'):
        model.eval()
        with torch.no_grad():
            excl_ids, incl_ids = data_stream.get_inclusions_exclusions()
            for iter in range(len(excl_ids)):
                data, target = data_stream.get_data_at_index(excl_ids[iter])
                data = torch.Tensor(data).to(device)
                target = torch.LongTensor(target).to(device)

                output = model(data)
                loss = criterion(output, target)
                loss_vector.append(loss.item())
            for iter in range(len(incl_ids)):
                data, target = data_stream.get_data_at_index(incl_ids[iter])
                data = torch.Tensor(data).to(device)
                target = torch.LongTensor(target).to(device)

                output = model(data)
                loss = criterion(output, target)
                loss_vector.append(loss.item())

    # Function for auditing the trained model given the inclusion and exclusion ids.
    def get_scores(indices, kin, kout):
        correct = 0
        incorrect = 0
        for i in range(kin):
            index = indices[i]
            if index >= FLAGS.m // 2:
                correct += 1
            else:
                incorrect += 1
        for i in range(kout):
            index = indices[-(i + 1)]
            if index < FLAGS.m // 2:
                correct += 1
            else:
                incorrect += 1
        assert correct + incorrect == kin + kout
        return kin + kout, correct

    # Get model for different dataset
    device = torch.device('cpu')
    model = get_nn({'mnist': 'small_nn',
                    'emnist_merge': 'small_nn',
                    'cifar10': 'vgg128'}[FLAGS.data],
                   nclass=nclass).to(device)

    # Set the (DP-)FTRL optimizer. For DP-FTRL, we
    # 1) use the opacus library to conduct gradient clipping without adding noise
    # (so we set noise_multiplier=0). Also we set alphas=[] as we don't need its
    # privacy analysis.
    # 2) use the CummuNoise module to generate the noise using the tree aggregation
    # protocol. The noise will be passed to the FTRL optimizer.
    optimizer = FTRLOptimizer(model.parameters(), momentum=FLAGS.momentum,
                              record_last_noise=FLAGS.restart > 0 and FLAGS.tree_completion)
    if FLAGS.dp_ftrl:
        privacy_engine = PrivacyEngine(model, batch_size=batch, sample_size=(ntrain - (FLAGS.m // 2)), alphas=[], noise_multiplier=0, max_grad_norm=clip)
        privacy_engine.attach(optimizer)
    shapes = [p.shape for p in model.parameters()]

    def get_cumm_noise(effi_noise):
        if FLAGS.dp_ftrl == False or noise_multiplier == 0:
            return lambda: [torch.Tensor([0]).to(device)] * len(shapes)  # just return scalar 0
        if not effi_noise:
            cumm_noise = CummuNoiseTorch(noise_multiplier * clip / batch, shapes, device)
        else:
            cumm_noise = CummuNoiseEffTorch(noise_multiplier * clip / batch, shapes, device)
        return cumm_noise

    cumm_noise = get_cumm_noise(FLAGS.effi_noise)

    # The training loop.
    writer = SummaryWriter(os.path.join(log_dir, 'tb'))
    for epoch in range(epochs):
        if epoch == 0:
            train_loop(model, device, optimizer, cumm_noise, epoch, writer, 0)
        elif epoch == epochs - 1:
            train_loop(model, device, optimizer, cumm_noise, epoch, writer, 1)
        else:
            train_loop(model, device, optimizer, cumm_noise, epoch, writer, -1)

        if epoch + 1 == epochs:
            break
        restart_now = epoch < epochs - 1 and FLAGS.restart > 0 and (epoch + 1) % FLAGS.restart == 0
        if restart_now:
            last_noise = None
            if FLAGS.tree_completion:
                actual_steps = num_batches * FLAGS.restart
                next_pow_2 = 2**(actual_steps - 1).bit_length()
                if next_pow_2 > actual_steps:
                    last_noise = cumm_noise.proceed_until(next_pow_2)
            optimizer.restart(last_noise)
            cumm_noise = get_cumm_noise(FLAGS.effi_noise)
            data_stream.shuffle(1)  # shuffle the data only when restart
    writer.close()
    scores = []
    if FLAGS.black_box == True:
        # Calculate scores
        emp_eps = []
        ah_counter = 0
        for i in range(FLAGS.m):
            if zero_level_losses[i] < final_level_losses[i]:
                ah_counter += 1
            scores.append(zero_level_losses[i] - final_level_losses[i])
        indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        r, v = get_scores(indices, FLAGS.kin, FLAGS.kout)
        print(FLAGS.m, r, v, FLAGS.delta, FLAGS.p, ah_counter)
        eps_value = get_eps_audit(FLAGS.m, r, v, FLAGS.delta, FLAGS.p)
        emp_eps.append(eps_value)
        print(emp_eps)
    else:
        pass

if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)
