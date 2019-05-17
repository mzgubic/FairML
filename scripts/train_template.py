#!/usr/bin/python3
"""
Tensorflow template for Simple hyperparameter optimization using the Ray Tune class based API.

Manual tuning will get you 90% of the way there. Tune hyperparameters to
bleed out the last 5-10% from your model.

For more information about scheduling trials:
    Population-Based Training: https://arxiv.org/abs/1711.09846
    Hyperband Variants: https://arxiv.org/abs/1810.05934
Planned work:
    Bayesian Optimizaton / Hyperband: https://arxiv.org/abs/1807.01774

January 2019
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse
import functools

import ray
import ray.tune as tune

# User-defined
from network import Network
from diagnostics import Diagnostics
from data import Data
from tune_model import Model
from config import config_train, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def trial_str_creator(trial, name):
    return "{}_{}_{}".format(trial.trainable_name, trial.trial_id, name)

def get_best_trial(trials, metric):
     """Retrieve the best trial."""
     return max(trials, key=lambda trial: trial.last_result.get(metric, 0))

class TrainModel(tune.Trainable):

    def _retrieve_objects(self, object_ids):
        return [tune.util.get_pinned_object(object_id) for object_id in object_ids]

    def _initialize_train_iterator(self, sess, model, features, labels):
        sess.run(model.train_iterator.initializer, feed_dict={
            model.tokens_placeholder: features,
            model.labels_placeholder: labels})

    def _setup(self, config):
        """
        args/user_config: User-controlled hyperparameters, that will be held constant
        config = tune_config: parameters to be optimized
        """
        self.args, self.user_config = config.pop("args"), config.pop("user_config")
        train_object_ids, test_object_ids = config.pop("train_ids"), config.pop("test_ids")
        self.tune_config = config

        # Model initialization
        self.start_time = time.time()
        self.train_blocks = 0
        self.global_step, self.n_checkpoints, self.metric_best = 0, 0, 0.

        print("Retrieving data ...")
        self.features, self.labels = self._retrieve_objects(train_object_ids)
        test_features, test_labels = self._retrieve_objects(test_object_ids)

        # Build graph
        """
        Change model definition
        """
        self.model = Model(config=self.user_config, tune_config=self.tune_config, 
            directories=directories, features=self.features, 
            labels=self.labels, args=self.args, **kwargs)
        self.saver = tf.train.Saver()

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
            log_device_placement=False))
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.train_handle = self.sess.run(self.model.train_iterator.string_handle())
        self.test_handle = self.sess.run(self.model.test_iterator.string_handle())

        self.sess.run(self.model.test_iterator.initializer, feed_dict={
            self.model.test_tokens_placeholder: test_features,
            self.model.test_labels_placeholder: test_features})

        self._initialize_train_iterator(self.sess, self.model, self.features, self.labels)

    def _train(self):
        """
        Run one logical block of training
        Number of iterations should balance overhead with diagnostic frequency
        """
        block_iterations = 128  # capped by equivalent number of epochs

        for i in range(block_iterations):
            try:
                global_step, *ops = self.sess.run([self.model.train_op, 
                    self.model.update_accuracy], feed_dict={self.model.training_phase: True,
                    self.model.handle: self.train_handle})

            except tf.errors.OutOfRangeError:
                print('End of epoch.')
                break

        # Calculate metrics - careful with overhead
        self.train_blocks += 1
        self._initialize_train_iterator(self.sess, self.model, self.features, self.labels)
        
        """
        Evaluate validation metrics
        v_acc, v_loss, v_reward = metrics(...)
        """

        # Reported on validation set
        metrics = {
            "episode_reward_mean": v_reward,
            "mean_loss": v_loss,
            "mean_accuracy": v_acc
        }

        return metrics

    def _save(self, checkpoint_dir):
        # Save weights to checkpoint $save_path
        save_path = os.path.join(checkpoint_dir, "save")
        # save_path = os.path.join(checkpoint_dir, 'model_{}'.format(self.args.name))
        print('Saving to checkpoint {}'.format(save_path))
        target = self.saver.save(self.sess, save_path, global_step=self.train_blocks)
        return target

    def _restore(self, checkpoint_path):
        # Restore from checkpoint $path
        print('Loading from checkpoint {}'.format(checkpoint_path))
        return self.saver.restore(self.sess, checkpoint_path)


def main(**kwargs):

    print('===> Tuning hyperparameters. For normal training use `train.py`')
    print('===> TensorFlow v. {}'.format(tf.__version__))
    
    if args.max_time:
        print('Tuning process will terminate after {} seconds'.format(str(args.max_time)))
        os.environ['TRIALRUNNER_WALLTIME_LIMIT'] = str(args.max_time)

    n_gpus = 2
    if args.gpu_id is not None:
        print('Restricting visible GPUs.')
        print('Using: GPU {}'.format(args.gpu_id))
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
        n_gpus = 1
    
    ray.init(num_gpus=n_gpus)

    features_id, labels_id = Data.load_data(directories.train, tune=True)
    test_features_id, test_labels_id = Data.load_data(directories.test, tune=True)

    """
    Adjust stopping criteria - this terminates individual trials
    """
    stopping_criteria = {  # For individual trials
        "time_total_s": args.max_time_s,
        "episode_reward_mean": 1.0  # Otherwise negative loss
        "mean_accuracy": 1.0
    }

    # Hyperparameters to be optimized
    # Important to define search space sensibly
    config = {
        "args": args,
        "user_config": config_train,
        "train_ids": [features_id, labels_id],
        "test_ids": [test_features_id, test_labels_id],
    }

    hp_config = {
        """
        Include hyperparameters to be tuned, and their permitted range
        """
    }

    config.update(hp_config)

    hp_resamples_pbt = {
        """
        Include perturbation/resample ranges for population-based training
        """
    }

    # Specify experiment configuration
    # Default uses machine with 32 CPUs / 2 GPUs
    """
    Params to modify:
        num_samples (grid fineness)
        checkpoint_freq
        time_attr
        reward_attr
    """

    experiment_spec = tune.Experiment(
        name='tune_opt',
        run=TrainModel,
        stop=stopping_criteria,
        config=config,
        resources_per_trial={'cpu': 8, 'gpu': 0.5},
        num_samples=16,
        local_dir='~/ray_results',
        checkpoint_freq=8,
        checkpoint_at_end=True,
        trial_name_creator=tune.function(functools.partial(trial_str_creator, 
            name=args.name))
    )

    pbt = tune.schedulers.PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="episode_reward_mean",
        perturbation_interval=8,  # Mutation interval in time_attr units
        hyperparam_mutations=hp_resamples_pbt,
        resample_probability=0.25  # Resampling resets to value sampled from lambda
    )

    ahb = tune.schedulers.AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="episode_reward_mean",
        max_t=64,
        grace_period=8,
        reduction_factor=3,
        brackets=3
    )

    scheduler = ahb
    if args.pbt is True:
        scheduler = pbt

    trials = tune.run_experiments(
        experiments=experiment_spec,
        scheduler=scheduler,
        resume=False  # "prompt"
    )

    # Save results 
    t_ids = [t.trial_id for t in trials]
    t_config = [t.config for t in trials]
    t_result = [t.last_result for t in trials]
    df = pd.DataFrame([t_ids, t_config, t_result]).transpose()
    df.columns = ['name', 'config', 'result']
    df.to_hdf('{}_results.h5'.format(args.name), key='df')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-opt", "--optimizer", default="adam", help="Selected optimizer", type=str)
    parser.add_argument("-n", "--name", default="text_clf_tune", help="Checkpoint/Tensorboard label")
    parser.add_argument("-t", "--max_time_s", default=3600, type=int, help="Maximum time before termination in seconds")
    parser.add_argument("-gpu_id", "--gpu_id", help="Which GPU to use, indexed according to nvidia-smi", type=int, choices=set((0,1)), default=None)
    parser.add_argument("-pbt", "--pbt", help="Use population based training scheduler", action="store_true")
    parser.add_argument("-mt", "--max_time", help="Time limit for tuning process - defaults to none.", type=int, default=None)
    args = parser.parse_args()

    main(kwargs=args)
