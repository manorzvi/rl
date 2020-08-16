import os
import pprint
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from google.cloud import storage
import gym

from utils import get_state, get_config
from StackedStates import StackedStates
from DDDQN import DDDQN


class DDDQN_GCP(DDDQN):

    def __init__(self, h, w, n_action, device, env_id, loss_func, optimizer_func,
                 exp_rep_capacity=100000, exp_rep_pretrain_size=100000,
                 batch_size=64, episodes=2000, target_update_interval=10, save_model_interval=100,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.00001, lr=0.00025, gamma=0.99,
                 logs_dir='logs', ckpt_dir='models', bucket_name : str = None, service_account_key_file : str = None):

        assert bucket_name is not None, "Please provide a new name for the Storage Bucket, or an existing one"
        assert service_account_key_file is not None, "Please provide a Service-Account key-file name"

        super(DDDQN_GCP, self).__init__(h, w, n_action, device, env_id, loss_func, optimizer_func, exp_rep_capacity,
                                        exp_rep_pretrain_size, batch_size, episodes, target_update_interval,
                                        save_model_interval, eps_start, eps_end, eps_decay, lr, gamma, logs_dir,
                                        ckpt_dir)

        # Explicitly use service account credentials by specifying the private key file.
        storage_client = storage.Client.from_service_account_json(service_account_key_file)

        buckets = list(storage_client.list_buckets())

        print(f'[I] - Search for existing bucket named {bucket_name} ... ', end='')
        for bucket in buckets:
            if bucket.name == bucket_name:
                print('Found! ', end='')
                self.bucket = storage_client.bucket(bucket_name)
                break
        print('Done.')
        if not hasattr(self, 'bucket'):
            print("[I] - Existing bucket hasn't found. Create a new Bucket ... ", end='')
            self.bucket = storage_client.create_bucket(bucket_name)
            print('Done.')
        print('[I] - Bucket metadata:\n'
              '----------------------')
        self.bucket_metadata()

    def bucket_metadata(self):
        """Prints out a bucket's metadata."""

        print("ID:                          {}".format(self.bucket.id))
        print("Name:                        {}".format(self.bucket.name))
        print("Storage Class:               {}".format(self.bucket.storage_class))
        print("Location:                    {}".format(self.bucket.location))
        print("Location Type:               {}".format(self.bucket.location_type))
        print("Cors:                        {}".format(self.bucket.cors))
        print("Default Event Based Hold:    {}".format(self.bucket.default_event_based_hold))
        print("Default KMS Key Name:        {}".format(self.bucket.default_kms_key_name))
        print("Metageneration:              {}".format(self.bucket.metageneration))
        print("Retention Effective Time:    {}".format(self.bucket.retention_policy_effective_time))
        print("Retention Period:            {}".format(self.bucket.retention_period))
        print("Retention Policy Locked:     {}".format(self.bucket.retention_policy_locked))
        print("Requester Pays:              {}".format(self.bucket.requester_pays))
        print("Self Link:                   {}".format(self.bucket.self_link))
        print("Time Created:                {}".format(self.bucket.time_created))
        print("Versioning Enabled:          {}".format(self.bucket.versioning_enabled))
        print("Labels:")
        pprint.pprint(self.bucket.labels)

    def upload_model_to_bucket(self, model_name):
        print(f'[I] - Upload {model_name} to bucket ... ', end='')
        blob = self.bucket.blob(model_name)
        blob.upload_from_filename(model_name)
        print('Done.')

    def upload_tensorboard_to_bucket(self):
        tensorboard_file = self.writer.file_writer.event_writer._file_name
        print(f'[I] - Upload {tensorboard_file} to bucket ... ', end='')
        blob = self.bucket.blob(tensorboard_file)
        blob.upload_from_filename(tensorboard_file)
        print('Done.')

    def save(self, **kwargs):
        model_name = super(DDDQN_GCP, self).save(**kwargs)
        self.upload_model_to_bucket(model_name)
        self.upload_tensorboard_to_bucket()


if __name__ == '__main__':
    parser = get_config()
    parser.add_argument('-bucket_name', type=str, required=True)
    parser.add_argument('-service_account_key_file', type=str, required=True)
    args = parser.parse_args()
    print(args)

    env = gym.make(args.env_id)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    print(f'env: {env}, action_space: {env.action_space}, observation_space: {env.observation_space}')

    n_action = env.action_space.n
    print("Action Space Size: ", n_action)

    stackedstates = StackedStates()
    env.reset()
    init_state = get_state(env, stackedstates, device)
    _, _, screen_height, screen_width = init_state.shape

    loss_func = F.smooth_l1_loss
    optimizer_func = optim.Adam

    bucket_name = f'{args.env_id}-bucket'

    model = DDDQN_GCP(h=screen_height, w=screen_width, n_action=n_action, device=device, env_id=args.env_id,
                  loss_func=loss_func, optimizer_func=optimizer_func, exp_rep_capacity=args.experience_replay_capacity,
                  exp_rep_pretrain_size=args.experience_replay_pretrain_size,
                  batch_size=args.batch_size, episodes=args.episodes_number,
                  target_update_interval=args.target_update_interval, save_model_interval=args.save_model_interval,
                  eps_start=args.epsilon_start, eps_end=args.epsilon_end, eps_decay=args.epsilon_decay,
                  lr=args.learning_rate, gamma=args.gamma, logs_dir=args.logs, ckpt_dir=args.models,
                      bucket_name=args.bucket_name, service_account_key_file=args.service_account_key_file)

    if args.load:
        model.load(path=args.path)

    print(model)

    if args.train:
        model.exp_rep_pretrain(env)

        model.train(env)

    if args.play:
        model.play(env)
