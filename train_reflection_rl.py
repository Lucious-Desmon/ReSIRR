import argparse
import os
import sys
import time
import numpy as np
import cv2

import MyFCN_el
import pixelwise_a3c_el
from State_reflect import StateReflect
from jsonl_paired_loader import (
    FuseDataset,
    JsonlBatchSampler,
    JsonlConcatDataset,
    JsonlPairedDataset,
    load_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description="RL dereflection training with jsonl datasets.")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--multiple_datasets", type=str, nargs="+", required=True)
    parser.add_argument("--multiple_datasets_probabilities", type=float, nargs="+", required=True)
    parser.add_argument("--test_data_dir", type=str, required=True)
    parser.add_argument("--test_datasets", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="./model")
    parser.add_argument("--result_dir", type=str, default="./result_reflect")
    parser.add_argument("--logging_dir", type=str, default="./logs_reflect_rl")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--resize_scale", type=float, default=1.1)
    parser.add_argument("--disable_augment", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--episode_len", type=int, default=5)
    parser.add_argument("--snapshot_episodes", type=int, default=500)
    parser.add_argument("--test_episodes", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n_actions", type=int, default=9)
    parser.add_argument("--move_step", type=float, default=0.005)
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


def compute_psnr(output, target):
    mse = np.mean((output.astype(np.float32) - target.astype(np.float32)) ** 2)
    if mse == 0:
        return 99.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def _get_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
    except Exception:
        try:
            from tensorboardX import SummaryWriter
            return SummaryWriter(log_dir=log_dir)
        except Exception:
            return None


def test(dataset, agent, fout, args, global_step, writer):
    sum_psnr = 0.0
    test_data_size = len(dataset)
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)

    for i in range(test_data_size):
        raw_x, label = dataset[i]
        raw_x = raw_x[np.newaxis, ...]
        label = label[np.newaxis, ...]
        h, w = raw_x.shape[2], raw_x.shape[3]
        current_state = StateReflect((1, 3, h, w), args.n_actions, args.move_step)
        current_state.reset(raw_x)

        for _ in range(args.episode_len):
            action = agent.act(current_state.image)
            current_state.step(action)
        agent.stop_episode()

        out_img = np.clip(current_state.image, 0, 1)
        gt_img = np.clip(label, 0, 1)

        out_u8 = (out_img * 255).astype(np.uint8)
        gt_u8 = (gt_img * 255).astype(np.uint8)
        psnr = compute_psnr(out_u8, gt_u8)
        sum_psnr += psnr

        out_save = np.transpose(out_u8[0], (1, 2, 0))
        cv2.imwrite(os.path.join(args.result_dir, str(i) + "_output.png"), out_save)

    avg_psnr = sum_psnr / test_data_size
    print("test PSNR {b}".format(b=avg_psnr))
    fout.write("test PSNR {b}\n".format(b=avg_psnr))
    if writer is not None:
        writer.add_scalar("val/psnr", avg_psnr, global_step)
    sys.stdout.flush()


def main(fout):
    args = parse_args()
    writer = _get_writer(args.logging_dir)

    if len(args.multiple_datasets) != len(args.multiple_datasets_probabilities):
        raise ValueError("multiple_datasets and probabilities must have same length.")

    train_datasets = []
    for name in args.multiple_datasets:
        jsonl_path = os.path.join(args.train_data_dir, name)
        entries = load_jsonl(jsonl_path)
        train_datasets.append(
            JsonlPairedDataset(entries, args.resolution, args.resize_scale, args.disable_augment)
        )

    fuse_dataset = FuseDataset(train_datasets, args.multiple_datasets_probabilities)
    sampler = JsonlBatchSampler(fuse_dataset, args.batch_size)

    test_datasets = []
    for name in args.test_datasets:
        jsonl_path = os.path.join(args.test_data_dir, name)
        entries = load_jsonl(jsonl_path)
        test_datasets.append(
            JsonlPairedDataset(entries, args.resolution, args.resize_scale, True, disable_resize=True)
        )
    test_dataset = JsonlConcatDataset(test_datasets)

    pixelwise_a3c_el.chainer.cuda.get_device_from_id(args.gpu_id).use()

    # load myfcn model
    model = MyFCN_el.MyFcn(args.n_actions)

    optimizer = pixelwise_a3c_el.chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(model)

    agent = pixelwise_a3c_el.PixelWiseA3C(model, optimizer, args.episode_len, args.gamma)
    agent.model.to_gpu()

    global_step = 0
    for episode in range(1, args.episodes + 1):
        print("episode %d" % episode)
        fout.write("episode %d\n" % episode)
        sys.stdout.flush()

        raw_x, label = sampler.sample_batch(args.resolution)
        current_state = StateReflect(
            (args.batch_size, 3, args.resolution, args.resolution),
            args.n_actions,
            args.move_step,
        )
        current_state.reset(raw_x)

        reward = np.zeros_like(raw_x, dtype=np.float32)
        sum_reward = 0.0
        sum_loss = 0.0

        for t in range(0, args.episode_len):
            action = agent.act_and_train(current_state.image, reward)
            current_state.step(action)
            abs_diff = np.abs(current_state.image - label).astype(np.float32)
            loss = np.mean(abs_diff).astype(np.float32)
            reward = -abs_diff
            sum_reward += float(np.mean(reward)) * np.power(args.gamma, t)
            sum_loss += loss

        agent.stop_episode_and_train(current_state.image, reward, True)
        global_step += 1

        avg_loss = sum_loss / float(args.episode_len)
        print("train total reward {a}".format(a=sum_reward))
        fout.write("train total reward {a}\n".format(a=sum_reward))
        sys.stdout.flush()

        if writer is not None:
            writer.add_scalar("train/reward", sum_reward, global_step)
            writer.add_scalar("train/l1", avg_loss, global_step)

        if episode % args.test_episodes == 0:
            test(test_dataset, agent, fout, args, global_step, writer)

        if episode % args.snapshot_episodes == 0:
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)
            agent.save(os.path.join(args.output_dir, "reflect_" + str(episode)))


if __name__ == "__main__":
    try:
        fout = open("log_reflect_rl.txt", "w")
        start = time.time()
        main(fout)
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start) / 60))
        print("{s}[h]".format(s=(end - start) / 60 / 60))
        fout.write("{s}[s]\n".format(s=end - start))
        fout.write("{s}[m]\n".format(s=(end - start) / 60))
        fout.write("{s}[h]\n".format(s=(end - start) / 60 / 60))
        fout.close()
    except Exception as error:
        print(str(error))
