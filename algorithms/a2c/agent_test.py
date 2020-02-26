from agent import A2cAgent

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v0')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--timesteps_max', type=int, default=300)
    parser.add_argument('--number_of_batches', type=int, default=1200)
    parser.add_argument('--discount_factor', type=float, default=0.1)
    parser.add_argument('--entropy_constant', type=float, default = 0.001)
    parser.add_argument('--minimum_batch_size', type=int, default = 200)

    args = parser.parse_args()

    A2cAgent(args.env_name, args.learning_rate, args.timesteps_max, args.number_of_batches, args.discount_factor,
             args.entropy_constant, args.minimum_batch_size).train()