import os
import subprocess
import argparse

def create_config(dim, beta, seed, dataname, epochs):
    config_name = '/hdd/dsvae/tcvae_models/{}/config/d_{}_b_{}.gin'.format(dataname, dim, beta)
    basefile = '/hdd/dsvae/tcvae_models/{}/config/base.gin'.format(dataname)
    lines = []
    file = open(basefile, 'r')
    for line in file: 
        lines.append(line)
    file.close()
    file = open(config_name, 'w+')
    for line in lines: 
        if line.startswith('beta_tc_vae.beta = '):#
            file.writelines(['beta_tc_vae.beta = {}.\n'.format(beta)]) #
        elif line.startswith('encoder.num_latent = '):
            file.writelines(['encoder.num_latent = {}\n'.format(dim)]) 
        elif line.startswith('model.random_seed='):
            file.writelines(['model.random_seed={}\n'.format(seed)]) 
        elif line.startswith('model.training_steps ='):
            file.writelines(['model.training_steps = {}\n'.format(epochs)]) 
        else:
            file.writelines([line]) 
    file.close()
    return config_name

def run_with_config(config_file, train_output_dir, eval_output_dir, device):
    train_command = ['python', 'dlib_train_gpu_{}.py', '--gin_config={}'.format(device, config_file), '--model_dir={}'.format(train_output_dir)]
    #eval_command = ['python', 'dlib_reproduce.py', '--model_dir={}'.format(train_output_dir), '--output_directory={}'.format(eval_output_dir)]
    print(train_command)
    with open('./exp_output/{}_train.out'.format(exp_name), 'w') as f:
        process = subprocess.Popen(train_command, stdout=f)
        exit_codes = process.wait()
        print(exit_codes)
    # no evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--dataname", help="dataset name", default="smallnorb", type=str)
parser.add_argument("--c_dim", help="c dimension", default=10, type=int)
parser.add_argument("--beta", help="beta", default=4, type=int)
parser.add_argument("--nb_epochs", help="number of epochs", default=300000, type=int)
parser.add_argument("--run_seed", help="run seed of TCVAE", default=0, type=int)
parser.add_argument("--device", help="device number", default=0, type=int)

args = parser.parse_args()

dataname = args.dataname      
dim = args.c_dim
beta = args.beta
seed = args.run_seed
epochs = args.nb_epochs

exp_name = 'd_{}_b_{}_{}'.format(dim, beta, seed) # first exp without repeative
print(exp_name)
config_file = create_config(dim, beta, seed, dataname, epochs)
train_output_dir = '/hdd/dsvae/tcvae_models/{}/train_output/{}'.format(dataname, exp_name)
eval_output_dir = '/hdd/dsvae/tcvae_models/{}/eval_output/{}'.format(dataname, exp_name)
run_with_config(config_file, train_output_dir, eval_output_dir, device)





        
