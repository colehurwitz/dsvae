import os
import subprocess

def create_config(dim, beta, seed):
    config_name = '/hdd/dsvae/tcvae_models/smallnorb/config/d_{}_b_{}.gin'.format(dim, beta)
    basefile = '/hdd/dsvae/tcvae_models/smallnorb/config/base.gin'
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
        else:
            file.writelines([line]) 
    file.close()
    return config_name

def run_with_config(config_file, train_output_dir, eval_output_dir):
    train_command = ['python', 'dlib_train_gpu_1.py', '--gin_config={}'.format(config_file), '--model_dir={}'.format(train_output_dir)]
    #eval_command = ['python', 'dlib_reproduce.py', '--model_dir={}'.format(train_output_dir), '--output_directory={}'.format(eval_output_dir)]
    print(train_command)
    with open('./exp_output/{}_train.out'.format(exp_name), 'w') as f:
        process = subprocess.Popen(train_command, stdout=f)
        exit_codes = process.wait()
        print(exit_codes)
    # no evaluation
        
        
dim_list = [20]
beta_list = [4, 10, 1]
seeds = [0, 1, 2] # !!!!!
print("running {}, {}, {}".format(dim_list, beta_list, seeds))
for dim in dim_list:
    for beta in beta_list:
        for seed in seeds:
            exp_name = 'd_{}_b_{}_{}'.format(dim, beta, seed) # first exp without repeative
            print(exp_name)
            config_file = create_config(dim, beta, seed)
            train_output_dir = '/hdd/dsvae/tcvae_models/smallnorb/train_output/{}'.format(exp_name)
            eval_output_dir = '/hdd/dsvae/tcvae_models/smallnorb/eval_output/{}'.format(exp_name)
            run_with_config(config_file, train_output_dir, eval_output_dir)





        
