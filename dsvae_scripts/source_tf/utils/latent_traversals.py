import imageio
import os, sys
import numpy as np

dSprites_range = {
    0: [1., 1.],
    1: [1., 3.],
    2: [0.5, 1.],
    3: [0., 6.28],
    4: [0., 1.],
    5: [0., 1.]
}

def code_traversals(settings, latent_index, latent_range, latent_code_dim, x, results_dir):
    if settings.dlaas:
        sys.path.insert(0, 'source_tf/utils')
    else:
        sys.path.insert(0, '../source_tf/utils')

    from misc import merge
    
    latent_noise = np.expand_dims(np.random.randn(settings.latent_noise_dim).astype(np.float32), 1).repeat(settings.batchsize, axis=1).T

    latent_code = 0.5*np.ones((settings.batchsize, latent_code_dim))
    latent_code[:, latent_index] = np.linspace(latent_range[0], latent_range[1], settings.batchsize)
    images = []

    feed_dict = {latent_code_input:latent_code, tdw_img: x, latent_noise_input:latent_noise}
    data = sess.run([G_dec],feed_dict = feed_dict)
    for val in range(settings.batchsize):
        sdata = data[0][val:val+1]
        sdata = np.clip(sdata,0,1)
        sdata = np.expand_dims(sdata,0)
        img = merge(sdata[0],[1,1])
        images.append(img)
        plt.figure(figsize=(1,1))
        plt.imshow(img)
        plt.savefig(os.path.join(results_dir, 'latent_code_traversals_'+str(latent_index) + '_' + str(val)+'.png'))
        imageio.mimsave(os.path.join(results_dir, 'latent_code_traversals_'+str(latent_index)+'.gif'), images)
        
    return np.expand_dims(np.array(images), 1)

def noise_traversals(settings, latent_code_in, x, results_dir):
    if settings.dlaas:
        sys.path.insert(0, 'source_tf/utils')
    else:
        sys.path.insert(0, '../source_tf/utils')

    from misc import merge
    
    latent_noise = np.random.randn(settings.batchsize, settings.latent_noise_dim).astype(np.float32)
    latent_code = np.expand_dims(latent_code_in, 1).repeat(settings.batchsize, axis=1).T

    images = []

    feed_dict = {latent_code_input:latent_code, tdw_img: x, latent_noise_input:latent_noise}
    data = sess.run([G_dec],feed_dict = feed_dict)
    for val in range(settings.batchsize):
        sdata = data[0][val:val+1]
        sdata = np.clip(sdata,0,1)
        sdata = np.expand_dims(sdata,0)
        img = merge(sdata[0],[1,1])
        images.append(img)
        plt.figure(figsize=(1,1))
        plt.imshow(img)
        plt.savefig(os.path.join(results_dir, 'latent_noise_traversals_' + str(val)+'.png'))
        imageio.mimsave(os.path.join(results_dir, 'latent_noise_traversals.gif'), images)
    
    return np.expand_dims(np.array(images), 1)


def save_latent_code_traversals(settings, latent_code_dim, x, results_dir, wandb):
    if settings.dataname=='dSprites':
        for (i, latent_ind) in enumerate(settings.latent_code_indices):
            img_array = code_traversals(settings, i, dSprites_range[latent_ind], latent_code_dim, x, results_dir)
            if settings.wandb_log_images:
                wandb.log({"latent_traversal_"+str(i): wandb.Video(img_array, fps=4, format="gif")})
    else:
        raise NotImplementedError
