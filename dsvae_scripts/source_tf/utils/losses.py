from keras import metrics
import tensorflow as tf

def get_encoder_loss(enc_loss_name):
    if enc_loss_name == 'mse_reconst':
        reconstr_loss = lambda x, y: metrics.mean_squared_error(x, y)
        loss_fn = lambda x, y: tf.reduce_mean(reconstr_loss(x, y))
    elif enc_loss_name == 'ce_reconst':
        reconstr_loss = lambda x, y: 64*64*1*metrics.binary_crossentropy(x, y)
        loss_fn = lambda x, y: tf.reduce_mean(reconstr_loss(x, y))
    return loss_fn
