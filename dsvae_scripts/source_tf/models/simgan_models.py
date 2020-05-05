import tensorflow as tf
slim = tf.contrib.slim

# standard convolution layer
def conv2d(x, inputFeatures, outputFeatures, name, train=True,bias=True, strides=[1,2,2,1], kernel=5):
    with tf.variable_scope(name):
        '''Akash: I changed the kernel from 5 to 4'''
        w = tf.get_variable("w",[kernel,kernel,inputFeatures, outputFeatures], initializer=tf.truncated_normal_initializer(stddev=0.02), trainable=train)
        b = tf.get_variable("b",[outputFeatures], initializer=tf.constant_initializer(0.0))
        if bias:
            conv = tf.nn.conv2d(x, w, strides=strides, padding="SAME") + b
        else:
            conv = tf.nn.conv2d(x, w, strides=strides, padding="SAME")
        return conv



def conv_transpose(x, outputShape, name, strides=[1,2,2,1]):
    with tf.variable_scope(name):
        # h, w, out, in
        '''Akash: I changed the kernel from 5 to 4'''
        w = tf.get_variable("w",[5,5, outputShape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b",[outputShape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.conv2d_transpose(x, w, output_shape=outputShape, strides=strides)
        return convt



# leaky reLu unit
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


# fully-conected layer
def dense(x, inputFeatures, outputFeatures, scope=None, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [inputFeatures, outputFeatures], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [outputFeatures], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias


def simvae(c, out_shape, reuse=False):
    with tf.variable_scope("simvae") as scope:
        if reuse:
                scope.reuse_variables()
        gf_dim=64
        c_dim = c.shape[1]
        batchsize = int(c.shape[0])
        h = int(out_shape[0])
        w = int(out_shape[1])
        in_channels = int(out_shape[2])
        
        s2, s4, s8, s16 = int(gf_dim/2), int(gf_dim/4), int(gf_dim/8), int(gf_dim/16)
        z2 = dense(c, c_dim, gf_dim*8*s16*s16, scope='g_h0_lin')
        
        h0 = tf.contrib.layers.instance_norm(tf.reshape(z2, [-1, s16, s16, gf_dim * 8]),center=False,scale=False)
        h0 = tf.nn.relu(h0)
        
        h1 = tf.contrib.layers.instance_norm(conv_transpose(h0, [batchsize, s8, s8, gf_dim*4], "g_h1"),center=False,scale=False)
        h1 = tf.nn.relu(h1)
        
        h2 = tf.contrib.layers.instance_norm(conv_transpose(h1, [batchsize, s4, s4, gf_dim*2], "g_h2"),center=False,scale=False)
        h2 = tf.nn.relu(h2)
        
        h3 = tf.contrib.layers.instance_norm(conv_transpose(h2, [batchsize, s2, s2, gf_dim*1], "g_h3"),center=False,scale=False)
        h3 = tf.nn.relu(h3)
        
        h4 = tf.nn.sigmoid(conv_transpose(h3, [batchsize, h, w, in_channels], "g_h4"))
        
        return h4
    
def simgan_decoder(y, z, out_shape, reuse=False):
    with tf.variable_scope("dec") as scope:
        if reuse:
                scope.reuse_variables()
        gf_dim=out_shape[0]
        batchsize = int(y.shape[0])
        h = int(out_shape[0])
        w = int(out_shape[1])
        in_channels = int(out_shape[2])
        df_dim=h

        s2, s4, s8, s16 = int(gf_dim/2), int(gf_dim/4), int(gf_dim/8), int(gf_dim/16)
        
        #Conv
        h0 = lrelu(conv2d(y, in_channels, df_dim, name='d_h0_conv'))
        
        mu = tf.reshape(slim.fully_connected(z,gf_dim*s4*s4*2),[-1, s4, s4, gf_dim*2])
        sig = tf.reshape(slim.fully_connected(z,gf_dim*s4*s4*2),[-1, s4, s4, gf_dim*2])
        h1 = tf.contrib.layers.instance_norm(conv2d(h0, df_dim, df_dim*2, name='d_h1_conv'),center=False,scale=False)
        h1 = lrelu(sig*h1 + mu) 
        
        mu = tf.reshape(slim.fully_connected(z,gf_dim*4*s8*s8),[-1, s8, s8, gf_dim * 4])
        sig = tf.reshape(slim.fully_connected(z,gf_dim*4*s8*s8),[-1, s8, s8, gf_dim * 4])
        h2 = tf.contrib.layers.instance_norm(conv2d(h1, df_dim*2, df_dim*4, name='d_h2_conv'),center=False,scale=False)
        h2 = lrelu(sig*h2 + mu) 
        
        mu = tf.reshape(slim.fully_connected(z,gf_dim*8*s16*s16),[-1, s16, s16, gf_dim * 8])
        sig = tf.reshape(slim.fully_connected(z,gf_dim*8*s16*s16),[-1, s16, s16, gf_dim * 8])
        h3 = tf.contrib.layers.instance_norm(conv2d(h2, df_dim*4, df_dim*8, name='d_h3_conv'),center=False,scale=False)
        h3 = lrelu(sig*h3 + mu)
        
        h3 = tf.reshape(h3, [batchsize, -1])

        z2 = dense(h3, h3.shape[1], gf_dim*8*s16*s16, scope='g_h0_lin')
        
        #Deconv
        mu = tf.reshape(slim.fully_connected(z,gf_dim*8*s16*s16),[-1, s16, s16, gf_dim * 8])
        sig = tf.reshape(slim.fully_connected(z,gf_dim*8*s16*s16),[-1, s16, s16, gf_dim * 8])
        h0 = tf.contrib.layers.instance_norm(tf.reshape(z2, [-1, s16, s16, gf_dim * 8]),center=False,scale=False)
        h0 = tf.nn.relu(sig*h0 + mu)
        
        mu = tf.reshape(slim.fully_connected(z,gf_dim*4*s8*s8),[batchsize, s8, s8, gf_dim*4])
        sig = tf.reshape(slim.fully_connected(z,gf_dim*4*s8*s8),[batchsize, s8, s8, gf_dim*4])
        h1 = tf.contrib.layers.instance_norm(conv_transpose(h0, [batchsize, s8, s8, gf_dim*4], name="g_h1"),center=False,scale=False)
        h1 = tf.nn.relu(sig*h1 + mu)
        
        mu = tf.reshape(slim.fully_connected(z,gf_dim*2*s4*s4),[batchsize, s4, s4, gf_dim*2])
        sig = tf.reshape(slim.fully_connected(z,gf_dim*2*s4*s4),[batchsize, s4, s4, gf_dim*2])
        h2 = tf.contrib.layers.instance_norm(conv_transpose(h1, [batchsize, s4, s4, gf_dim*2], name="g_h2"),center=False,scale=False)
        h2 = tf.nn.relu(sig*h2 + mu)
        
        mu = tf.reshape(slim.fully_connected(z,gf_dim*1*s2*s2),[batchsize, s2, s2, gf_dim*1])
        sig = tf.reshape(slim.fully_connected(z,gf_dim*1*s2*s2),[batchsize, s2, s2, gf_dim*1])
        h3 = tf.contrib.layers.instance_norm(conv_transpose(h2, [batchsize, s2, s2, gf_dim*1], name="g_h3"),center=False,scale=False)
        h3 = tf.nn.relu(sig*h3 + mu)
        
        h4 = tf.nn.sigmoid(conv_transpose(h3, [batchsize, h, w, in_channels], name="g_h4"))
        
        return h4
    
def simgan_decoder_hr(y, z, out_shape, reuse=False):
    with tf.variable_scope("dec") as scope:
        if reuse:
            scope.reuse_variables()
        gf_dim=out_shape[0]
        batchsize = int(y.shape[0])
        h = int(out_shape[0])
        w = int(out_shape[1])
        in_channels = int(out_shape[2])
        df_dim=h
        y_shape = y.shape[1].value
        print(y_shape)
        s1, s2, s4, s8, s16, s32 = int(y_shape), int(y_shape/2), int(y_shape/4), int(y_shape/8), int(y_shape/16), int(y_shape/32)
        d1, d2, d4, d8, d16 = int(gf_dim), int(gf_dim/2), int(gf_dim/4), int(gf_dim/8), int(gf_dim/16)
        scale = int(gf_dim/y_shape)

        #Conv
        h0 = lrelu(conv2d(y, in_channels, y_shape, name='d_h0_conv')) #(32, 32, 32, 128)
        print(h0.shape)
        mu = tf.reshape(slim.fully_connected(z,y_shape*s4*s4*2),[-1, s4, s4, y_shape*2])
        sig = tf.reshape(slim.fully_connected(z,y_shape*s4*s4*2),[-1, s4, s4, y_shape*2])
        h1 = tf.contrib.layers.instance_norm(conv2d(h0, y_shape, y_shape*2, name='d_h1_conv'),center=False,scale=False)
        h1 = lrelu(sig*h1 + mu) #(32, 16, 16, 128)
        print(h1.shape)
        
        mu = tf.reshape(slim.fully_connected(z,y_shape*4*s8*s8),[-1, s8, s8, y_shape * 4])
        sig = tf.reshape(slim.fully_connected(z,y_shape*4*s8*s8),[-1, s8, s8, y_shape * 4])
        h2 = tf.contrib.layers.instance_norm(conv2d(h1, y_shape*2, y_shape*4, name='d_h2_conv'),center=False,scale=False)
        h2 = lrelu(sig*h2 + mu) 
        print(h2.shape)
        
        mu = tf.reshape(slim.fully_connected(z,y_shape*8*s16*s16),[-1, s16, s16, y_shape * 8])
        sig = tf.reshape(slim.fully_connected(z,y_shape*8*s16*s16),[-1, s16, s16, y_shape * 8])
        h3 = tf.contrib.layers.instance_norm(conv2d(h2, y_shape*4, y_shape*8, name='d_h3_conv'),center=False,scale=False)
        h3 = lrelu(sig*h3 + mu)
        print(h3.shape)
        
#         raise ValueError
        
        h3 = tf.reshape(h3, [batchsize, -1])

        z2 = dense(h3, h3.shape[1], gf_dim*8*s32*s32, scope='g_h0_lin')
#         raise ValueError
        #Deconv
        mu = tf.reshape(slim.fully_connected(z,gf_dim*8*s16*s16),[-1, s16, s16, gf_dim * 8])
        sig = tf.reshape(slim.fully_connected(z,gf_dim*8*s16*s16),[-1, s16, s16, gf_dim * 8])
        h0 = tf.contrib.layers.instance_norm(tf.reshape(z2, [-1, s16, s16, gf_dim * 8]),center=False,scale=False)
        h0 = tf.nn.relu(sig*h0 + mu)
        
        #Deconv
        mu = tf.reshape(slim.fully_connected(z,gf_dim*8*s16*s16),[-1, s16, s16, gf_dim * 8])
        sig = tf.reshape(slim.fully_connected(z,gf_dim*8*s16*s16),[-1, s16, s16, gf_dim * 8])
        h0 = tf.contrib.layers.instance_norm(tf.reshape(z2, [-1, s16, s16, gf_dim * 8]),center=False,scale=False)
        h0 = tf.nn.relu(sig*h0 + mu)
        
        mu = tf.reshape(slim.fully_connected(z,gf_dim*4*s8*s8),[batchsize, s8, s8, gf_dim*4])
        sig = tf.reshape(slim.fully_connected(z,gf_dim*4*s8*s8),[batchsize, s8, s8, gf_dim*4])
        h1 = tf.contrib.layers.instance_norm(conv_transpose(h0, [batchsize, s8, s8, gf_dim*4], name="g_h1"),center=False,scale=False)
        h1 = tf.nn.relu(sig*h1 + mu)
        
        mu = tf.reshape(slim.fully_connected(z,gf_dim*2*s4*s4),[batchsize, s4, s4, gf_dim*2])
        sig = tf.reshape(slim.fully_connected(z,gf_dim*2*s4*s4),[batchsize, s4, s4, gf_dim*2])
        h2 = tf.contrib.layers.instance_norm(conv_transpose(h1, [batchsize, s4, s4, gf_dim*2], name="g_h2"),center=False,scale=False)
        h2 = tf.nn.relu(sig*h2 + mu)
        
        mu = tf.reshape(slim.fully_connected(z,gf_dim*1*s2*s2),[batchsize, s2, s2, gf_dim*1])
        sig = tf.reshape(slim.fully_connected(z,gf_dim*1*s2*s2),[batchsize, s2, s2, gf_dim*1])
        h3 = tf.contrib.layers.instance_norm(conv_transpose(h2, [batchsize, s2, s2, gf_dim*1], name="g_h3"),center=False,scale=False)
        h3 = tf.nn.relu(sig*h3 + mu)
        
        mu = tf.reshape(slim.fully_connected(z,gf_dim*1*s1*s1),[batchsize, s1, s1, gf_dim*1])
        sig = tf.reshape(slim.fully_connected(z,gf_dim*1*s1*s1),[batchsize, s1, s1, gf_dim*1])
        h4 = tf.contrib.layers.instance_norm(conv_transpose(h3, [batchsize, s1, s1, gf_dim*1], name="g_h4"),center=False,scale=False)
        h4 = tf.nn.relu(sig*h4 + mu)
        
        
        mu = tf.reshape(slim.fully_connected(z,gf_dim*1*s1*2*s1*2),[batchsize, s1*2, s1*2, gf_dim*1])
        sig = tf.reshape(slim.fully_connected(z,gf_dim*1*s1*2*s1*2),[batchsize, s1*2, s1*2, gf_dim*1])
        h5 = tf.contrib.layers.instance_norm(conv_transpose(h4, [batchsize, s1*2, s1*2, gf_dim*1], name="g_h5"),center=False,scale=False)
        h5 = tf.nn.relu(sig*h5 + mu)
        
#         mu = tf.reshape(slim.fully_connected(z,gf_dim*1*s1*4*s1*4),[batchsize, s1*4, s1*4, gf_dim*1])
#         sig = tf.reshape(slim.fully_connected(z,gf_dim*1*s1*4*s1*4),[batchsize, s1*4, s1*4, gf_dim*1])
#         h6 = tf.contrib.layers.instance_norm(conv_transpose(h5, [batchsize, s1*4, s1*4, gf_dim*1], name="g_h6"),center=False,scale=False)
#         h6 = tf.nn.relu(sig*h6 + mu)
        print(h5.shape)
        h7 = tf.nn.sigmoid(conv_transpose(h5, [batchsize, h, w, in_channels], name="g_h7"))
        
        return h7

def simgan_discriminator(image, z, reuse=False):
    
    with tf.variable_scope("disc") as scope:
        if reuse:
                scope.reuse_variables()

        batchsize = int(image.shape[0])
        h = int(image.shape[1])
        w = int(image.shape[2])
        in_channels = int(image.shape[3])
        df_dim = h
        z_dim = int(z.shape[1])
        
        h0 = tf.concat([image, z], 3)
        
        h0 = lrelu(conv2d(h0, 2*in_channels, df_dim, name='d_h0_conv'))
        h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, df_dim, df_dim*2, name='d_h1_conv'))) #8x8x64
        h2 = lrelu(tf.contrib.layers.batch_norm(conv2d(h1, df_dim*2, df_dim*4, name='d_h2_conv'))) #4x4x128
        h3 = lrelu(tf.contrib.layers.batch_norm(conv2d(h2, df_dim*4, df_dim*8, name='d_h3_conv'))) #4x4x128
        h3 = tf.reshape(h3, [batchsize, -1])
        
        logit = (dense(h3, 8192, 1, scope='d_mu_lin')) 
           
        return logit,h3
  

def simgan_encoder(X,Y,dim=100, reuse=False):
    with tf.variable_scope("enc") as scope:
        if reuse:
                scope.reuse_variables()
        batchsize = int(Y.shape[0])
        gf_dim=64
        df_dim=64
        h=64
        w=64
        s2, s4, s8, s16 = int(gf_dim/2), int(gf_dim/4), int(gf_dim/8), int(gf_dim/16)
        
        #Conv
        h0 = lrelu(conv2d(X-Y, 3, df_dim, name='d_h0_conv'))
        
        h1 = tf.contrib.layers.instance_norm(conv2d(h0, df_dim, df_dim*2, name='d_h1_conv'),center=False,scale=False)
        h1 = lrelu(h1) 
        
        h2 = tf.contrib.layers.instance_norm(conv2d(h1, df_dim*2, df_dim*4, name='d_h2_conv'),center=False,scale=False)
        h2 = lrelu(h2) 
        
        h3 = tf.contrib.layers.instance_norm(conv2d(h2, df_dim*4, df_dim*8, name='d_h3_conv'),center=False,scale=False)
        h3 = lrelu(h3)
        
        h3 = tf.reshape(h3, [batchsize, -1])

        return dense(h3, 8192, dim, scope='g_h0_lin'),\
                tf.nn.softplus(dense(h3, 8192, dim, scope='g_h0_lin_'))
