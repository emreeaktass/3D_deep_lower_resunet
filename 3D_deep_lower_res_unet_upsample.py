import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, UpSampling2D, concatenate, Flatten, Dense, \
    BatchNormalization, Add, Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


def resize_layer(input):
    p_1 = conv3D_layer(input=input, filter_num=24, filter_size=(3, 3, 3), padding='SAME', strides=(1, 1, 1))
    p_2 = conv3D_layer(input=input, filter_num=32, filter_size=(2, 2, 2), padding='VALID', strides=(2, 2, 2))
    p_4 = conv3D_layer(input=input, filter_num=48, filter_size=(4, 4, 4), padding='VALID', strides=(4, 4, 4))
    p_8 = conv3D_layer(input=input, filter_num=60, filter_size=(12, 8, 8), padding='VALID', strides=(12, 8, 8))
    p_16 = conv3D_layer(input=input, filter_num=64, filter_size=(24, 16, 16), padding='VALID', strides=(24, 16, 16))
    return p_1, p_2, p_4, p_8, p_16


def conv3D_layer(input, filter_num, filter_size, padding, strides):
    conv = Conv3D(filters=filter_num, kernel_size=filter_size, padding=padding, strides=strides,
                  kernel_initializer='he_normal', use_bias=False)(input)
    return conv


def conv3D_transpose_layer(input, size=(2, 2, 2)):
    conv = UpSampling3D(size=size)(input)
    return conv


def pool3D_layer(input, pool_size):
    pool = MaxPooling3D(pool_size=pool_size)(input)
    return pool


def act_layer(input):
    return tf.nn.relu(input)


def bn_layer(input):
    return BatchNormalization(momentum=0.99, epsilon=1e-3)(input)


def res3D_conv_block(input, filter_num, filter_size, padding, strides, batch_norm):
    x = conv3D_layer(input, filter_num, filter_size, padding, strides)
    if batch_norm:
        x = bn_layer(x)
    x = act_layer(x)
    x = conv3D_layer(x, filter_num, filter_size, padding, strides)
    if batch_norm:
        x = bn_layer(x)
    bridge = conv3D_layer(input=input, filter_num=filter_num, filter_size=(1, 1, 1), padding=padding, strides=strides)
    x = Add()([x, bridge])
    x = act_layer(x)
    return x


def pool_drop_3Dlayer(input, pool, dropout, pool_size=(2, 2, 2)):
    if pool:
        input = pool3D_layer(input, pool_size)
    if dropout:
        input = Dropout(0.2)(input)
    return input


def concatenate_with_low_resolution(input_1, input_2):
    concat = tf.concat((input_1, input_2), axis=-1)
    return concat


def merging_with_transpose_layer(input_1, input_2):
    concat = tf.concat((input_1, input_2), axis=-1)
    return concat


def res_deep_lower_unet(input_size=(24, 64, 64, 1)):
    inputs = Input(input_size)
    p_1, p_2, p_4, p_8, p_16 = resize_layer(inputs)
    r_1 = res3D_conv_block(input=p_1, filter_num=24, filter_size=(3, 3, 3), padding='SAME', strides=(1, 1, 1),
                           batch_norm=True)
    pd_1 = pool_drop_3Dlayer(input=r_1, pool=True, dropout=True)
    c_1 = concatenate_with_low_resolution(pd_1, p_2)
    r_2 = res3D_conv_block(input=c_1, filter_num=32, filter_size=(3, 3, 3), padding='SAME', strides=(1, 1, 1),
                           batch_norm=True)
    pd_2 = pool_drop_3Dlayer(input=r_2, pool=True, dropout=True)
    c_2 = concatenate_with_low_resolution(pd_2, p_4)
    r_3 = res3D_conv_block(input=c_2, filter_num=48, filter_size=(3, 3, 3), padding='SAME', strides=(1, 1, 1),
                           batch_norm=True)
    pd_3 = pool_drop_3Dlayer(input=r_3, pool=True, dropout=True, pool_size=(3, 2, 2))
    c_3 = concatenate_with_low_resolution(pd_3, p_8)
    r_4 = res3D_conv_block(input=c_3, filter_num=60, filter_size=(3, 3, 3), padding='SAME', strides=(1, 1, 1),
                           batch_norm=True)
    pd_4 = pool_drop_3Dlayer(input=r_4, pool=True, dropout=False)
    c_4 = concatenate_with_low_resolution(pd_4, p_16)
    r_5 = res3D_conv_block(input=c_4, filter_num=64, filter_size=(3, 3, 3), padding='SAME', strides=(1, 1, 1),
                           batch_norm=True)
    pd_5 = pool_drop_3Dlayer(input=r_5, pool=False, dropout=False)
    t_1 = conv3D_transpose_layer(input=pd_5)
    tc_5 = merging_with_transpose_layer(t_1, r_4)
    tr_5 = res3D_conv_block(input=tc_5, filter_num=60, filter_size=(3, 3, 3), padding='SAME', strides=(1, 1, 1),
                            batch_norm=True)
    pd_6 = pool_drop_3Dlayer(input=tr_5, pool=False, dropout=True)
    t_2 = conv3D_transpose_layer(input=pd_6, size=(3, 2, 2))
    tc_6 = merging_with_transpose_layer(t_2, r_3)
    tr_6 = res3D_conv_block(input=tc_6, filter_num=48, filter_size=(3, 3, 3), padding='SAME', strides=(1, 1, 1),
                            batch_norm=True)
    pd_7 = pool_drop_3Dlayer(input=tr_6, pool=False, dropout=True)
    t_3 = conv3D_transpose_layer(input=pd_7)
    tc_7 = merging_with_transpose_layer(t_3, r_2)
    tr_7 = res3D_conv_block(input=tc_7, filter_num=32, filter_size=(3, 3, 3), padding='SAME', strides=(1, 1, 1),
                            batch_norm=True)
    pd_8 = pool_drop_3Dlayer(input=tr_7, pool=False, dropout=True)
    t_4 = conv3D_transpose_layer(input=pd_8)
    tc_8 = merging_with_transpose_layer(t_4, r_1)
    tr_8 = res3D_conv_block(input=tc_8, filter_num=24, filter_size=(3, 3, 3), padding='SAME', strides=(1, 1, 1),
                            batch_norm=True)
    pd_9 = pool_drop_3Dlayer(input=tr_8, pool=False, dropout=False)
    res_1 = Conv3D(8, 3, activation='relu', padding='SAME')(pd_9)
    res_2 = Conv3D(4, 3, activation='relu', padding='SAME')(res_1)
    final = Conv3D(1, 1, activation='sigmoid', padding='SAME')(res_2)
    model = Model(inputs, final)
    model.compile(optimizer=Adam(lr=0.0001), loss=my_loss, metrics=[my_coef])
    # print(model.summary())
    return model


def my_coef(y_true, y_pred):
    T = tf.reshape(y_true, [-1])
    P = tf.reshape(y_pred, [-1])
    intersection = tf.constant(2.) * tf.tensordot(T, P, 1) + 0.0001
    union = tf.reduce_sum((tf.add(T, P))) + 0.0001
    dice = intersection / union
    return dice


def my_loss(y_true, y_pred):
    return 1 - my_coef(y_true, y_pred)


if __name__ == '__main__':


    model = res_deep_lower_unet()
    model.summary()
