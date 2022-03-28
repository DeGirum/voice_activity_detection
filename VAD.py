import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
import math

def get_keras_padding(input_size: int, 
                      kernel_size: int=3, 
                      stride: int=1, 
                      dilation_rate: int=1
                      ):
    total_pad= max((math.ceil(input_size / stride) - 1) * stride + (kernel_size - 1) * dilation_rate + 1 - input_size, 0)
    return [total_pad//2, total_pad-total_pad//2]

def get_pytorch_padding(kernel_size: int=3, 
                        stride: int = 1, 
                        dilation_rate: int = 1
                        ):
    padding = ((stride - 1) + dilation_rate * (kernel_size - 1)) // 2
    return [padding, padding]

def pytorch_compat_padding(iH,
                           iW,
                           padding,
                           pytorch_compat,
                           stride_h,
                           stride_w,
                           kernel_size_h,
                           kernel_size_w,
                           dilation_rate_h,
                           dilation_rate_w):
    add_pad_layer=False
    pad_params=[(0,0),(0,0)]
    new_pad_mode=padding
    if ((stride_h,stride_w)!=(1,1)) and (pytorch_compat) and (padding=='same'):
        [keras_pad_t, keras_pad_b] = get_keras_padding(iH,
                                                       kernel_size_h,
                                                       stride_h,
                                                       dilation_rate_h
                                                       )
        [keras_pad_l, keras_pad_r] = get_keras_padding(iW,
                                                       kernel_size_w,
                                                       stride_w,
                                                       dilation_rate_w
                                                       )
        [pytorch_pad_t, pytorch_pad_b] = get_pytorch_padding(kernel_size_h, 
                                                             stride_h, 
                                                             dilation_rate_h
                                                             ) 
        [pytorch_pad_l, pytorch_pad_r] = get_pytorch_padding(kernel_size_w, 
                                                             stride_w, 
                                                             dilation_rate_w
                                                             )
        if (keras_pad_t, keras_pad_b, keras_pad_l, keras_pad_r)!=(pytorch_pad_t, pytorch_pad_b, pytorch_pad_l, pytorch_pad_r):
            add_pad_layer=True
            pad_params= [(pytorch_pad_t, pytorch_pad_b),(pytorch_pad_l, pytorch_pad_r)]
            new_pad_mode='valid'
        
    return [add_pad_layer, pad_params, new_pad_mode]    
        

def Conv2DBnAct(input_shape, 
                filters, 
                kernel_size,
                strides=(1,1),
                padding="valid", 
                dilation_rate=(1, 1), 
                groups=1, 
                activation=None,
                epsilon=0.001, 
                pytorch_compat=False,
                batch_norm=True,
                name_prefix=''):
    inputs = keras.Input(shape=input_shape[1:], 
                         batch_size=input_shape[0])
    x=inputs
    add_pad_layer,pad_params,new_pad_mode = pytorch_compat_padding(x.shape[1],
                                                                   x.shape[2],
                                                                   padding,
                                                                   pytorch_compat,
                                                                   strides[0],
                                                                   strides[1],
                                                                   kernel_size[0],
                                                                   kernel_size[1],
                                                                   dilation_rate[0],
                                                                   dilation_rate[1]
                                                                   )
    if add_pad_layer:
        x=layers.ZeroPadding2D(padding=(pad_params),
                               name=name_prefix+'_padding'
                               )(x)
    x=layers.Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=new_pad_mode,
                    groups=groups,
                    activation=None,
                    use_bias=False,
                    name=name_prefix+'_conv'
                    )(x)
    if batch_norm:
        x=layers.BatchNormalization(axis=-1, 
                                    epsilon=epsilon, 
                                    name=name_prefix+'_bn'
                                    )(x)
    if activation is not None:
        x=layers.Activation(activation,
                            name=name_prefix+'_act'
                            )(x)
    return keras.Model(inputs=inputs, 
                       outputs=x, 
                       name=name_prefix+'_Conv2DBnAct'
                       )


def VAD():
    layer0_Input = layers.Input(name='layer0_Input', shape=(65, 16, 1), batch_size=None)
    layer1_ZeroPadding2D = layers.ZeroPadding2D(name='layer1_ZeroPadding2D', padding=(3, 4, 0, 0))(layer0_Input)
    layer5_Conv2DBnAct = Conv2DBnAct(layer0_Input.shape, filters=32, kernel_size=(1, 16), strides=(1, 1), padding='valid', dilation_rate=(1, 1), groups=1, activation=None, epsilon=0.001, pytorch_compat=False, batch_norm=True, name_prefix='layer5')(layer0_Input)
    layer2_Conv2DBnAct = Conv2DBnAct(layer1_ZeroPadding2D.shape, filters=32, kernel_size=(8, 16), strides=(1, 1), padding='valid', dilation_rate=(1, 1), groups=1, activation='relu', epsilon=0.00001, pytorch_compat=False, batch_norm=True, name_prefix='layer2')(layer1_ZeroPadding2D)
    layer3_Conv2DBnAct = Conv2DBnAct(layer2_Conv2DBnAct.shape, filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1,1), groups=1, activation='relu', epsilon=0.0001, pytorch_compat=False, batch_norm=True, name_prefix='layer3')(layer2_Conv2DBnAct)
    layer4_Conv2DBnAct = Conv2DBnAct(layer3_Conv2DBnAct.shape, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), groups=1, activation=None, epsilon=0.00001, pytorch_compat=False, batch_norm=True, name_prefix='layer4')(layer3_Conv2DBnAct)
    layer6_Add = layers.Add(name='layer6_Add')([layer4_Conv2DBnAct, layer5_Conv2DBnAct])
    layer7_ReLU = layers.ReLU(name='layer7_ReLU', max_value=None, negative_slope=0.0, threshold=0.0)(layer6_Add)
    layer8_ZeroPadding2D = layers.ZeroPadding2D(name='layer8_ZeroPadding2D', padding=(3, 4, 0, 0))(layer7_ReLU)
    layer12_Conv2DBnAct = Conv2DBnAct(layer7_ReLU.shape, filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), groups=1, activation=None, epsilon=0.00001, pytorch_compat=False, batch_norm=True, name_prefix='layer12')(layer7_ReLU)
    layer9_Conv2DBnAct = Conv2DBnAct(layer8_ZeroPadding2D.shape, filters=64, kernel_size=(8, 1), strides=(1, 1), padding='valid', dilation_rate=(1, 1), groups=1, activation='relu', epsilon=0.00001, pytorch_compat=False, batch_norm=True, name_prefix='layer9')(layer8_ZeroPadding2D)
    layer10_Conv2DBnAct = Conv2DBnAct(layer9_Conv2DBnAct.shape, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), groups=1, activation='relu', epsilon=0.00001, pytorch_compat=False, batch_norm=True, name_prefix='layer10')(layer9_Conv2DBnAct)
    layer11_Conv2DBnAct = Conv2DBnAct(layer10_Conv2DBnAct.shape, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), groups=1, activation=None, epsilon=0.00001, pytorch_compat=False, batch_norm=True, name_prefix='layer11')(layer10_Conv2DBnAct)
    layer13_Add = layers.Add(name='layer13_Add')([layer11_Conv2DBnAct, layer12_Conv2DBnAct])
    layer14_ReLU = layers.ReLU(name='layer14_ReLU', max_value=None, negative_slope=0.0, threshold=0.0)(layer13_Add)
    layer15_ZeroPadding2D = layers.ZeroPadding2D(name='layer15_ZeroPadding2D', padding=(3, 4, 0, 0))(layer14_ReLU)
    layera19_Conv2DBnAct = Conv2DBnAct(layer14_ReLU.shape, filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), groups=1, activation=None, epsilon=0.00001, pytorch_compat=False, batch_norm=True, name_prefix='layera19')(layer14_ReLU)
    layer16_Conv2DBnAct = Conv2DBnAct(layer15_ZeroPadding2D.shape, filters=128, kernel_size=(8, 1), strides=(1, 1), padding='valid', dilation_rate=(1, 1), groups=1, activation='relu', epsilon=0.001, pytorch_compat=False, batch_norm=True, name_prefix='layer16')(layer15_ZeroPadding2D)
    layer17_Conv2DBnAct = Conv2DBnAct(layer16_Conv2DBnAct.shape, filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), groups=1, activation='relu', epsilon=0.00001, pytorch_compat=False, batch_norm=True, name_prefix='layer17')(layer16_Conv2DBnAct)
    layer18_Conv2DBnAct = Conv2DBnAct(layer17_Conv2DBnAct.shape, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), groups=1, activation=None, epsilon=0.00001, pytorch_compat=False, batch_norm=True, name_prefix='layer18')(layer17_Conv2DBnAct)
    layer14_Add = layers.Add(name='layer14_Add')([layera19_Conv2DBnAct, layer18_Conv2DBnAct])
    layer15_ReLU = layers.ReLU(name='layer15_ReLU', max_value=None, negative_slope=0.0, threshold=0.0)(layer14_Add)
    layer16_ZeroPadding2D = layers.ZeroPadding2D(name='layer16_ZeroPadding2D', padding=(3, 4, 0, 0))(layer15_ReLU)
    layer19_Conv2DBnAct = Conv2DBnAct(layer15_ReLU.shape, filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), groups=1, activation=None, epsilon=0.00001, pytorch_compat=False, batch_norm=True, name_prefix='layer19')(layer15_ReLU)
    layer20_Conv2DBnAct = Conv2DBnAct(layer16_ZeroPadding2D.shape, filters=128, kernel_size=(8, 1), strides=(1, 1), padding='valid', dilation_rate=(1, 1), groups=1, activation='relu', epsilon=0.00001, pytorch_compat=False, batch_norm=True, name_prefix='layer20')(layer16_ZeroPadding2D)
    layer21_Conv2DBnAct = Conv2DBnAct(layer20_Conv2DBnAct.shape, filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), groups=1, activation='relu', epsilon=0.00001, pytorch_compat=False, batch_norm=True, name_prefix='layer21')(layer20_Conv2DBnAct)
    layer23_Conv2DBnAct = Conv2DBnAct(layer21_Conv2DBnAct.shape, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), groups=1, activation=None, epsilon=0.00001, pytorch_compat=False, batch_norm=True, name_prefix='layer23')(layer21_Conv2DBnAct)

    return keras.Model(name='VAD', inputs=[layer0_Input], outputs=[layer25_Dense, layer25_Dense, layer25_Dense, layer27_Dense])