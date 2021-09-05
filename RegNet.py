import tensorflow as tf

def X_block_s1(x, wi, bi, gi):
    """
    Block with stride=1
    """
    
    x1 = tf.keras.layers.Conv2D(wi//bi, 
                                kernel_size=1,
                                strides=(1, 1),
                                groups=1, 
                                activation=None,
                                kernel_initializer='glorot_uniform')(x)

    x2 = tf.keras.layers.BatchNormalization()(x1)
    x3 = tf.keras.layers.Activation('relu')(x2)

    x4 = tf.keras.layers.Conv2D(wi//bi, 
                                kernel_size=3,
                                strides=(1, 1),
                                padding='same',
                                groups=gi, 
                                activation=None,
                                kernel_initializer='glorot_uniform')(x3)

    x5 = tf.keras.layers.BatchNormalization()(x4)
    x6 = tf.keras.layers.Activation('relu')(x5)

    x7 = tf.keras.layers.Conv2D(wi, 
                                kernel_size=1,
                                strides=(1, 1),
                                groups=1, 
                                activation=None,
                                kernel_initializer='glorot_uniform')(x6)

    x8 = tf.keras.layers.BatchNormalization()(x7)
    x9 = tf.keras.layers.Activation('relu')(x8)

    x10 = tf.keras.layers.Add()([x, x9])

    return x10


def X_block_s2(x, wi,bi,gi):
    """
    Block with stride=2
    """
    
    x1 = tf.keras.layers.Conv2D(wi//bi,
                                kernel_size=1,
                                strides=(1, 1),
                                groups=1,
                                activation=None,
                                kernel_initializer='glorot_uniform')(x)

    x2 = tf.keras.layers.BatchNormalization()(x1)
    x3 = tf.keras.layers.Activation('relu')(x2)

    x4 = tf.keras.layers.Conv2D(wi//bi, 
                                kernel_size=3,
                                strides=(2, 2),
                                padding='same',
                                groups=gi, 
                                activation=None,
                                kernel_initializer='glorot_uniform')(x3)

    x5 = tf.keras.layers.BatchNormalization()(x4)
    x6 = tf.keras.layers.Activation('relu')(x5)

    x7 = tf.keras.layers.Conv2D(wi, 
                                kernel_size=1,
                                strides=(1, 1),
                                groups=1, 
                                activation=None,
                                kernel_initializer='glorot_uniform')(x6)

    x8 = tf.keras.layers.BatchNormalization()(x7)
    x9 = tf.keras.layers.Activation('relu')(x8)

    x1_2 = tf.keras.layers.Conv2D(wi,
                                  kernel_size=1,
                                  strides=(2, 2),
                                  groups=1,
                                  activation=None,
                                  kernel_initializer='glorot_uniform')(x)

    x2_2 = tf.keras.layers.BatchNormalization()(x1_2)
    x3_2 = tf.keras.layers.Activation('relu')(x2_2)

    x10 = tf.keras.layers.Add()([x9, x3_2])

    return x10


def stem(x, w0):
    x1 = tf.keras.layers.Conv2D(w0,
                               kernel_size=1,
                               strides=(2, 2),
                               groups=1,
                               activation=None,
                               kernel_initializer='glorot_uniform')(x)

    x2 = tf.keras.layers.BatchNormalization()(x1)
    x3 = tf.keras.layers.Activation('relu')(x2)

    return x3

def head(x, units, activation):
    #x1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None)(x)
    x1 = tf.keras.layers.GlobalAveragePooling2D()(x)
    x2 = tf.keras.layers.Dense(units=units, 
                                activation=activation,
                                kernel_initializer='glorot_uniform')(x1)
    return x2


def stage1(x, di, wi, bi, gi):
    x = X_block_s2(x, wi,bi,gi)
    for i in range(2, di+1):
        x = X_block_s1(x, wi, bi, gi)
    return x

def stage2(x, di, wi, bi, gi):
    x = X_block_s2(x, wi,bi,gi)
    for i in range(2, di+1):
        x = X_block_s1(x, wi, bi, gi)
    return x

def stage3(x, di, wi, bi, gi):
    x = X_block_s2(x, wi,bi,gi)
    for i in range(2, di+1):
        x = X_block_s1(x, wi, bi, gi)
    return x

def stage4(x, di, wi, bi, gi):
    x = X_block_s2(x, wi,bi,gi)
    for i in range(2, di+1):
        x = X_block_s1(x, wi, bi, gi)
    return x


def RegNetX_200MF(x):
    """
        di = [1,1,4,7]
        wi = [24,56,152,368]
        g = 8
        b = 1
        wa = 36, w0 = 24, wm =2.5
        2.7 Million Parameters
        error rate 30.8%
        
    """
    # stem
    x = stem(x, w0=24)
    
    # body
    x = stage1(x, di=1, wi=24, bi=1, gi=8)
    x = stage2(x, di=1, wi=56, bi=1, gi=8)
    x = stage3(x, di=4, wi=152, bi=1, gi=8)
    x = stage4(x, di=7, wi=368, bi=1, gi=8)
    
    return x
    

def RegNetX_400MF(x):
    """
        di = [1,2,7,12]
        wi = [32,64,160,384]
        g = 16
        b = 1
        wa = 24, w0 = 24, wm =2.5
        5.2 Million Parameters
        error rate 27.2%
        
    """
    # stem
    x = stem(x, w0=24)
    
    # body
    x = stage1(x, di=1, wi=32, bi=1, gi=16)
    x = stage2(x, di=2, wi=64, bi=1, gi=16)
    x = stage3(x, di=7, wi=160, bi=1, gi=16)
    x = stage4(x, di=12, wi=384, bi=1, gi=16)
    
    return x
    
def RegNetX_600MF(x):
    """
        di = [1,3,5,7]
        wi = [48,96,240,528]
        g = 24
        b = 1
        wa = 37, w0 = 48, wm =2.2
        6.2 Million Parameters
        error rate 25.5%
        
    """
    # stem
    x = stem(x, w0=48)
    
    # body
    x = stage1(x, di=1, wi=48, bi=1, gi=24)
    x = stage2(x, di=3, wi=96, bi=1, gi=24)
    x = stage3(x, di=5, wi=240, bi=1, gi=24)
    x = stage4(x, di=7, wi=528, bi=1, gi=24)
    
    return x
    
def RegNetX_800MF(x):
    """
        di = [1,3,7,5]
        wi = [64,128,288,672]
        g = 16
        b = 1
        wa = 36, w0 = 56, wm =2.3
        7.3 Million Parameters
        error rate 24.8%
        
    """
    # stem
    x = stem(x, w0=56)
    
    # body
    x = stage1(x, di=1, wi=64, bi=1, gi=16)
    x = stage2(x, di=3, wi=128, bi=1, gi=16)
    x = stage3(x, di=7, wi=288, bi=1, gi=16)
    x = stage4(x, di=5, wi=672, bi=1, gi=16)
    
    return x
    
