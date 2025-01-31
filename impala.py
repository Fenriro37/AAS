from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, Add, MaxPooling2D, Dense, Flatten
import tensorflow as tf

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.conv1 = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')
        self.bn1 = BatchNormalization()  

        self.relu = ReLU()
        self.conv2 = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = BatchNormalization()  

        self.add = Add()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)  
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)  

        return self.relu(self.add([x, inputs]))


class IMPALABlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(IMPALABlock, self).__init__(**kwargs)
        self.conv = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')
        self.bn = BatchNormalization()  
        self.pool = MaxPooling2D(pool_size=(3, 3), strides=2)

        self.relu = ReLU()
        self.res1 = ResBlock(filters)
        self.res2 = ResBlock(filters)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)  
        x = self.relu(x)
        x = self.pool(x)

        x = self.res1(x, training=training)
        x = self.res2(x, training=training)

        return x


class IMPALANet(tf.keras.Model):
    def __init__(self, num_actions, impala_filters=[16, 32, 32], dense_units=256, **kwargs):
        super(IMPALANet, self).__init__(**kwargs)
        self.impala1 = IMPALABlock(impala_filters[0])
        self.impala2 = IMPALABlock(impala_filters[1])
        self.impala3 = IMPALABlock(impala_filters[2])

        self.flatten = Flatten()
        self.bn = BatchNormalization()  
        self.relu = ReLU()
        self.dense = Dense(dense_units)

        self.policy_layer = Dense(num_actions, name='policy')
        self.value_layer = Dense(1, name='value')

    def call(self, inputs, training=False):
        normalized_inputs = tf.cast(inputs, tf.float32) / 255.0
        x = self.impala1(normalized_inputs, training=training)
        x = self.impala2(x, training=training)
        x = self.impala3(x, training=training)

        x = self.flatten(x)
        x = self.bn(x, training=training)  
        x = self.dense(x)
        x = self.relu(x)

        policy = self.policy_layer(x)
        value = self.value_layer(x)

        return policy, value

class IMPALAPolicyNet(tf.keras.Model):
    def __init__(self, num_actions, impala_filters=[16, 32, 32], dense_units=256, **kwargs):
        super(IMPALAPolicyNet, self).__init__(**kwargs)
        self.impala1 = IMPALABlock(impala_filters[0])
        self.impala2 = IMPALABlock(impala_filters[1])
        self.impala3 = IMPALABlock(impala_filters[2])
        self.flatten = Flatten()
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.dense = Dense(dense_units)
        self.policy_layer = Dense(num_actions, activation="softmax", name="policy")  

    def call(self, inputs, training=False):
        normalized_inputs = tf.cast(inputs, tf.float32) / 255.0
        x = self.impala1(normalized_inputs, training=training)
        x = self.impala2(x, training=training)
        x = self.impala3(x, training=training)
        x = self.flatten(x)
        x = self.bn(x, training=training)
        x = self.dense(x)
        x = self.relu(x)
        policy_probs = self.policy_layer(x)
        return policy_probs  


class IMPALAValueNet(tf.keras.Model):
    def __init__(self, impala_filters=[16, 32, 32], dense_units=256, **kwargs):
        super(IMPALAValueNet, self).__init__(**kwargs)
        self.impala1 = IMPALABlock(impala_filters[0])
        self.impala2 = IMPALABlock(impala_filters[1])
        self.impala3 = IMPALABlock(impala_filters[2])
        self.flatten = Flatten()
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.dense = Dense(dense_units)
        self.value_layer = Dense(1, name='value')

    def call(self, inputs, training=False):
        normalized_inputs = tf.cast(inputs, tf.float32) / 255.0
        x = self.impala1(normalized_inputs, training=training)
        x = self.impala2(x, training=training)
        x = self.impala3(x, training=training)
        x = self.flatten(x)
        x = self.bn(x, training=training)
        x = self.dense(x)
        x = self.relu(x)
        value = self.value_layer(x)
        return value