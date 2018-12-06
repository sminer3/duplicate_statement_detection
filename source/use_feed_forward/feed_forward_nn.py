from keras.layers import Dense, Input, Dropout
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers


def feed_forward_nn(emb_shape=512, features_shape=5, layer1=75, n_layer1_hidden=0, dropout_rate1 = .4, layer1_reg =.01, layer2=30, n_layer2_hidden = 0, dropout_rate2 = .3, layer2_reg =.01, features=10, dropout_rate_features = .1, feature_reg = .01, activation='relu'):
    layer_one = Dense(layer1,activation=activation, kernel_regularizer = regularizers.l2(layer1_reg), name = "first_layer")

    s1_emb = Input(shape=(emb_shape,), dtype="float32", name = "s1_input")
    s1_out = layer_one(s1_emb)

    s2_emb = Input(shape=(emb_shape,), dtype="float32", name = "s2_input")
    s2_out = layer_one(s2_emb)

    for i in range(n_layer1_hidden):
        dropout_layer1 = Dropout(dropout_rate1, name="hidden1_dropout"+str(i))
        layer_extra = Dense(layer1,activation=activation, kernel_regularizer = regularizers.l2(layer1_reg), name= "layer1_hidden"+str(i))
        s1_out = dropout_layer1(s1_out)
        s2_out = dropout_layer1(s2_out)
        s1_out = layer_extra(s1_out)
        s2_out = layer_extra(s2_out)
        

    #features_input = Input(shape=(features_shape,), dtype="float32", name = "features_input")
    #features_dense = BatchNormalization(name = "features_batch_normalization")(features_input)
    #features_dense = Dense(features, activation=activation, kernel_regularizer = regularizers.l2(feature_reg), name = "features_layer")(features_input)
    #features_dense = Dropout(dropout_rate_features, name = "features_dropout")(features_dense)

    addition = add([s1_out, s2_out], name="add_s1_and_s2")
    minus_s2 = Lambda(lambda x: -x, name = "negative_s2")(s2_out)
    merged = add([s1_out, minus_s2], name = "subtract_s1_and_s2")
    merged = multiply([merged, merged], name= "squared_difference")
    merged = concatenate([merged, addition], name="merge_subtract_and_add")
    merged = Dropout(dropout_rate1, name = "statement_dropout")(merged)
	
    #merged = concatenate([merged,features_dense], name = "merge_statements_and_features")

    merged = Dense(layer2,activation=activation, kernel_regularizer = regularizers.l2(layer2_reg), name = "second_layer")(merged)
    merged = Dropout(dropout_rate2, name = "second_dropout")(merged)
    for i in range(n_layer2_hidden):
        merged = Dense(layer2, activation=activation, kernel_regularizer = regularizers.l2(layer2_reg), name = "layer2_hidden"+str(i))(merged)
        merged = Dropout(dropout_rate2, name = "hidden2_dropout"+str(i))(merged)

    out = Dense(1, activation="sigmoid", name = "output")(merged)

    #inputs = [s1_emb,s2_emb,features_input]
    inputs = [s1_emb,s2_emb]

    model = Model(inputs=inputs, outputs=out)
    return model