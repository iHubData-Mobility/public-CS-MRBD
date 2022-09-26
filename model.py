import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    LSTMCell,
    Input,
    Dense,
    concatenate,
    TimeDistributed,
    Conv2D,
    Flatten,
    Dropout,
    Masking,
    RepeatVector
)
from tensorflow.nn import tanh, softmax, sigmoid
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from attention_module import attach_attention_module
#proof of using LSTM all outputs as hidden state. https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/

# Attention LSTM usage: https://matthewmcateer.me/blog/getting-started-with-attention-for-classification/
TIME_STEPS = 15

UNITS_DENSE = 5
UNITS_GRU = 128
UNITS_LSTM = 512

DROPOUT_STRENGTH = 0.5
RECURRENT_DROPOUT_STRENGTH = 0.5

BIAS_INITIALIZER = "ones"
KERNEL_INITIALIZER = "VarianceScaling"

class AttentionLSTM:
    """
        Class implements LSTM with Attention.
    """
    def __init__(self, time_steps, dense_units=UNITS_DENSE, lstm_units=UNITS_LSTM, 
                 dropout=DROPOUT_STRENGTH, rec_dropout=RECURRENT_DROPOUT_STRENGTH, 
                 kernel_initializer=KERNEL_INITIALIZER, 
                 bias_initializer=BIAS_INITIALIZER):
        """
            Method initialized the network.
        """
        self.time_steps = time_steps
        
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        self.lstm_units = lstm_units
        self.dense_units = dense_units

    def video_rnn(self, inputs):
        """
            Method implements recurrent network.
        """
	#iinception 
        base_model = TimeDistributed(
            InceptionResNetV2(
                weights="imagenet", 
                pooling=None, 
                include_top=False, 
                input_shape=(270, 480, 3)
            ), input_shape=(self.time_steps, 270, 480, 3)
        )(inputs)
        base_model2 = tf.unstack(base_model,axis = 1)
        base_model3 = []
        for t in base_model2: base_model3.append(attach_attention_module(t))#https://github.com/kobiso/CBAM-keras/blob/master/main.py
        base_model3 = tf.convert_to_tensor(base_model3)#https://stackoverflow.com/questions/43327668/looping-over-a-tensor
        base_model4 = tf.transpose(base_model3, [1,0,2,3,4])
        cnnfeatures = TimeDistributed(
            Conv2D(
                20, (1,1), 
                activation="relu", 
                name="cnn_conv"
            )
        )(base_model4)
        cnnfeatures = TimeDistributed(Flatten())(cnnfeatures)        
        lstm_out_cnnfeatures, hidden_states, _ = LSTM(
            self.lstm_units,
            kernel_initializer=self.kernel_initializer,
            return_state=True, 
            return_sequences=True,
            recurrent_dropout=self.rec_dropout,
            bias_initializer=self.bias_initializer,
            name="lstm_videos"
        )(cnnfeatures)
        
        context_features = []
        lstm_out_cnnfeatures = Dropout(self.dropout)(lstm_out_cnnfeatures)
        for i in range(TIME_STEPS):
            context_vector = self.attention_cell(10, lstm_out_cnnfeatures, lstm_out_cnnfeatures[:, i])
            context_features.append(context_vector)
        context_features = tf.stack(context_features, axis=1)
        
        # return lstm_out_cnnfeatures, hidden_states, context_features
        return context_features

    def attention_cell(self, units, inputs, hidden):
            """
                Method implements attention cell.
            """
            hidden = tf.expand_dims(hidden, 1)

            confidence = tanh(
                Dense(units)(inputs)+Dense(units)(hidden)
            )
            attention_weights = softmax(
                Dense(1)(confidence),
                axis=1
            )
            context_vector = attention_weights * inputs
            context_vector = tf.reduce_sum(
                context_vector,
                axis=1
            )
            return context_vector
    
    def vehicle_rnn(self, inputs):
        """
           Method implements vechile lstm.
        """
        lstm_out_vehicle = TimeDistributed(Masking(mask_value=0), name='dense_vehicle')(inputs)
        lstm_out_vehicle, hidden_state,_ = LSTM(
            16, kernel_initializer=self.kernel_initializer, return_sequences=True,
            return_state=True,
            recurrent_dropout=self.rec_dropout,
            bias_initializer=self.bias_initializer,
            name='lstm_vehicle'
        )(lstm_out_vehicle)
        
        context_features = []
        lstm_out_vehicle = Dropout(self.dropout)(lstm_out_vehicle)
        for i in range(TIME_STEPS):
            context_vector = self.attention_cell(10, lstm_out_vehicle, lstm_out_vehicle[:, i])
            context_features.append(context_vector)
        context_features = tf.stack(context_features, axis=1)

        # return lstm_out_vehicle, hidden_state, context_features
        return context_features

    def gaze_rnn(self, inputs):
        """
            Methods implements gaze lstm.
        """
        lstm_out_gaze = TimeDistributed(Masking(mask_value=0), name='dense_gaze')(inputs)
        lstm_out_gaze, hidden_state,_ = LSTM(
            16, kernel_initializer=self.kernel_initializer, return_sequences=True,
            recurrent_dropout=self.rec_dropout,
            bias_initializer=self.bias_initializer,
            name='lstm_gaze',
            return_state=True
        )(lstm_out_gaze)
       
        context_features = []
        lstm_out_gaze = Dropout(self.dropout)(lstm_out_gaze)
        for i in range(TIME_STEPS):
            context_vector = self.attention_cell(10, lstm_out_gaze, lstm_out_gaze[:, i])
            context_features.append(context_vector)
        context_features = tf.stack(context_features, axis=1)
 
        # return lstm_out_gaze, hidden_state, context_features
        return context_features
    def get_attention_focused_hidden_state(self, units, inputs, hidden):
        """
            Method implements attention cell.
        """
        hidden1 = tf.expand_dims(hidden, 1)#?X1X16

        confidence = tanh(
            Dense(units,use_bias=False)(inputs)+Dense(units,use_bias=True)(hidden1)#ψt,t′ = tanh(Wψ ht + Wψ′ ht′ + bψ ) #?X15X5
        )
        beta_t_tdash = sigmoid(Dense(1,use_bias=True)(confidence))#βt,t′ = σ(Wg ψt,t′ + bg ) t′ is 1 to 15 #1X15
        at = hidden + tf.reduce_sum(beta_t_tdash * inputs, axis=1) #at = ht +T∑t′ =1βt,t′ ht′  # beta_t_tdash * inputs -> #1X15 * 512X15, at is 512X1
        return at


    def final_rnn(self, inputs):
        """
            Methods implements gaze lstm.
        """
        lstm_out_final = TimeDistributed(Masking(mask_value=0), name='dense_final')(inputs)
        lstm_out_final, hidden_state, _ = LSTM(
            5, kernel_initializer=self.kernel_initializer, return_sequences=True,
            recurrent_dropout=self.rec_dropout,
            bias_initializer=self.bias_initializer,
            name='lstm_final',
            return_state=True
        )(lstm_out_final)
        lstm_out_final = Dropout(self.dropout)(lstm_out_final)
        ats = []
        w_phi = Dense(1, use_bias=True)
        wats = []
        #lstm_out_gaze = Dropout(self.dropout)(lstm_out_gaze)
        for i in range(TIME_STEPS):
            at = self.get_attention_focused_hidden_state(3, lstm_out_final, lstm_out_final[:, i])
            ats.append(at)
            wats.append(w_phi(at))
        #apply softmax and reduce sum as per eqn 3 of https://openaccess.thecvf.com/content/WACV2021/papers/Wharton_Coarse_Temporal_Attention_Network_CTA-Net_for_Drivers_Activity_Recognition_WACV_2021_paper.pdf
        #lstm_out_final = Dropout(self.dropout)(lstm_out_final)
        ats = tf.convert_to_tensor(ats)#15XNoneX5 #[15,?,16]
        wats = tf.convert_to_tensor(wats)#15XNoneX5
        ats = tf.transpose(ats,[1,0,2]) #NoneX15X5
        wats = tf.transpose(wats,[1,0,2]) #Nonex15x1
        wt = softmax(
            wats,
            axis=1
        )

        op = tf.reduce_sum(wt * ats, axis=1)
        #op = Dropout(self.dropout)(op)
        op = softmax(op, axis=1) 
        #op = Dropout(self.dropout)(op)
        return op     
        #return lstm_out_final, hidden_state

    def generate(self):
        """
            Method generates model architecture.
        """
        input_video = Input(shape=(self.time_steps, 270, 480, 3), name="input_videos")
        input_vehicle = Input(shape=(self.time_steps, 12), name='input_vehicle')
        input_gaze = Input(shape=(self.time_steps, 30), name='input_gaze')

        # video_features, video_hidden_state
        video_context = self.video_rnn(input_video)
        # video_context = self.attention_cell(10, video_features, video_hidden_state)
        
        # vehicle_features, vehicle_hidden_state
        vehicle_context = self.vehicle_rnn(input_vehicle)
        # vehicle_context = self.attention_cell(10, vehicle_features, vehicle_hidden_state)

        # gaze_features, gaze_hidden_state
        gaze_context = self.gaze_rnn(input_gaze)
        # gaze_context = self.attention_cell(10, gaze_features, gaze_hidden_state)

        concat_context = concatenate([video_context, vehicle_context, gaze_context])
        #concat_features = concatenate([video_features, vehicle_features, gaze_features])
        #concat_hidden_states = concatenate([video_hidden_state, vehicle_hidden_state, gaze_hidden_state])
        
        # Final LSTM output
        #concat_features, concat_hidden_states 
        classifier = self.final_rnn(concat_context)
        '''concat_context = self.attention_cell(10, concat_features, concat_hidden_states)

        aux_output = Dense(
            self.dense_units,
            activation="tanh",
            kernel_initializer=self.kernel_initializer,
            name="action_context_fusion"
        )(concat_context)

        aux_output = Dropout(self.dropout)(aux_output)

        classifier = Dense(
            self.dense_units,
            activation="softmax",
            kernel_initializer=self.kernel_initializer
        )(aux_output)'''
        model = tf.keras.Model(inputs=[input_video, input_vehicle, input_gaze], outputs=[classifier])
        return model

class FineTuneTrident_l3:
    def __init__(self, time_steps, train_pth=None):
        self.time_steps = time_steps
        self.train_pth = train_pth

    def generate(self):
        input_video = Input(shape=(self.time_steps, 270, 480, 3), name="input_videos")
        input_vehicle = Input(shape=(self.time_steps, 12), name='input_vehicle')
        input_gaze = Input(shape=(self.time_steps, 30), name='input_gaze')

        model_l1 = AttentionLSTM(self.time_steps).generate()
        # make all layers of model_part non trainable
        if self.train_pth:
            model_part.load_weights(self.train_pth)
        outputs_part = model_part([input_video, input_vehicle, input_gaze])
        
        classifier = Dense(
            14,
            activation="softmax",
            kernel_initializer="VarianceScaling"
        )(outputs_part)

        modell3 = tf.keras.Model(inputs=[input_video, input_vehicle, input_gaze], outputs=[classifier])

        return modell3, modell1
        
class FineTuneTrident_l2:
    def __init__(self, time_steps, train_pth=None):
        self.time_steps = time_steps
        self.train_pth = train_pth

    def generate(self):
        input_video = Input(shape=(self.time_steps, 270, 480, 3), name="input_videos")
        input_vehicle = Input(shape=(self.time_steps, 12), name='input_vehicle')
        input_gaze = Input(shape=(self.time_steps, 30), name='input_gaze')

        model_l1 = AttentionLSTM(self.time_steps).generate()
        # make all layers of model_part non trainable
        if self.train_pth:
            model_part.load_weights(self.train_pth)
        outputs_part = model_part([input_video, input_vehicle, input_gaze])
        
        classifier = Dense(
            7,
            activation="softmax",
            kernel_initializer="VarianceScaling"
        )(outputs_part)

        modell2 = tf.keras.Model(inputs=[input_video, input_vehicle, input_gaze], outputs=[classifier])

        return modell2, modell1

#model = AttentionLSTM(15).generate()
#print(model.summary())
