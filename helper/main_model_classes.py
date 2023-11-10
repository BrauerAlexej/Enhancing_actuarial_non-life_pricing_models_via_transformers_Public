import tensorflow as tf
import math as m
from typing import Tuple, Dict
import numpy as np

@tf.function
def reglu(x):
    """The ReGLU activation function from [1].

    References:

        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = tf.split(x, num_or_size_splits=2, axis=-1)
    # return a * tf.nn.relu(b)
    return tf.multiply(a, tf.nn.relu(b))


class ReGLU(tf.keras.layers.Layer):
    def __init__(self,name: str = "ReGlU_activation",**kwargs):
        super(ReGLU, self).__init__(name=name)
        """
        ReGLU Activation Layer
        represents the ReGLU activation function from [1].

        It splits ith lust dim in half and applies relu to the second half and multiplies it with the first half.
        
        References:
            [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    def forward(self, x):
        return reglu(x)
    def call(self, x):
        return reglu(x)


class Embedding_Nr_Features(tf.keras.Model):
    """Linear embedding for the numerical features.

    The code ideas are based on the code from [gorishniy2021revisiting]
    See the paper for more details: https://arxiv.org/abs/2106.11959 
    See here their code written for torch nn's: https://github.com/Yura52/rtdl
        
    * **Input shape**: ``(batch_size, count_nr_features)``
    * **Output shape**: ``(batch_size, count_nr_features, emb_dim)``
    
    Args:
        nr_features (list): the list of numerical feature columns that should go into the embedding
        emb_dim (int): the embedding dimension (every feature will be embedded into this dimension)
        seed_nr (int): sets the random seed for the random uniform initializer
    
    For each feature we create a linear embedding layer of dim emb_dim. 
    The result is that, for every feature a linear embedding layer with emb_dim neurons is created. 
    (feature values of the j'th feature x_j (R^1) are thereby transformed to W_j*xj + b_j where W_j and b_j are trainable in R^emb_dim)
    
    References:
        * [gorishniy2021revisiting]  Gorishniy, Rubachev, Khrulkov, Babenko "Revisiting Deep Learning Models for Tabular Data" 2021 
    """

    def __init__(
        self,
        nr_features: list,
        emb_dim: int = 32,
        seed_nr: int = 42,
        name: str = "Embedding_Layers_Nr_Features"
    ):
        super(Embedding_Nr_Features, self).__init__(name=name)
        
        self.count_features = len(nr_features)
        self.nr_features = nr_features
        self.emb_dim = emb_dim
        self.seed_nr = seed_nr
        self.init_val = 1 / m.sqrt(self.emb_dim)
        init_rand = tf.random_uniform_initializer(- self.init_val, self.init_val, seed=self.seed_nr)
        
        # initialize the weights and biases for the linear embedding as described in the paper:
        self.linear_w = tf.Variable(initial_value=init_rand(shape=(self.count_features, self.emb_dim), # features, emb_dim 
                                                            dtype='float32'), 
                                    trainable=True)
        # as far as I understood in the paper, all emb_dim have there own bias 
        self.linear_b = tf.Variable(init_rand(shape=(self.count_features, self.emb_dim), # features, n_bins, all emb_dim have there own bias 
                                              dtype='float32'), 
                                    trainable=True)
    
    def call(self, x):
        '''
        * **Input shape**: ``(batch_size, count_nr_features)``
        * **Output shape**: ``(batch_size, count_nr_features, emb_dim)``
           
        Returns:
            Linear embedding model for the numerical features (tf.tensor)
        '''
        
        # all the x values are multiplied with the weights (one x value gets multiplied by emb_dim different weights) and then the bias is added.
        # all emb_dim have there own bias 
        x_extended = tf.tile(tf.expand_dims(x, axis=-1), multiples=[1, 1, self.emb_dim])
        nr_embs = x_extended * self.linear_w + self.linear_b
        # NOTE that i saw in another code on github that used here a relu activation function, but I think that they didn't used it in the paper
        # so I used here no activation function: embs = tf.nn.relu(embs)
        return nr_embs

class Embedding_Cat_Features(tf.keras.Model):
    """Embedding for the categorical features.

    The code ideas are based on the code from [gorishniy2021revisiting]
    See the paper for more details: https://arxiv.org/abs/2106.11959 
    See here their code written for torch nn's: https://github.com/Yura52/rtdl
        
    * **Input shape**: ``(batch_size, count_cat_features)``
    * **Output shape**: ``(batch_size, count_cat_features, emb_dim)``
    
    Args:
        cat_features (list): the list of cat feature columns that should go into the embedding
        cat_vocabulary (dict): dict that contains for every feature in cat_features the vocabulary for the corresponding feature
        emb_dim (int): the embedding dimension (every feature will be embedded into this dimension)
        seed_nr (int): sets the random seed for the random uniform initializer
    
    For each feature we create lookup table via the given cat_vocabulary (transformation from string to int).
    Then we create for every feature then create a linear embedding layer of dim emb_dim. 
    The result is that, for every possible feature value a linear embedding layer with emb_dim neurons is created. 
    (The Idea works like this:  
        the j'th feature has k_j possible values
        then the j'th feature is transformed to a vector of dim k_j (one-hot encoding)
        then the one-hot encoded vector is multiplied with a matrix W_j of dim k_j x emb_dim
        then the result is a vector of dim emb_dim
        then a bias b_j of dim emb_dim is added to the vector. 
        So just like the linear embedding layer above, 
        but there is x_j for every possible feature value and it has either the value 0 or 1 (one-hot encoding)
        Therefore there are for every feature: #possible-feature-values*#emb_dim weights and #emb_dim bias's to train). 
    
    References:
        * [gorishniy2021revisiting]  Gorishniy, Rubachev, Khrulkov, Babenko "Revisiting Deep Learning Models for Tabular Data" 2021 
    """
    
    def __init__(
        self,
        cat_features: list,
        cat_vocabulary: dict,
        emb_dim: int = 32,
        seed_nr: int = 42,
        name: str = "Embedding_Layers_Cat_Features"
    ):
                
        super(Embedding_Cat_Features, self).__init__(name=name)
        self.cat_features = cat_features
        self.emb_dim = emb_dim
        self.seed_nr = seed_nr
        self.init_val = 1 / m.sqrt(self.emb_dim)
        init_rand = tf.random_uniform_initializer(- self.init_val, self.init_val, seed=self.seed_nr)
        
        self.c_lookup_layers = {}
        self.c_emb_layers = {}
        for c in self.cat_features:
            # for each feature create a lookup table via the given cat_vocabulary (transformation from string to int).
            self.c_lookup_layers[c] = tf.keras.layers.StringLookup(vocabulary=cat_vocabulary[c])
            # for each cat feature create a embedding layer of dim emb_dim.
            self.c_emb_layers[c] = tf.keras.layers.Embedding(input_dim=self.c_lookup_layers[c].vocabulary_size(), 
                                                             embeddings_initializer = init_rand,
                                                             output_dim=self.emb_dim)
                 
    def call(self, x):
        '''
        * **Input shape**: ``(batch_size, count_cat_features)``
        * **Output shape**: ``(batch_size, count_cat_features, emb_dim)``
           
        Returns:
            Embedding model for categorical features (tf.tensor)
        '''        
        
        cat_embs = []
        # NOTE for the following to work, the order of the features in the list self.cat_features 
        # must be the same as the order of the features in the tensor x
        for i, c in enumerate(self.cat_features):
            cat_embs.append(self.c_emb_layers[c](self.c_lookup_layers[c](x[:, i])))

        return tf.stack(cat_embs, axis=1)

class Embedding_CLS_token(tf.keras.Model):
    """Adding for the cls token to the stack of tokenized features.

    The code ideas are based on the code from [gorishniy2021revisiting]
    See the paper for more details: https://arxiv.org/abs/2106.11959 
    See here their code written for torch nn's: https://github.com/Yura52/rtdl
    
    Args:
        emb_dim (int): the embedding dimension (every feature will be embedded into this dimension)
        seed_nr (int): sets the random seed for the random uniform initializer
        
    * **Input shape**: ``(batch_size, count_features, emb_dim)``
    * **Output shape**: ``(batch_size, count_features + 1, emb_dim)``
     
    References:
        * [gorishniy2021revisiting]  Gorishniy, Rubachev, Khrulkov, Babenko "Revisiting Deep Learning Models for Tabular Data" 2021 
    """
    def __init__(
        self,
        emb_dim: int = 32,
        seed_nr: int = 42,
        name: str = "Embedding_Layer_Add_CLS_token"
    ):
        super(Embedding_CLS_token, self).__init__(name=name)
        self.emb_dim = emb_dim
        self.seed_nr = seed_nr
        self.init_val = 1 / m.sqrt(self.emb_dim)
        init_rand = tf.random_uniform_initializer(- self.init_val, self.init_val, seed=self.seed_nr)
        # create the trainable weights for the cls-token:
        self.cls_w = tf.Variable(initial_value=init_rand(shape=(1, 1, self.emb_dim), dtype='float32'),  # shape (1, 1, emb_dim)
                            trainable=True)
    def call(self, x):
        # we repeat the cls-token weights cls_w (shape (1, 1, emb_dim)) for every sample in the batch: 
        # and then we add the cls-token to the stack of tokenized features (bottom of the stack) 
        # NOTE: maybe add it to the top of the stack, but then change the output of the ft-transformer?
        return tf.concat([x,tf.tile(self.cls_w, [tf.shape(x)[0], 1, 1])], axis=1)


class FT_Transformer_MultiheadAttention(tf.keras.Model): # (tf.keras.layers.Layer):
    """Multihead Attention Layer

    The code is based on the code from [gorishniy2021revisiting]
    See the paper for more details: https://arxiv.org/abs/2106.11959 
    See here their code written for torch nn's: https://github.com/Yura52/rtdl
    And original idea are from the paper [vaswani2017attention] https://arxiv.org/abs/1706.03762
    and paper [devlin2018bert] : https://arxiv.org/abs/1810.04805
    To understand the multihead attention layer better see also [thickstun2022equations]. 
    
    * **Input shape**: ``(batch_size, n_tokens, emb_dim)``
    * **Output shape**: ``(batch_size, n_tokens, emb_dim)``
  
    References:
        * [gorishniy2021revisiting]  Gorishniy, Rubachev, Khrulkov, Babenko "Revisiting Deep Learning Models for Tabular Data" 2021 
        * [vaswani2017attention]  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need", NeurIPS 2017
        * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
        * [thickstun2022equations] John Thickstun "The Transformer Model in Equations" 2022
    """

    def __init__(
        self,
        d_model: int,
        attention_n_heads: int,
        attention_dropout: float,
        bias_bool: bool,
        name: str = "Multihead_Attention_Layer"
    ) -> None:

        super().__init__(name=name)
        if attention_n_heads > 1:
            assert d_model % attention_n_heads == 0, 'NOTE: we are using for d_key, d_value as d_model/attention_n_heads (as in [vaswani2017attention]) so: d_model must be a multiple of attention_n_heads'

        self.d_model = d_model
        self.init_val = 1 / m.sqrt(self.d_model)
        init_rand_W_q = tf.random_uniform_initializer(- self.init_val, self.init_val)
        init_rand_W_k = tf.random_uniform_initializer(- self.init_val, self.init_val)
        init_rand_W_v = tf.random_uniform_initializer(- self.init_val, self.init_val)
        init_rand_W_out = tf.random_uniform_initializer(- self.init_val, self.init_val)

        self.W_q = tf.keras.layers.Dense(d_model, "linear",bias_bool,kernel_initializer=init_rand_W_q,bias_initializer='zeros', name="W_q") 
        self.W_k = tf.keras.layers.Dense(d_model, "linear",bias_bool,kernel_initializer=init_rand_W_k,bias_initializer='zeros', name="W_k")
        self.W_v = tf.keras.layers.Dense(d_model, "linear",bias_bool,kernel_initializer=init_rand_W_v,bias_initializer='zeros', name="W_v")
        self.W_out = tf.keras.layers.Dense(d_model, "linear",bias_bool,kernel_initializer=init_rand_W_out,bias_initializer='zeros',name="W_out") if attention_n_heads > 1 else None
        self.attention_n_heads = attention_n_heads
        self.dropout = tf.keras.layers.Dropout(attention_dropout)

    def _reshape(self, x: tf.Tensor) -> tf.Tensor:
        batch_size, n_tokens, d_model = x.shape
        d_key = d_model // self.attention_n_heads
        return tf.reshape(
            tf.transpose(tf.reshape(x,(-1, n_tokens, self.attention_n_heads, d_key)), # Reshape to: (batch_size, n_tokens, n_heads, d_key)
                         perm=[0, 2, 1, 3]), #change to: (batch_size, n_heads, n_tokens, d_key)
            (-1, n_tokens, d_key)) # Reshape to: (batch_size * n_heads, n_tokens, d_key)

    def call(
        self,
        x_q: tf.Tensor,
        x_kv: tf.Tensor,
    ) -> tf.Tensor: # Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Perform the forward pass.
        Args:
            x_q: query tokens
            x_kv: key-value tokens
        Returns:
            tokens
        """
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.attention_n_heads == 0, 'NOTE: we are using for d_key, d_value as d_model/attention_n_heads (as in [vaswani2017attention]) so: the last dimension (d_model or emb_dim) must be a multiple of attention_n_heads'

        batch_size = q.shape[0]
        d_head_key = k.shape[-1] // self.attention_n_heads
        d_head_value = v.shape[-1] // self.attention_n_heads
        n_q_tokens = q.shape[1]
        # convert d_head_key to float to avoid type mismatch in tf.sqrt
        d_head_key = tf.cast(d_head_key, tf.float32)
        
        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = tf.matmul(q, tf.transpose(k, (0, 2, 1))) / tf.sqrt(d_head_key) # see [vaswani2017attention] eq. 1
        attention_probs = tf.nn.softmax(attention_logits, axis=-1)
        attention_probs = self.dropout(attention_probs)
        x = tf.matmul(attention_probs, self._reshape(v))
        x = tf.reshape(tf.transpose(tf.reshape(x, 
                                               (-1, self.attention_n_heads, n_q_tokens, d_head_value)), # Reshape to: (batch_size, n_heads, n_q_tokens * d_head_value)
                                    perm=[0, 2, 1, 3]),
                        (-1, n_q_tokens, self.attention_n_heads * d_head_value)) # Reshape to: (batch_size, n_q_tokens, n_heads * d_head_value)
        
        if self.W_out is not None:
            x = self.W_out(x)
        return x # NOTE: Maybe also output those if needed for interpretability: {'attention_logits': attention_logits,'attention_probs': attention_probs,}
        


class Block_4_FT_Transformer(tf.keras.Model):
    '''
    The code ideas are based on the code from [gorishniy2021revisiting]
    See the paper for more details: https://arxiv.org/abs/2106.11959 
    See here their code written for torch nn's: https://github.com/Yura52/rtdl
    And original idea is from the paper [vaswani2017attention] https://arxiv.org/abs/1706.03762
    This github repo helped me also here to get started: https://github.com/aruberts/TabTransformerTF
        
    * **Input shape**: ``(batch_size, n_tokens, emb_dim)``
    * **Output shape**: ``(batch_size, n_tokens, emb_dim)``
    
    Hereby is in our case n_tokens equal to count_features + 1 (for cls token)
    
    Args:
        emb_dim (int): the embedding dimension (every feature is embedded into this dimension)
        attention_n_heads (int): number of attention heads
        ffn_d_hidden (int): the size of the second layer of the ffn in a transformer (the input and output layer are fix)
        attention_dropout (float): the dropout rate for attention 
        ffn_dropout (float): the dropout rate for the hidden ffn layer
        prenormalization (bool): should prenormalization (True) be used or postnormalization (False)
        first_prenormalization (bool): should the first layer be prenormalized (True) or not (False) - see note below.
        ffn_activation_ReGLU (bool): should the activation function in the ffn be ReGLU (True) or GELU (False)
        seed_nr (int): sets the random seed for the dropout layers       
        
    This Transformer-Block is based on the one in the paper [gorishniy2021revisiting]: 
    Pseudo-Code: was taken from the paper: 
        ``Block(x) =  ResidualPreNorm(FFN, ResidualPreNorm(MHSA, x))``
        ``ResidualPreNorm(Module, x) = x + Dropout(Module(Norm(x)))``
        ``FFN(x) = Linear(Dropout(Activation(Linear(x))))``
        
    NOTE: In [gorishniy2021revisiting] and in the github repo they say that for prenormalization the common belief is that it enables easier optimization, but sometimes at the cost of worse results. 
          They say that they found that prenorm works better for tabular data. But the write also: 
          "In the PreNorm setting, we also found it to be necessary to remove the first normalization from the first Transformer layer to achieve good performance."       
    
    Regarding the default settings:
    In the paper they write: 
        * As norm they used LayerNorm as one typically does in the transformer usecase (not BatchNorm) 
        * MHSA, they set nheads = 8 and did not tune this parameter
        * For Activation they used ReGLU activation, since it is reported to be superior to the usually used GELU activation (but not big difference to RELU in experiments). 
        * I reimplemented the ReGLU layer and class for tf since it was not in the tf package (see above).-> Note this layer changes the number of neurons in half.
        * As dimension for the second layer of the FFN they used 4/3*emb_dim in the default settings
        * Dropout rates are set almost always to non-zero by the tuning (for att/ffn/final) => We used here Defaults: (0.2/0.1/0.0)
        * For the Initialization they used Kaiming (He et al., 2015a) [nn.init.uniform_(weight, a=-d_sqrt_inv, b=d_sqrt_inv)]
    
    References:
        * [gorishniy2021revisiting]  Gorishniy, Rubachev, Khrulkov, Babenko "Revisiting Deep Learning Models for Tabular Data" 2021 
        * [vaswani2017attention]  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need", NeurIPS 2017
    ''' 

    def __init__(
        self,
        emb_dim: int,
        attention_n_heads: int = 8,
        ffn_d_hidden: int = None,
        ffn_activation_ReGLU: bool = True,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        prenormalization: bool = True,
        first_prenormalization: bool = False,
        seed_nr: int = 42, 
        name: str = "FT_Transformer_Block"        
    ):
        super(Block_4_FT_Transformer, self).__init__(name=name)
        self.emb_dim = emb_dim
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization
        self.seed_nr = seed_nr
        self.ffn_activation_ReGLU = ffn_activation_ReGLU
        if ffn_d_hidden is None:
            self.ffn_d_hidden = int(4/3*emb_dim)
        else:
            self.ffn_d_hidden = ffn_d_hidden
        
        self.init_val = 1 / m.sqrt(self.emb_dim)

        init_rand_1_weight = tf.random_uniform_initializer(- self.init_val, self.init_val)
        init_rand_2_weight = tf.random_uniform_initializer(- self.init_val, self.init_val)
        init_rand_1_bias = tf.random_uniform_initializer(- self.init_val, self.init_val)
        init_rand_2_bias = tf.random_uniform_initializer(- self.init_val, self.init_val)               
        
        # define all the layers needed for a transformer block:
        self.MHSA_layer = FT_Transformer_MultiheadAttention(d_model=self.emb_dim,
                                                            attention_n_heads=self.attention_n_heads,
                                                            attention_dropout=self.attention_dropout,
                                                            bias_bool=True,
                                                            name="MHSA_for_transformer_block")
        # NOTE: Just use code below if one want to use the keras implementation of the MHSA_Layer (has the same number of parameters) 
        ''' 
        self.MHSA_layer = tf.keras.layers.MultiHeadAttention(num_heads=self.attention_n_heads, 
                                            key_dim= int(self.emb_dim / self.attention_n_heads),  
                                            # NOTE: as in the paper of Vaswani "Attention Is All You Need" 2017 we use here for the key, query and value
                                            # And we use as in the paper described "h = 8 parallel attention layers, or heads." 
                                            # "For each of these we use dk = dv = dmodel/h"
                                            dropout=self.attention_dropout,
                                            name="MHSA_for_transformer_block")        
        '''
        
         
        self.add_residual_1 = tf.keras.layers.Add(name="addition_input_and_MHSA_output")
        if self.first_prenormalization:
            self.norm_layer_1 = tf.keras.layers.LayerNormalization(name="first_LayerNormalization", epsilon=1e-5,scale = True, center = True) 
        
        if self.ffn_activation_ReGLU:    
            self.mlp = tf.keras.Sequential(
                [   
                    tf.keras.layers.Dense(self.ffn_d_hidden*2, activation="linear",kernel_initializer=init_rand_1_weight, bias_initializer=init_rand_1_bias, name="mlp_linear_layer_1"),   # NOTE: before gelu activation
                    # NOTE: here is ffn_d_hidden*2 because ReGlU splits the number of neurons in half.
                    ReGLU(name="mlp_ReGLU_activation_layer"),
                    tf.keras.layers.Dropout(self.ffn_dropout,name="mlp_dropout_layer"),
                    tf.keras.layers.Dense(self.emb_dim, activation= "linear",kernel_initializer=init_rand_2_weight, bias_initializer=init_rand_2_bias, name="mlp_linear_layer_2")
                ]
            , name="mlp_for_transformer_block")
        else:
            self.mlp = tf.keras.Sequential(
                [   
                    tf.keras.layers.Dense(self.ffn_d_hidden, activation="gelu", name="mlp_linear_layer_1"),
                    tf.keras.layers.Dropout(self.ffn_dropout,name="mlp_dropout_layer"),
                    tf.keras.layers.Dense(self.emb_dim, activation= "linear", name="mlp_linear_layer_2")
                ]
            , name="mlp_for_transformer_block")
        
        self.norm_layer_2 = tf.keras.layers.LayerNormalization(name="LayerNormalization_after_sum_input_MHSA", epsilon=1e-5,scale = True, center = True)
        self.add_residual_2 = tf.keras.layers.Add(name="addition_Input_MHSA_output_and_mlp_output")

    def call(self, inputs):
        # prenormalization
        if self.prenormalization:
            if self.first_prenormalization:
                norm_inputs = self.norm_layer_1(inputs)
            else: 
                norm_inputs = inputs
            MHSA_output = self.MHSA_layer(norm_inputs, norm_inputs) 
            # NOTE: i think here should be maybe a dropout layer applied on MHSA_output (See in the paper appendix E.1.)? 
            add_Input_MHSA_output = self.add_residual_1([inputs, MHSA_output]) # NOTE: maybe here with input_norm as first input?
            norm_add_Input_MHSA_output = self.norm_layer_2(add_Input_MHSA_output)
            ffn_output = self.mlp(norm_add_Input_MHSA_output) 
            # NOTE: i think here should be maybe a dropout layer applied on ffn_output (See in the paper appendix E.1.)? 
            transf_output = self.add_residual_2([ffn_output, add_Input_MHSA_output])
        # postnormalization
        else:
            raise ValueError('postnormalization is not jet implemented')
        return transf_output

    
class Feature_Tokenizer_Transformer(tf.keras.Model):
    ''' Feature Tokenizer Transformer Model:
    
    The code ideas are based on the code from [gorishniy2021revisiting]
    See the paper for more details: https://arxiv.org/abs/2106.11959 
    See here their code written for torch nn's: https://github.com/Yura52/rtdl
    And original idea is from the paper [vaswani2017attention] https://arxiv.org/abs/1706.03762
    This github repo helped me also here to get started: https://github.com/aruberts/TabTransformerTF
        
    Args:
        emb_dim (int): the embedding dimension (every feature is embedded into this dimension)
        nr_features (list): the list of numerical feature columns that should go into the embedding
        cat_features (list): the list of cat feature columns that should go into the embedding
        cat_vocabulary (dict): dict that contains for every feature in cat_features the vocabulary for the corresponding feature
        count_transformer_blocks (int): number of transformer blocks
        attention_n_heads (int): number of attention heads
        attention_dropout (float): the dropout rate for attention 
        ffn_d_hidden (int): the size of the second layer of the ffn in a transformer (the input and output layer are fix)
        ffn_dropout (float): the dropout rate for the hidden ffn layer
        ffn_activation_ReGLU (bool): should the activation function in the ffn be ReGLU (True) or GELU (False)
        prenormalization (bool): should prenormalization (True) be used or postnormalization (False)
        output_dim (int): output dimension of the model
        last_activation (str): activation function applied to the last dense layer
        last_layer_initial_weights (str): string (e.g. "zeros", "ones") initial weights for the last dense layer
        last_layer_initial_bias (str): string (e.g. "zeros", "ones") initial weights for the last dense layer
        exposure_name (str): name of the exposure column - if None, then no exposure column is used - if not None, then output is multiplied by the exposure tensor
        seed_nr (int): sets the random seed     
        
    NOTE: most of the heavy lifting is done by the class Block_4_FT_Transformer (for more details see there).
    
    
    Regarding the default settings (as in the paper done):
        * as default for the number of transformer blocks they used 3
        * As norm they used LayerNorm as one typically does in the transformer usecase (not BatchNorm) 
        * in the MHSA, they set nheads = 8 and did not tune this parameter
        * For Activation they used ReGLU activation, since it is reported to be superior to the usually used GELU activation (but not big difference to RELU in experiments). 
        * I reimplemented the ReGLU layer and class for tf since it was not in the tf package (see above).-> Note this layer changes the number of neurons in half.
        * As dimension for the second layer of the FFN they used 4/3*emb_dim in the default settings
        * Dropout rates are set almost always to non-zero by the tuning (for att/ffn/final) => We used here Defaults: (0.2/0.1/0.0)
        * For the Initialization we used also Kaiming [nn.init.uniform_(weight, a=-d_sqrt_inv, b=d_sqrt_inv)]
    
    
    References:
        * [gorishniy2021revisiting]  Gorishniy, Rubachev, Khrulkov, Babenko "Revisiting Deep Learning Models for Tabular Data" 2021 
        * [vaswani2017attention]  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need", NeurIPS 2017
    '''
    
    def __init__(
        self,
        emb_dim: int,
        nr_features: list = None,
        cat_features: list = None, 
        cat_vocabulary: dict = None,
        count_transformer_blocks: int = 3,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_d_hidden: int = None,
        ffn_dropout: float = 0.1,
        ffn_activation_ReGLU: bool = True,
        prenormalization: bool = True,
        output_dim: int = 1,
        last_activation: str = "linear",
        last_layer_initial_weights: str =  None,
        last_layer_initial_bias: str =  None,
        exposure_name: str = None,
        seed_nr: int = 42,
        name: str = "Feature_Tokenizer_Transformer_Model"        
    ): 
    
        super(Feature_Tokenizer_Transformer, self).__init__(name=name)
        self.emb_dim = emb_dim
        self.nr_features = nr_features
        self.cat_features = cat_features
        self.cat_vocabulary = cat_vocabulary
        self.count_transformer_blocks = count_transformer_blocks
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation_ReGLU = ffn_activation_ReGLU
        self.prenormalization = prenormalization
        self.output_dim = output_dim
        self.last_activation = last_activation
        self.last_layer_initial_weights = last_layer_initial_weights
        self.last_layer_initial_bias = last_layer_initial_bias
        self.exposure_name = exposure_name
        self.seed_nr = seed_nr
        
        if self.last_layer_initial_weights is not None:
            self.init_rand_kernel = self.last_layer_initial_weights
        else: 
            self.init_val = 1 / m.sqrt(self.emb_dim)
            self.init_rand_kernel = tf.random_uniform_initializer(- self.init_val, self.init_val)
            
        if self.last_layer_initial_weights is not None:
            self.init_rand_bias = self.last_layer_initial_bias
        else:
            self.init_rand_bias = tf.random_uniform_initializer(- self.init_val, self.init_val)
        
        # since it is good practice to have the input tensors lower case will do that: 
        if self.nr_features is not None:
            self.nr_features = [c.lower() for c in nr_features] # lower case all the nr_features
        if self.cat_features is not None:  
            self.cat_features = [c.lower() for c in cat_features] # lower case all the nr_features
        if self.cat_vocabulary is not None:
            self.cat_vocabulary = {k.lower(): v for k, v in cat_vocabulary.items()} # lowercase all the keys in the cat_vocabulary dict:
        if self.exposure_name  is not None:
            self.exposure_name = exposure_name.lower()
        
        # Quick Checks for the input (not all checks are done here, but at least some):
        if (self.nr_features is not None) and (self.cat_features is not None):
            if (len(self.nr_features) == 0) and (len(self.cat_features) == 0):
                raise ValueError('The input must have at least one numerical or categorical feature')
        if (self.cat_features is not None):
            if len(self.cat_features) > 0 and self.cat_vocabulary is None:
                raise ValueError('If cat_features are given, then also the cat_vocabulary must be given')
        if self.exposure_name is not None and self.output_dim != 1:
            raise ValueError('If exposure_name is not None, then output_dim must be 1')
                
        # Nr. Embeddings:
        if len(self.nr_features) > 0:
            self.nr_embedding = Embedding_Nr_Features(self.nr_features,self.emb_dim) 

        # Cat. Embeddings:
        if len(self.cat_features) > 0:
            self.cat_embedding = Embedding_Cat_Features(self.cat_features,self.cat_vocabulary,self.emb_dim)

        # Embedding to add CLS-Token:
        self.cls_token_embedding = Embedding_CLS_token(self.emb_dim, self.seed_nr)
        
        # List of Transformer Blocks:
        self.transformer_block_list = []
        for i in range(self.count_transformer_blocks):
            if i == 0:
                self.first_prenormalization = False
            else: 
                self.first_prenormalization = True
            # Regarding the first prenormalization see paper (or comments in Block_4_FT_Transformer):
            self.transformer_block_list.append(Block_4_FT_Transformer(
                                                    emb_dim = self.emb_dim,
                                                    attention_n_heads = self.attention_n_heads,
                                                    ffn_d_hidden = self.ffn_d_hidden,
                                                    attention_dropout = self.attention_dropout,
                                                    ffn_dropout = self.ffn_dropout,
                                                    ffn_activation_ReGLU = self.ffn_activation_ReGLU,
                                                    prenormalization = self.prenormalization,
                                                    first_prenormalization = self.first_prenormalization,
                                                    seed_nr = self.seed_nr,
                                                    name=f"FT_Transformer_Block_{i}"
                                                ))
        # FT_Transformer_Blocks is the combination of the transformer blocks:
        self.FT_Transformer_Blocks = tf.keras.Sequential(self.transformer_block_list, name="FT_Transformer_Blocks")
        
        
        # prediction module: As described in the paper we use just the cls-token and the same prediction-module: 
        # linear(relu(layer_norm(cls-token)))
        self.prediction_module = tf.keras.Sequential(
            [   
                tf.keras.layers.LayerNormalization(name="LayerNormalization_for_Prediction_Module", epsilon=1e-5,scale = True, center = True),
                tf.keras.layers.ReLU(name="ReLU_for_Prediction_Module"),
                tf.keras.layers.Dense(self.output_dim, activation=self.last_activation,kernel_initializer=self.init_rand_kernel,bias_initializer=self.init_rand_bias,
                                      name="Dense_for_Prediction_Module")
            ], name = "Prediction_Module"
        )
        
        if self.exposure_name  is not None:
            self.multiply_layer = tf.keras.layers.Multiply(name="Multiply_with_Exposure")
    
    def get_config(self):
        config = super(Feature_Tokenizer_Transformer, self).get_config()
        config.update({
            'emb_dim': self.emb_dim,
            'nr_features': self.nr_features,
            'cat_features': self.cat_features,
            'cat_vocabulary': self.cat_vocabulary,
            'count_transformer_blocks': self.count_transformer_blocks,
            'attention_n_heads': self.attention_n_heads,
            'attention_dropout': self.attention_dropout,
            'ffn_d_hidden': self.ffn_d_hidden,
            'ffn_dropout': self.ffn_dropout,
            'ffn_activation_ReGLU': self.ffn_activation_ReGLU,
            'prenormalization': self.prenormalization,
            'output_dim': self.output_dim,
            'last_activation': self.last_activation,
            'last_layer_initial_weights': self.last_layer_initial_weights,
            'last_layer_initial_bias': self.last_layer_initial_bias,
            'exposure_name': self.exposure_name,
            'seed_nr': self.seed_nr
        })
        return config
    
        
    def call(self, inputs):
        # Create the input for the ft-transformer:
        # -----------------
        # numerical features:
        if (self.nr_features is not None) and (len(self.nr_features) > 0):
            num_input = tf.concat([inputs[c] for c in self.nr_features], axis=1) 
            # inputs[n] is a tensor of shape (batch_size, 1)
            # num_input is a tensor of shape (batch_size, count_numerical_features)
            tokenized_features_nr = self.nr_embedding(num_input) # tokenized_features_nr is a tensor of shape (batch_size, count_numerical_features, emb_dim)
        
        # categorical features:
        if (self.cat_features is not None) and (len(self.cat_features) > 0):
            cat_input = tf.concat([inputs[c] for c in self.cat_features], axis=1) 
            # inputs[c] is a tensor of shape (batch_size, 1)
            # cat_input is a tensor of shape (batch_size, count_categorical_features)
            tokenized_features_cat = self.cat_embedding(cat_input) # tokenized_features_cat is a tensor of shape (batch_size, count_categorical_features, emb_dim)
        
        # stack all the inputs features together (actually concat them along the token axis, since stack would create a new axis):
        if (self.nr_features is not None) and (self.cat_features is not None):
            if (len(self.nr_features) != 0) and (len(self.cat_features) != 0):
                tokenized_features = tf.concat([tokenized_features_nr,tokenized_features_cat], axis=1) #tokenized_features is a tensor of shape (batch_size, count_features, emb_dim)
        if (self.cat_features is None) or (len(self.cat_features) == 0):
            tokenized_features = tokenized_features_nr #tokenized_features is a tensor of shape (batch_size, count_features, emb_dim)
        if (self.nr_features is None) or (len(self.nr_features) == 0):
            tokenized_features = tokenized_features_cat #tokenized_features is a tensor of shape (batch_size, count_features, emb_dim)            
        
        # Add cls-token: Input: (batch_size, count_features, emb_dim) -> Output: (batch_size, count_features + 1, emb_dim)
        tokenized_features = self.cls_token_embedding(tokenized_features)
        
        # Go through the FT_Transformer_Blocks:
        # -----------------
        tokenized_features_output = self.FT_Transformer_Blocks(tokenized_features)
        
        # As described in the paper we use just just cls-token for the prediction and the same prediction fct: linear(relu(layer_norm(cls-token)))
        # if we added cls token on top:
        # output = self.prediction_module(tokenized_features_output[:, 0, :]) # transformer_inputs[:, 0, :] gets us the first token (there fore the cls-token) and the the resulting tensor has shape (batch_size, emb_dim)
        # if we added cls token on top: NOTE: maybe add it to the top of the stack?
        output = self.prediction_module(tokenized_features_output[:, -1, :])  # transformer_inputs[:, -1, :] gets us the last token (there fore the cls-token) and the the resulting tensor has shape (batch_size, emb_dim)
        
        if self.exposure_name is not None:
            output = self.multiply_layer([output, inputs[self.exposure_name]])
            return {"output": output}
        else:
            return {"output": output} 





    
#------------------------------------------------
# LocalGLM - Feature Tokenizer Transformer Model:    
#------------------------------------------------

class GLM_Embedding_Nr_Features(tf.keras.Model):
    """GLM Linear embedding for the numerical features.
    
        Gets the numerical features and multiplies them with the glm betas for the numerical features.
        
        * **Input shape**: ``(batch_size, count_nr_features)``
        * **Output shape**: ``(batch_size, count_nr_features)``
    
    Args:
        nr_features (list): the list of numeric feature columns that should go into the embedding
        init_glm_nr_col_weights (np.ndarray): array that contains for every feature in nr_features the initial weights for the GLM embedding layer
        trainable_glm_emb (bool): should the GLM embedding layer be trainable or not
        
    """

    def __init__(
        self,
        nr_features: list,
        init_glm_nr_col_weights: np.ndarray = None,
        trainable_glm_emb: bool = True,
        name: str = "GLM_Embedding_Layers_Nr_Features"
    ):
        super(GLM_Embedding_Nr_Features, self).__init__(name=name)
        
        self.count_features = len(nr_features)
        self.nr_features = nr_features
        self.init_glm_nr_col_weights = init_glm_nr_col_weights
        self.trainable_glm_emb = trainable_glm_emb
        
        if init_glm_nr_col_weights is None:
            self.linear_w = tf.expand_dims(tf.ones((self.count_features,)), axis=0)
            self.linear_w = tf.Variable(initial_value=self.linear_w, trainable=self.trainable_glm_emb)
        else: 
            self.linear_w = tf.expand_dims(tf.convert_to_tensor(init_glm_nr_col_weights), axis=0) 
            self.linear_w = tf.Variable(initial_value=self.linear_w, trainable=self.trainable_glm_emb)

        self.multiply_layer = tf.keras.layers.Multiply(name="emb_multiply")
        
    def get_config(self):
        config = super(GLM_Embedding_Nr_Features, self).get_config()
        config.update({
            'nr_features': self.nr_features,
            'init_glm_nr_col_weights': self.init_glm_nr_col_weights,
            'trainable_glm_emb': self.trainable_glm_emb
        })
        return config
    
    def call(self, x):
        '''
        * **Input shape**: ``(batch_size, count_nr_features)``
        * **Output shape**: ``(batch_size, count_nr_features)``
           
        Returns:
            Linear embedding model for the numerical features (tf.tensor)
        '''
        # all the x values are multiplied with the glm weights
        linear_w_tiled = tf.tile(self.linear_w, [tf.shape(x)[0], 1])
        
        nr_embs = self.multiply_layer([x, linear_w_tiled])
        # NOTE that i saw in another code on github that used here a relu activation function, but I think that they didn't used it in the paper
        # so I used here no activation function: embs = tf.nn.relu(embs)
        return nr_embs
        
        
class GLM_Embedding_Cat_Features(tf.keras.Model):
    """GLM Embedding for the categorical features.
    
        Gets the categorical features and multiplies them with the glm betas for the categorical features.
        
        * **Input shape**: ``(batch_size, count_nr_features)``
        * **Output shape**: ``(batch_size, count_nr_features)``
    
    Args:
        cat_features (list): the list of categorical feature columns that should go into the embedding
        cat_vocabulary (dict): dict that contains for every feature in cat_features the vocabulary for the corresponding feature
        init_glm_cat_col_weights (dict): dict that contains for every feature in cat_features the initial weights for the GLM embedding layer
        trainable_glm_emb (bool): should the GLM embedding layer be trainable or not
    """
    def __init__(
        self,
        cat_features: list,
        cat_vocabulary: dict,
        init_glm_cat_col_weights: np.ndarray = None,
        trainable_glm_emb: bool = True,
        name: str = "GLM_Embedding_Layers_Cat_Features"
    ):
                
        super(GLM_Embedding_Cat_Features, self).__init__(name=name)
        self.cat_features = cat_features
        self.init_glm_cat_col_weights = init_glm_cat_col_weights
        self.cat_vocabulary = cat_vocabulary
        self.trainable_glm_emb = trainable_glm_emb
        self.c_lookup_layers = {}
        self.c_emb_layers = {}
        for c in self.cat_features:
            # for each feature create a lookup table via the given cat_vocabulary (transformation from string to int).
            # NOTE: that currently the out of vocabulary strings (oov) are mapped to the the first string. maybe better give oov token as an input?
            self.c_lookup_layers[c] = tf.keras.layers.StringLookup(vocabulary=cat_vocabulary[c],name=f"glm_StringLookup_{c}",oov_token=cat_vocabulary[c][0])
            # for each cat feature create a embedding layer
            if self.init_glm_cat_col_weights is None:
                self.c_emb_layers[c] = tf.keras.layers.Embedding(input_dim=self.c_lookup_layers[c].vocabulary_size(), 
                                                             embeddings_initializer = "ones",
                                                             output_dim=1, trainable=True, name=f"glm_emb_{c}") 
            else: 
                self.c_emb_layers[c] = tf.keras.layers.Embedding(input_dim=self.c_lookup_layers[c].vocabulary_size(), 
                                                             weights=[np.array(init_glm_cat_col_weights[c]).reshape(-1, 1)],
                                                             output_dim=1, trainable=True, name=f"glm_emb_{c}") 
    def get_config(self):
        config = super(GLM_Embedding_Cat_Features, self).get_config()
        config.update({
            'cat_features': self.cat_features,
            'cat_vocabulary': self.cat_vocabulary,
            'init_glm_cat_col_weights': self.init_glm_cat_col_weights,
            'trainable_glm_emb': self.trainable_glm_emb
        })
        return config
                 
    def call(self, x):
        '''
        * **Input shape**: ``(batch_size, count_cat_features)``
        * **Output shape**: ``(batch_size, count_cat_features)``
           
        Returns:
            Embedding model for categorical features (tf.tensor)
        '''     
        
        cat_embs = []
        # NOTE for the following to work, the order of the features in the list self.cat_features 
        # must be the same as the order of the features in the tensor x
        for i, c in enumerate(self.cat_features):
            cat_embs.append(self.c_emb_layers[c](self.c_lookup_layers[c](x[:, i])))

        return tf.squeeze(tf.stack(cat_embs, axis=1), axis=-1)


class LocalGLM_FT_Transformer(tf.keras.Model):
    '''
    LocalGLM - Feature Tokenizer Transformer Model:
    
    This model is a combination of the two approaches LocalGLMnet and Feature Tokenizer Transformer Model
    So it is a LocalGLMnet as described in the paper [richman2023localglmnet]
    with the difference that the the neural network is replaced by a FT-Transformer-Block [gorishniy2021revisiting].
        
    Regarding the FT-Transformer-Block see the docstring of the class Feature_Tokenizer_Transformer.
            
    Args:
        emb_dim (int): the embedding dimension of the Feature_Tokenizer_Transformer (every feature is embedded into this dimension)
        nr_features (list): the list of numerical feature columns that should go into the embedding
        cat_features (list): the list of cat feature columns that should go into the embedding
        cat_vocabulary (dict): dict that contains for every feature in cat_features the vocabulary for the corresponding feature
        count_transformer_blocks (int): number of transformer blocks
        attention_n_heads (int): number of attention heads
        attention_dropout (float): the dropout rate for attention 
        ffn_d_hidden (int): the size of the second layer of the ffn in a transformer (the input and output layer are fix)
        ffn_dropout (float): the dropout rate for the hidden ffn layer
        ffn_activation_ReGLU (bool): should the activation function in the ffn be ReGLU (True) or GELU (False)
        prenormalization (bool): should prenormalization (True) be used or postnormalization (False)
        output_dim (int): output dimension of the model
        last_activation (str): activation function applied to the last dense layer
        last_layer_initial_weights (str): string (e.g. "zeros", "ones") initial weights for the last dense layer of the ft-transformer
        last_layer_initial_bias (str): string (e.g. "zeros", "ones") initial bias for the last dense layer of the ft-transformer
        init_glm_cat_col_weights (dict): dict that contains for every feature in cat_features the initial weights for the GLM embedding layer
        init_glm_nr_col_weights (np.ndarray): array that contains for every feature in nr_features the initial weights for the GLM embedding layer
        init_glm_bias (float): initial bias for the GLM embedding layer
        trainable_glm_emb (bool): should the GLM embedding layer be trainable or not
        exposure_name (str): name of the exposure column - if None, then no exposure column is used - if not None, then output is multiplied by the exposure tensor
        seed_nr (int): sets the random seed     
 
 
    References:
        * [richman2023localglmnet] Ronald Richman, Mario V. Wüthrich "LocalGLMnet: interpretable deep learning for tabular data" 2023
        * [gorishniy2021revisiting]  Gorishniy, Rubachev, Khrulkov, Babenko "Revisiting Deep Learning Models for Tabular Data" 2021 
        * [vaswani2017attention]  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need", NeurIPS 2017
    '''
        
    def __init__(
        self,
        emb_dim: int,
        nr_features: list = None,
        cat_features: list = None, 
        cat_vocabulary: dict = None,
        count_transformer_blocks: int = 3,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_d_hidden: int = None,
        ffn_dropout: float = 0.1,
        ffn_activation_ReGLU: bool = True,
        prenormalization: bool = True,
        output_dim: int = 1,
        last_activation: str = "linear",
        last_layer_initial_weights: np.ndarray = None,
        last_layer_initial_bias: np.ndarray = None,
        init_glm_cat_col_weights: dict = None,
        init_glm_nr_col_weights: np.ndarray = None,
        init_glm_bias: float = 0.0,
        trainable_glm_emb: bool = True,
        exposure_name: str = None,
        seed_nr: int = 42,
        name: str = "Local_GLM_Feature_Tokenizer_Transformer_Model"        
    ): 
    
        super(LocalGLM_FT_Transformer, self).__init__(name=name)
        self.emb_dim = emb_dim
        self.nr_features = nr_features
        self.cat_features = cat_features
        self.cat_vocabulary = cat_vocabulary
        self.count_transformer_blocks = count_transformer_blocks
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation_ReGLU = ffn_activation_ReGLU
        self.prenormalization = prenormalization
        self.output_dim = output_dim
        self.last_activation = last_activation
        self.last_layer_initial_weights = last_layer_initial_weights
        self.last_layer_initial_bias = last_layer_initial_bias
        self.init_glm_cat_col_weights = init_glm_cat_col_weights
        self.init_glm_nr_col_weights = init_glm_nr_col_weights
        self.init_glm_bias = init_glm_bias
        self.trainable_glm_emb = trainable_glm_emb
        self.exposure_name = exposure_name
        self.seed_nr = seed_nr
        
        self.dim_glm = len(nr_features) + len(cat_features)       
            
        # FT_transformer that has as many output dimensions as the nr of features:
        self.FT_transformer = Feature_Tokenizer_Transformer(
                    emb_dim = self.emb_dim, 
                    nr_features = self.nr_features,
                    cat_features = self.cat_features,
                    cat_vocabulary = self.cat_vocabulary,
                    count_transformer_blocks = self.count_transformer_blocks,
                    attention_n_heads = self.attention_n_heads,
                    attention_dropout = self.attention_dropout,
                    ffn_d_hidden = self.ffn_d_hidden, 
                    ffn_activation_ReGLU = self.ffn_activation_ReGLU,
                    ffn_dropout = self.ffn_dropout,
                    prenormalization = self.prenormalization,
                    output_dim = self.dim_glm, # NOTE: for the local glm transformer have the same output dimension as inputs for the local glm layer
                    last_activation = "linear", # NOTE: for the local glm transformer we use linear activation since apply the last activation in the local glm layer
                    exposure_name = None, # NOTE: for the local glm transformer we use no exposure since apply the exposure after local glm layer
                    last_layer_initial_weights = self.last_layer_initial_weights,
                    last_layer_initial_bias = self.last_layer_initial_bias,
                    seed_nr = self.seed_nr
            )
        
        
        # since it is good practice to have the input tensors lower case will do that: 
        if self.nr_features is not None:
            self.nr_features = [c.lower() for c in nr_features] # lower case all the nr_features
        if self.cat_features is not None:  
            self.cat_features = [c.lower() for c in cat_features] # lower case all the nr_features
        if self.cat_vocabulary is not None:
            self.cat_vocabulary = {k.lower(): v for k, v in cat_vocabulary.items()} # lowercase all the keys in the cat_vocabulary dict:
        if self.init_glm_cat_col_weights is not None:
            self.init_glm_cat_col_weights = {k.lower(): v for k, v in init_glm_cat_col_weights.items()} # lowercase all the keys in the init_glm_cat_col_weights dict:            
        if self.exposure_name  is not None:
            self.exposure_name = exposure_name.lower()
        
        # Quick Checks for the input (not all checks are done here, but at least some):
        if (self.nr_features is not None) and (self.cat_features is not None):
            if (len(self.nr_features) == 0) and (len(self.cat_features) == 0):
                raise ValueError('The input must have at least one numerical or categorical feature')
        if (self.cat_features is not None):
            if len(self.cat_features) > 0 and self.cat_vocabulary is None:
                raise ValueError('If cat_features are given, then also the cat_vocabulary must be given')
        if self.exposure_name is not None and self.output_dim != 1:
            raise ValueError('If exposure_name is not None, then output_dim must be 1')

                
                
        # Nr. Embeddings:
        if len(self.nr_features) > 0:
            self.glm_nr_embedding = GLM_Embedding_Nr_Features(self.nr_features,self.init_glm_nr_col_weights,self.trainable_glm_emb) 

        # Cat. Embeddings:
        if len(self.cat_features) > 0:
            self.cat_embedding = GLM_Embedding_Cat_Features(self.cat_features,self.cat_vocabulary,self.init_glm_cat_col_weights,self.trainable_glm_emb)

        
        # create a layer that calculates the dot product between the attention weights (ftt output) and the input for the glm:
        self.multiply_ftt_x_glm_input_layer = tf.keras.layers.Multiply(name="feature_contributions")
        
        self.scalar_product_layer = tf.keras.layers.Dense(units=1, activation='linear', name='scalar_product', 
                            weights=[np.ones((self.dim_glm, 1)), np.array([0])], trainable=False)
        
        # Note that we actually don't want to make the following weights trainable, 
        # but to get the bias to be trainable we need to do so. see comment in Book Wüthrich & Merz (2023) page 500 
        self.output_layer = tf.keras.layers.Dense(units=1, activation=self.last_activation, name='Result_LocalGLMftt_without_Exposure',
                        weights=[np.ones((1, 1)), np.array([self.init_glm_bias])], trainable=self.trainable_glm_emb) 

        
        if self.exposure_name  is not None:
            self.multiply_exposure_layer = tf.keras.layers.Multiply(name="Multiply_with_Exposure")
    
    def get_config(self):
        config = super(LocalGLM_FT_Transformer, self).get_config()
        config.update({
            'emb_dim': self.emb_dim,
            'nr_features': self.nr_features,
            'cat_features': self.cat_features,
            'cat_vocabulary': self.cat_vocabulary,
            'count_transformer_blocks': self.count_transformer_blocks,
            'attention_n_heads': self.attention_n_heads,
            'attention_dropout': self.attention_dropout,
            'ffn_d_hidden': self.ffn_d_hidden,
            'ffn_dropout': self.ffn_dropout,
            'ffn_activation_ReGLU': self.ffn_activation_ReGLU,
            'prenormalization': self.prenormalization,
            'output_dim': self.output_dim,
            'last_activation': self.last_activation,
            'last_layer_initial_weights': self.last_layer_initial_weights,
            'last_layer_initial_bias': self.last_layer_initial_bias,
            'init_glm_cat_col_weights': self.init_glm_cat_col_weights,
            'init_glm_nr_col_weights': self.init_glm_nr_col_weights,
            'init_glm_bias': self.init_glm_bias,
            'trainable_glm_emb': self.trainable_glm_emb,
            'exposure_name': self.exposure_name,
            'seed_nr': self.seed_nr,
        })
        return config
    
    def call(self, inputs):
        
        
        # Create the input for the ft-transformer:
        # -----------------
        # numerical features:
        if (self.nr_features is not None) and (len(self.nr_features) > 0):
            num_input = tf.concat([inputs[c] for c in self.nr_features], axis=1) 
            # inputs[n] is a tensor of shape (batch_size, 1)
            # num_input is a tensor of shape (batch_size, count_numerical_features)
            tokenized_features_nr = self.glm_nr_embedding(num_input) # tokenized_features_nr is a tensor of shape (batch_size, count_numerical_features, emb_dim)
        
        # categorical features:
        if (self.cat_features is not None) and (len(self.cat_features) > 0):
            cat_input = tf.concat([inputs[c] for c in self.cat_features], axis=1) 
            # inputs[c] is a tensor of shape (batch_size, 1)
            # cat_input is a tensor of shape (batch_size, count_categorical_features)
            tokenized_features_cat = self.cat_embedding(cat_input) # tokenized_features_cat is a tensor of shape (batch_size, count_categorical_features, emb_dim)
        
        # stack all the inputs features together (actually concat them along the token axis, since stack would create a new axis):
        if (self.nr_features is not None) and (self.cat_features is not None):
            if (len(self.nr_features) != 0) and (len(self.cat_features) != 0):
                tokenized_features = tf.concat([tokenized_features_nr,tokenized_features_cat], axis=1) #tokenized_features is a tensor of shape (batch_size, count_features, emb_dim)
        if (self.cat_features is None) or (len(self.cat_features) == 0):
            tokenized_features = tokenized_features_nr #tokenized_features is a tensor of shape (batch_size, count_features, emb_dim)
        if (self.nr_features is None) or (len(self.nr_features) == 0):
            tokenized_features = tokenized_features_cat #tokenized_features is a tensor of shape (batch_size, count_features, emb_dim)            
        

        
        # this represents a tensor of shape (batch_size, count_features), were every entry for each row is glm_beta times input x.
        local_glms_x = tokenized_features
       
        # Create the output of the transformer:
        output_FT_transformer = self.FT_transformer(inputs)['output']
        
        
        # note that the weights are set to 0 and the bias is set to the initial glm betas
        # create a layer that calculates the dot product between the attention weights (Attention) and the input matrix Input_Matrix_OHE:
        # (Attention has the same dimension as the input matrix Input_Matrix_OHE):
        weighted_input = self.multiply_ftt_x_glm_input_layer([output_FT_transformer, local_glms_x])
        scalar_product = self.scalar_product_layer(weighted_input)
        output = self.output_layer(scalar_product)
        
        if self.exposure_name is not None:
            output = self.multiply_exposure_layer([output, inputs[self.exposure_name]])
            return {"output": output}
        else:
            return {"output": output} 

            
    
    
        
        
    