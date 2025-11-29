import tensorflow as tf
import six
from math_tool import \
    get_activation, \
    create_initializer, \
    get_shape_list, reshape_to_matrix, \
    dropout, layer_norm, layer_norm_and_dropout, \
    create_attention_mask_from_input_mask
from my_transformer import transformer_model
import json
import copy

class BertConfig(object):
    """`BertModel` 的配置类。定义 / 存储 BERT 模型的所有核心超参数和结构参数"""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """构造 BertConfig。

        参数:
          vocab_size: `BertModel` 中 `input_ids` 的词汇表大小。
          hidden_size: 编码器层和池化层的隐藏层大小。
          num_hidden_layers: Transformer 编码器中的隐藏层数量。
          num_attention_heads: 每个注意力层的注意力头数量。
          intermediate_size: Transformer 编码器中“中间层”（即前馈层）的大小。
          hidden_act: 编码器和池化层中非线性激活函数（函数或字符串）。
          hidden_dropout_prob: 嵌入层、编码器和池化层中所有全连接层的 dropout 概率。
          attention_probs_dropout_prob: 注意力概率的 dropout 比例。
          max_position_embeddings: 模型可能使用的最大序列长度。通常设置较大值（例如 512、1024 或 2048）
          type_vocab_size: `BertModel` 中传入的 `token_type_ids` 的词汇表大小。
          initializer_range: 初始化所有权重矩阵时截断正态分布的标准差。
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """从 Python 字典构建 `BertConfig`。"""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """从 JSON 文件构建 `BertConfig`。"""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """将此实例序列化为 Python 字典。"""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """将此实例序列化为 JSON 字符串。"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def tf_get_variavle(name, shape, initializer):
   """TF1 -> TF2"""
   embedding_table = tf.Variable(
    initial_value=initializer(shape=shape, dtype=tf.float32),
    name=name,
    trainable=True)
   return embedding_table

def tf_variable_scope(scope, default_name="bert"):
   return tf.name_scope(scope or default_name)

def tf_layers_dense(tensor, units, activation,kernel_initializer,name="test"):
   return tf.keras.layers.Dense(
      units=units,
      activation=activation,
      kernel_initializer=kernel_initializer,
      name="test",
   )(tensor)


class BertModel(object):
    """BERT 模型（来自 Transformer 的双向编码器表示）

    示例用法:

    ```python
    # 已经转换为 WordPiece 令牌 IDs
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
      num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
      input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf_get_variavle(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=False,
                 scope=None):
        """BertModel 的构造方法

        参数:
          config: `BertConfig` 实例。
          is_training: bool. 训练模式为 True，评估模式为 False。控制 dropout 是否生效。
          input_ids: int32 张量，形状为 [batch_size, seq_length]。
          input_mask: （可选）int32 张量，形状为 [batch_size, seq_length]。
          token_type_ids: （可选）int32 张量，形状为 [batch_size, seq_length]。
          use_one_hot_embeddings: （可选）bool. 
            True 使用独热向量表示词嵌入，否则使用 tf.embedding_lookup()。
          scope: （可选）变量作用域，默认 "bert"。

        异常:
          ValueError: 如果配置不合法或者输入张量形状不匹配。
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf_variable_scope(scope, default_name="bert"):
            with tf_variable_scope("embeddings"):
                # 通过词 ID 查找词嵌入
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings)

                # 加上位置嵌入和词类别嵌入，然后进行层归一化和 dropout
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            with tf_variable_scope("encoder"):
                # 将 2D mask [batch_size, seq_length] 转换为 
                # 3D mask [batch_size, seq_length, seq_length]
                # 用于计算注意力分数
                attention_mask = create_attention_mask_from_input_mask(
                    input_ids, input_mask)

                # 运行堆叠的 Transformer
                # `sequence_output` 形状 = [batch_size, seq_length, hidden_size]
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            self.sequence_output = self.all_encoder_layers[-1]

            # "池化器" 将编码后的序列张量 [batch_size, seq_length, hidden_size]
            # 转换为固定大小的张量 [batch_size, hidden_size]，用于分类任务
            with tf_variable_scope("pooler"):
                # 取第一个 token 的隐藏状态作为池化输出
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf_layers_dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(config.initializer_range))

    def get_pooled_output(self):
        """获取池化输出（固定维度表示）"""
        return self.pooled_output

    def get_sequence_output(self):
        """获取编码器的最终隐藏层输出

        返回:
          float 张量，形状为 [batch_size, seq_length, hidden_size],
          对应 Transformer 编码器的最终隐藏层。
        """
        return self.sequence_output

    def get_all_encoder_layers(self):
        """获取编码器的所有层输出"""
        return self.all_encoder_layers

    def get_embedding_output(self):
        """获取词嵌入层输出(Transformer 的输入)

        返回:
          float 张量，形状为 [batch_size, seq_length, hidden_size],
          对应词嵌入层输出（包括词嵌入、位置嵌入和 token type 嵌入叠加后，进行层归一化）。
        """
        return self.embedding_output

    def get_embedding_table(self):
        """获取词嵌入表"""
        return self.embedding_table

def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  """查找id张量对应的词嵌入表示

  Args:
    input_ids: int32 张良,形状 [batch_size, seq_length] 包含词ids.
    vocab_size: int. 嵌入词表的大小.
    embedding_size: int. 嵌入向量的维度.
    initializer_range: float. 嵌入矩阵初始化的范围.
    word_embedding_name: string. 嵌入表表名.
    use_one_hot_embeddings: bool. 如果设置为 True , 使用独热编码
      表示词， 否则使用 `tf.gather()` 查找词.

  Returns:
    float 张量,形如 [batch_size, seq_length, embedding_size*num_inputs].
  """
  # 方法假定输入形如[batch_size, seq_length, num_inputs].
  # NOTE: `num_inputs` 考虑到例如(分词+词性+词频)这种多输入的情况
  # 如果输入是一个二维张量,形如 [batch_size, seq_length], 我们
  # 重塑为 [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  embedding_table = tf_get_variavle( # [vocab_size, embedding_size]
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  flat_input_ids = tf.reshape(input_ids, [-1]) # => 一维
  if use_one_hot_embeddings: # N:总词数 = batch_size*seq_length*num_inputs
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size) # => [N, vocab_size]
    output = tf.matmul(one_hot_input_ids, embedding_table) # => [N, embedding_size]
  else:
    output = tf.gather(embedding_table, flat_input_ids)

  input_shape = get_shape_list(input_ids)

  output = tf.reshape(output, # 最后一维拼接num_inputs个embedding
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """对词向量张量进行多种后处理操作。

  参数:
    input_tensor: float 型张量，形状为 [batch_size, seq_length, embedding_size]。
                  —— 就是前面 embedding_lookup 的输出。
    use_token_type: bool。是否添加 segment/token type embedding。
                    —— 在 BERT 里用于区分句子 A 和句子 B。
    token_type_ids: (可选) int32 型张量，形状 [batch_size, seq_length]。
                    —— 每个位置是 token 属于哪个 segment(0 或 1)。
                       必须在 use_token_type=True 时提供。
    token_type_vocab_size: int。token type 的种类数，默认 16。
    token_type_embedding_name: string。token type embedding 的变量名。
    use_position_embeddings: bool。是否添加位置 embedding。
    position_embedding_name: string。位置 embedding 的变量名。
    initializer_range: float。embedding 表的初始化范围。
    max_position_embeddings: int。模型支持的最大序列长度。
    dropout_prob: float。对最终输出应用 dropout 的概率。

  返回:
    float 型张量，形状与 input_tensor 相同。

  异常:
    ValueError: 如果输入的张量形状或参数非法。
  """

  # 获取输入张量的形状，并检查阶数是否为 3（[batch_size, seq_length, width]）
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]   # 批量大小
  seq_length = input_shape[1]   # 序列长度
  width = input_shape[2]        # embedding 维度

  output = input_tensor  # 初始化输出张量

  if use_token_type:
    if token_type_ids is None:
        raise ValueError("`token_type_ids` 必须在 `use_token_type=True` 时指定")
    
    # 创建 token type embedding 表
    token_type_table = tf_get_variavle(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))
    
    # 将 token_type_ids 扁平化并转为 one-hot
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    
    # 查表得到 token type embedding
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    # reshape 回 [batch_size, seq_length, width]
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    
    # 与原词向量相加
    output += token_type_embeddings

  if use_position_embeddings:
    # assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    assert_op = tf.debugging.assert_less_equal(seq_length, max_position_embeddings)  # HL: 适配TF2
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = tf_get_variavle(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=create_initializer(initializer_range))
      # 由于位置 embedding 表是一个可学习的变量，
      # 我们用较长的序列长度 `max_position_embeddings` 来创建它。
      # 实际输入序列长度可能比这个短，这样可以在处理短序列的任务时训练更快。

      # 因此，`full_position_embeddings` 实际上是一个位置 
      # [0, 1, 2, ..., max_position_embeddings-1] 的 embedding 表，而当前序列只有 
      # [0, 1, 2, ..., seq_length-1] 的位置，
      # 因此我们可以直接做 slice 取前 seq_length 个 embedding。

      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # 只有最后两个维度是关键的（`seq_length` 和 `width`），
      # 所以我们对前面的维度进行广播，这通常只是 batch size。
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  return output

