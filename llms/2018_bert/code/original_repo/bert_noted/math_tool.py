import six
import numpy as np
import tensorflow as tf

# MODULE: tensor tools
def assert_rank(tensor, expected_rank, name=None):
  """如果张量的阶(rank)不符合预期，则抛出异常。

  参数:
    tensor: 需要检查阶数的 tf.Tensor。
    expected_rank: Python 整数或整数列表，表示期望的阶数。
    name: （可选）张量的名字，用于报错信息。

  异常:
    ValueError: 如果张量的实际阶数与期望阶数不一致，则抛出该错误。
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    # scope_name = tf.get_variable_scope().name
    scope_name = tf.name_scope("bert")
    raise ValueError(
        "在作用域 `%s` 中，张量 `%s` 的实际阶数 `%d` (形状 = %s) "
        "与期望的阶数 `%s` 不一致。" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def get_shape_list(tensor, expected_rank=None, name=None):
  """返回张量的形状列表，优先使用静态维度。

  参数:
    tensor: 一个 tf.Tensor, 需要获取其形状。
    expected_rank: (可选) int。期望的张量阶(rank)。如果指定了并且
      张量的实际阶与期望不符，就会抛出异常。
    name: （可选）张量的名字，用于报错信息。

  返回:
    一个张量形状的列表。所有静态维度会作为 Python 整数返回，
    动态维度会作为 tf.Tensor 标量返回。
  """
  if name is None:
    # name = tensor.name
    name = "tensor1"

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)  # 检查张量阶是否符合预期

  shape = tensor.shape.as_list()  # 尝试获取静态维度（可能有 None）

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:  # 如果某个维度在图构建时不能确定（动态）
      non_static_indexes.append(index)

  if not non_static_indexes:  # 如果全是静态维度
    return shape

  dyn_shape = tf.shape(tensor)  # 动态获取真实形状
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]  # 用动态维度填补 None

  return shape

def reshape_to_matrix(input_tensor):
  """将张量（rank ≥ 2）重塑为二维张量（矩阵）。"""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("输入张量至少是二维。Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor

def reshape_from_matrix(output_tensor, orig_shape_list):
    """将二维张量（矩阵）重塑回原始的 rank ≥ 2 张量。"""
    # 如果原始张量本身就是二维，直接返回
    if len(orig_shape_list) == 2:
        return output_tensor
    output_shape = get_shape_list(output_tensor) # 获取输出张量的形状
    orig_dims = orig_shape_list[0:-1] # 原始张量除了最后一维的形状
    width = output_shape[-1] # 最后一维大小（通常是 hidden_size）
    return tf.reshape(output_tensor, orig_dims + [width]) # 重塑回原始形状


# MODULE: layers and initializers
def create_initializer(initializer_range=0.02):
  """创建一个标准差为 `initializer_range` 的截断正态分布初始化器。"""
  # return tf.truncated_normal_initializer(stddev=initializer_range)
  def initializer(shape, dtype=tf.float32):  # HL: 适配TF2
        # TF2 原生接口：生成截断正态分布张量（均值0，标准差initializer_range，截断在±2σ）
        return tf.random.truncated_normal(
            shape=shape,
            mean=0.0,
            stddev=initializer_range,
            dtype=dtype
        )
  return initializer

def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """对输入张量先进行 LayerNorm 再进行 Dropout"""
  output_tensor = layer_norm(input_tensor, name)  # 先做层归一化
  output_tensor = dropout(output_tensor, dropout_prob)  # 再做 dropout
  return output_tensor


def dropout(input_tensor, dropout_prob):
  """对张量执行 dropout 操作

  参数:
    input_tensor: float 型张量
    dropout_prob: float, 丢弃概率 (注意不是保留概率)

  返回:
    对 input_tensor 应用 dropout 后的张量
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor  # 如果丢弃率为 0，则直接返回原张量

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output

def layer_norm(input_tensor, name=None):
  """对张量最后一个维度执行 LayerNorm"""
  # return tf.contrib.layers.layer_norm(
  #     inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
  """对张量最后一个维度执行 LayerNorm（适配 TF2 原生接口）"""
  # HL: 适配TF2
  # 创建 LayerNormalization 层，指定归一化轴和参数轴（与原逻辑一致）
  layer = tf.keras.layers.LayerNormalization(
      axis=-1,  # 对最后一个维度归一化（begin_norm_axis=-1）
      center=True,  # 原接口默认包含偏置项（beta）
      scale=True,   # 原接口默认包含缩放项（gamma）
      name=name
  )
  # 执行归一化（Keras 层直接调用即可，自动处理参数初始化）
  return layer(input_tensor)


def gelu(x):
    """高斯误差线性单元(Gaussian Error Linear Unit, GELU)

    这是 ReLU 的一种平滑版本。
    原始论文: https://arxiv.org/abs/1606.08415

    参数:
      x: 需要进行激活的 float 张量。

    返回:
      对 x 应用 GELU 激活后的张量。
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def get_activation(activation_string):
    """将字符串映射为对应的 Python 激活函数，例如 "relu" => `tf.nn.relu`。

    参数:
      activation_string: 激活函数的名称字符串。

    返回:
      对应的 Python 激活函数。如果 `activation_string` 为 None、空字符串或 "linear"，则返回 None。
      如果 `activation_string` 不是字符串，则直接返回 `activation_string`（假设它已经是一个函数）。

    异常:
      ValueError: 如果 `activation_string` 不是已知的激活函数名称。
    """

    # 假设任何非字符串对象已经是激活函数，直接返回
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("不支持的激活函数: %s" % act)

def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """根据二维张量 mask 创建三维注意力 mask。

    参数:
      from_tensor: 2D 或 3D 张量，形状为 [batch_size, from_seq_length, ...]。
      to_mask: int32 张量，形状为 [batch_size, to_seq_length]。

    返回:
      float 张量，形状为 [batch_size, from_seq_length, to_seq_length]。
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    # 将 to_mask 变形为 [batch_size, 1, to_seq_length] 并转为 float
    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # 我们不假设 from_tensor 本身是 mask（虽然它也可能是）
    # 我们只关心是否 attend 到 padding token（to 方向的 padding）
    # 因此创建一个全 1 张量作为基础
    #
    # `broadcast_ones` 形状 = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # 广播乘法，得到最终 mask
    mask = broadcast_ones * to_mask

    return mask
