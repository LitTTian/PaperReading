import tensorflow as tf
import math
from math_tool import \
    get_activation, \
    get_shape_list, reshape_to_matrix, reshape_from_matrix, \
    dropout, layer_norm, \
    create_initializer


def tf_variable_scope(scope, default_name="bert"):
   return tf.name_scope(scope or default_name)

def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    """从 `from_tensor` 到 `to_tensor` 的多头注意力层。

    这是基于 "Attention is All You Need" 的多头注意力实现。
    如果 `from_tensor` 和 `to_tensor` 相同，则为自注意力（self-attention）。
    每个 from_tensor 的时间步都会关注 to_tensor 中对应的序列，并返回一个固定长度向量。

    该函数会先将 `from_tensor` 投影为 query 张量，
    `to_tensor` 投影为 key 和 value 张量。
    这些张量的形状为 [batch_size, seq_length, size_per_head * num_attention_heads]。

    接着计算 query 与 key 的点积并缩放，softmax 得到注意力概率，
    再用这些概率对 value 张量进行加权平均，最后拼接回单个张量返回。

    实际实现中，多头注意力是通过 reshape 和 transpose 操作完成的，而非真正分离的张量列表。

    参数:
      from_tensor: float 张量，形状为 [batch_size, from_seq_length, from_width]。
      to_tensor: float 张量，形状为 [batch_size, to_seq_length, to_width]。
      attention_mask: (可选) int32 张量，形状 [batch_size, from_seq_length, to_seq_length]。
                      值为 1 或 0。为 0 的位置会被 mask 掉（softmax 前加 -10000）。
      num_attention_heads: int，注意力头数。
      size_per_head: int，每个注意力头的大小。
      query_act: (可选) query 投影激活函数。
      key_act: (可选) key 投影激活函数。
      value_act: (可选) value 投影激活函数。
      attention_probs_dropout_prob: (可选) float，注意力概率的 dropout。
      initializer_range: float，权重初始化范围。
      do_return_2d_tensor: bool，如果 True，输出形状为 [B*F, N*H]；否则为 [B, F, N*H]。
      batch_size: (可选) int，如果输入为 2D 张量，需要提供 batch_size。
      from_seq_length: (可选) int，如果输入为 2D 张量，需要提供 from_seq_length。
      to_seq_length: (可选) int，如果输入为 2D 张量，需要提供 to_seq_length。

    返回:
      float 张量，形状为 [B, F, N*H]，或者 [B*F, N*H]（取决于 do_return_2d_tensor）。

    异常:
      ValueError: 参数或张量形状不合法。
    """

    # 内部函数：调整张量形状以适应多头注意力计算
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])
        # 转置成 [B, N, F/T, H]
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    # 获取输入张量的形状
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "`from_tensor` 和 `to_tensor` 的维度必须相同。")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "当输入为 2D 张量时，必须提供 batch_size, from_seq_length 和 to_seq_length。")

    # 将张量 reshape 为 2D，方便线性变换
    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # 构建 query, key, value
    # query_layer = tf.layers.dense(
    query_layer = tf.keras.layers.Dense(
        # from_tensor_2d,
        units=num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))(from_tensor_2d)

    # key_layer = tf.layers.dense(
    key_layer = tf.keras.layers.Dense(
        # to_tensor_2d,
        units = num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))(to_tensor_2d)

    # value_layer = tf.layers.dense(
    value_layer = tf.keras.layers.Dense(
        # to_tensor_2d,
        units = num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))(to_tensor_2d)

    # reshape 并转置成 [B, N, F/T, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # 计算注意力分数
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    # 应用 attention mask
    if attention_mask is not None:
        attention_mask = tf.expand_dims(attention_mask, axis=[1])
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
        attention_scores += adder

    # softmax 得到注意力概率
    attention_probs = tf.nn.softmax(attention_scores)
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # reshape value 并转置
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # 加权求和得到 context
    context_layer = tf.matmul(attention_probs, value_layer)
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    # reshape 回 2D 或 3D
    if do_return_2d_tensor:
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=get_activation("gelu"),
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    """多头、多层 Transformer 模型（来源于 "Attention is All You Need"）。

    这几乎是原始 Transformer 编码器的完整实现。

    参考论文：
    https://arxiv.org/abs/1706.03762

    参考实现：
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    参数:
      input_tensor: float 张量，形状为 [batch_size, seq_length, hidden_size]。
      attention_mask: （可选）int32 张量，形状为 [batch_size, seq_length, seq_length]，
        1 表示可注意的位置，0 表示不可注意的位置。
      hidden_size: int，Transformer 的隐藏层维度。
      num_hidden_layers: int，Transformer 的层数（块数）。
      num_attention_heads: int，Transformer 的注意力头数。
      intermediate_size: int，中间层（前馈层）的大小。
      intermediate_act_fn: 函数，中间层的非线性激活函数。
      hidden_dropout_prob: float，隐藏层的 dropout 概率。
      attention_probs_dropout_prob: float，注意力概率的 dropout 概率。
      initializer_range: float，权重初始化范围（截断正态分布标准差）。
      do_return_all_layers: bool，是否返回所有层输出，默认只返回最终层。

    返回:
      float 张量，形状为 [batch_size, seq_length, hidden_size]，Transformer 的最终隐藏层。

    异常:
      ValueError: 如果张量形状或参数不合法。
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "隐藏层维度 (%d) 不是注意力头数 (%d) 的整数倍" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # Transformer 对所有层使用残差连接，因此输入维度必须与 hidden_size 一致
    if input_width != hidden_size:
        raise ValueError("输入张量宽度 (%d) != hidden_size (%d)" % (input_width, hidden_size))

    # 将输入张量保持为 2D，避免在 2D 与 3D 间频繁 reshape
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf_variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf_variable_scope("attention"):
                attention_heads = []
                with tf_variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # 如果有其他序列，则先在 self-attention 头上 concat，再投影
                    attention_output = tf.concat(attention_heads, axis=-1)

                # 线性投影到 hidden_size，并添加残差
                with tf_variable_scope("output"):
                    # attention_output = tf.layers.dense(
                    attention_output = tf.keras.layers.Dense(
                        # attention_output,
                        units = hidden_size,
                        kernel_initializer=create_initializer(initializer_range))(attention_output)
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)

            # 仅对中间层应用激活函数
            with tf_variable_scope("intermediate"):
                # intermediate_output = tf.layers.dense(
                intermediate_output = tf.keras.layers.Dense(
                    # attention_output,
                    units = intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range))(attention_output)

            # 下投影回 hidden_size，并添加残差
            with tf_variable_scope("output"):
                # layer_output = tf.layers.dense(
                layer_output = tf.keras.layers.Dense(
                    # intermediate_output,
                    units = hidden_size,
                    kernel_initializer=create_initializer(initializer_range))(intermediate_output)
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output
