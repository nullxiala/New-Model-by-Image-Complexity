import os
import time
from absl import app, flags, logging
from tensorflow import keras
import cv2
from data.imagedata import transform_images, load_tfrecord_dataset
from data.kits import draw_outputs
from data.get_percent import density, overlap_area
#---------------
from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from data.kits import broadcast_iou

flags.DEFINE_integer('karl_max_boxes', 100,
                     'maximum number of boxes per image')
flags.DEFINE_float('karl_iou_threshold', 0.5, 'iou threshold')
flags.DEFINE_float('karl_score_threshold', 0.1, 'score threshold')

karl_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
karl_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

karl_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
karl_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def DarknetTiny(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x, 256, 3)  # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)


def karlConv(filters, name=None):
    def karl_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return karl_conv


def karlConvTiny(filters, name=None):
    def karl_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)
    return karl_conv


def karlOutput(filters, anchors, classes, name=None):
    def karl_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return karl_output


def karl_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def karl_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=FLAGS.karl_max_boxes,
        max_total_size=FLAGS.karl_max_boxes,
        iou_threshold=FLAGS.karl_iou_threshold,
        score_threshold=FLAGS.karl_score_threshold
    )

    return boxes, scores, classes, valid_detections


def Set_karl(size=None, channels=3, anchors=karl_anchors,
           masks=karl_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_36, x_61, x = Darknet(name='karl_darknet')(x)

    x = karlConv(512, name='karl_conv_0')(x)
    output_0 = karlOutput(512, len(masks[0]), classes, name='karl_output_0')(x)

    x = karlConv(256, name='karl_conv_1')((x, x_61))
    output_1 = karlOutput(256, len(masks[1]), classes, name='karl_output_1')(x)

    x = karlConv(128, name='karl_conv_2')((x, x_36))
    output_2 = karlOutput(128, len(masks[2]), classes, name='karl_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='karl')

    boxes_0 = Lambda(lambda x: karl_boxes(x, anchors[masks[0]], classes),
                     name='karl_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: karl_boxes(x, anchors[masks[1]], classes),
                     name='karl_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: karl_boxes(x, anchors[masks[2]], classes),
                     name='karl_boxes_2')(output_2)

    outputs = Lambda(lambda x: karl_nms(x, anchors, masks, classes),
                     name='karl_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='karl')


def Set_karlTiny(size=None, channels=3, anchors=karl_tiny_anchors,
               masks=karl_tiny_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_8, x = DarknetTiny(name='karl_darknet')(x)

    x = karlConvTiny(256, name='karl_conv_0')(x)
    output_0 = karlOutput(256, len(masks[0]), classes, name='karl_output_0')(x)

    x = karlConvTiny(128, name='karl_conv_1')((x, x_8))
    output_1 = karlOutput(128, len(masks[1]), classes, name='karl_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='karl')

    boxes_0 = Lambda(lambda x: karl_boxes(x, anchors[masks[0]], classes),
                     name='karl_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: karl_boxes(x, anchors[masks[1]], classes),
                     name='karl_boxes_1')(output_1)
    outputs = Lambda(lambda x: karl_nms(x, anchors, masks, classes),
                     name='karl_nms')((boxes_0[:3], boxes_1[:3]))
    return Model(inputs, outputs, name='karl_tiny')


def karlLoss(anchors, classes=80, ignore_thresh=0.5):
    def karl_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = karl_boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return karl_loss

#---------------

flags.DEFINE_string('classes', './data/karl_class.names', '可识别种类的路径')
flags.DEFINE_string('weights', './checkpoints/karl.tf',
                    '权重的路径')
flags.DEFINE_boolean('tiny', False, 'karl or karl-tiny 是否使用轻量化模型')
flags.DEFINE_integer('size', 416, '设置图片的尺寸默认 416')
flags.DEFINE_string('image', 'D:\codePrograms\karl-tf2\imagesm', '输入文件的路径')

flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', '输出图片的路径，默认 ./output.jpg')
flags.DEFINE_integer('num_classes', 80, '种类的数量，默认 80')


def m_predict(x1, x2):
    # 将输入数据转换为numpy数组，并增加一个维度
    x = np.array([x1, x2])
    x = np.expand_dims(x, axis=0)

    # 加载模型
    model = keras.models.load_model("mdata.h5")

    # 用模型进行预测，并返回结果
    y = model.predict(x)
    y = y[0][0]
    return y

#定义主函数 如果是直接运行脚本那么使用
def main(_argv):
    '''
    karl = Set_karlTiny(classes=FLAGS.num_classes
    # 加载预训练权重
    print(f'Loading weights from: {FLAGS.weights}')
    karl.load_weights(FLAGS.weights).expect_partial()
    print('权重加载完成')

    # 加载类别名称 注意utf-8打开
    class_names = [c.strip() for c in open(FLAGS.classes, encoding='utf-8').readlines()]
    print('类型加载完成')
    '''
    input_path  = FLAGS.image
    #可以在此处手动更改路径 上面
    array = []
    if os.path.isdir(input_path):
        # 如果输入的是一个目录
        image_found = False
        output_dir = os.path.join(input_path, 'karl_out')#此处修改储存目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for filename in os.listdir(input_path):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.JPG') or filename.endswith('.PNG'):
                #第一次加载模型
                FLAGS.karl_score_threshold = 0.1
                karl = Set_karl(classes=FLAGS.num_classes)
                # 加载预训练权重
                print(f'Loading weights from: {FLAGS.weights}')
                karl.load_weights(FLAGS.weights).expect_partial()
                print('第一次权重加载完成')

                # 加载类别名称 注意utf-8打开
                class_names = [c.strip() for c in open(FLAGS.classes, encoding='utf-8').readlines()]
                print('第一次类型加载完成')
                # 处理每个图像文件
                image_found = True
                image_path = os.path.join(input_path, filename)
                FLAGS.image = image_path
                #FLAGS.output = os.path.join(output_dir,filename)
                # 如果指定了 tfrecord 文件，则从中加载图像数据
                if FLAGS.tfrecord:
                    dataset = load_tfrecord_dataset(
                        FLAGS.tfrecord, FLAGS.classes, FLAGS.size)

                    dataset = dataset.shuffle(512)
                    img_raw, _label = next(iter(dataset.take(1)))
                else:
                    # 否则，从指定的图像文件中加载图像数据
                    img_raw = tf.image.decode_image(
                        open(FLAGS.image, 'rb').read(), channels=3)

                # 调整图像大小并添加批次维度
                img = tf.expand_dims(img_raw, 0)
                img = transform_images(img, FLAGS.size)

                # 运行模型并记录运行时间
                t1 = time.time()
                boxes, scores, classes, nums = karl(img)
                t2 = time.time()

                den = density(boxes, nums)
                area= overlap_area(boxes, nums)
                FLAGS.karl_score_threshold = m_predict(den ,area)

                karl = Set_karl(classes=FLAGS.num_classes)
                karl.load_weights(FLAGS.weights).expect_partial()
                print(FLAGS.karl_score_threshold)
                print('权重加载完成')

                # 加载类别名称 注意utf-8打开
                class_names = [c.strip() for c in open(FLAGS.classes, encoding='utf-8').readlines()]
                print('类型加载完成')
                boxes, scores, classes, nums = karl(img)
                print('\033[31m运行time: {}\033[0m'.format(t2 - t1))

                # 输出检测结果
                print('Karl_AI:检测结果如下:')
                for i in range(nums[0]):
                    print('\033[34m\t{}, {}, {}\033[0m'.format(class_names[int(classes[0][i])],
                                                               np.array(scores[0][i]),
                                                               np.array(boxes[0][i])))

                # 将图像从 RGB 转换为 BGR 格式
                img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
                # 在图像上绘制检测结果
                img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

                # 保存输出图像
                FLAGS.output = os.path.join(output_dir, '%.2f' % den +'_'+'%.2f' % area+'_'+filename)
                cv2.imwrite(FLAGS.output, img)
                print('Karl_AI:Output to {}'.format(FLAGS.output))
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass