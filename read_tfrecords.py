import tensorflow as tf


def _extract_features(example):
    features = {
        "image": tf.FixedLenFeature((), tf.string),
        "mask": tf.FixedLenFeature((), tf.string)
    }
    parsed_example = tf.parse_single_example(example, features)
    images = tf.cast(tf.image.decode_jpeg(parsed_example["image"]), dtype=tf.float32)
    images.set_shape([800, 600, 3])
    masks = tf.cast(tf.image.decode_jpeg(parsed_example["mask"]), dtype=tf.float32) / 255.
    masks.set_shape([800, 600, 1])
    return images, masks


if __name__ == '__main__':
    filenames = ["train-00001-of-00001"]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_extract_features)
    dataset = dataset.batch(8)
    dataset = dataset.shuffle(buffer_size=50)
    dataset = dataset.repeat(1)

    iterator = dataset.make_one_shot_iterator()

    it = 0
    with tf.Session() as sess:
        next_images, next_masks = iterator.get_next()
        try:
            while True:
                images, masks = sess.run([next_images, next_masks])
                if it == 0:
                    print("Sample shape are {img}, {mask}".format(img=images.shape, mask=masks.shape))
                it += 1
        except tf.errors.OutOfRangeError:
            print("End of batches")
        finally:
            print("There are {bs} number of batches".format(bs=it))

