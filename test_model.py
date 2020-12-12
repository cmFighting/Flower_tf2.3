import tensorflow as tf


# 数据加载，按照8:2的比例加载花卉数据
def data_load(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    return train_ds, val_ds, class_names


def test(is_transfer=True):
    train_ds, val_ds, class_names = data_load("../data/flower_photos", 224, 224, 4)
    if is_transfer:
        model = tf.keras.models.load_model("models/mobilenet_flower.h5")
    else:
        model = tf.keras.models.load_model("models/cnn_flower.h5")
    model.summary()
    loss, accuracy = model.evaluate(val_ds)
    print('Test accuracy :', accuracy)


if __name__ == '__main__':
    test(True)

