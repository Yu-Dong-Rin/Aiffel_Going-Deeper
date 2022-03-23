# Going Deeper 1. ResNet Ablation Study
# 1. 라이브러리 import


```python
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
```

# 2. 데이터 로드 및 전처리
## 2-1. 데이터 로드


```python
tf.config.list_physical_devices('GPU')
```




    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]




```python
import urllib3
urllib3.disable_warnings()
(ds_train, ds_test), ds_info = tfds.load(
    'stanford_dogs',
    split=['train', 'test'],
    as_supervised=True,
    shuffle_files=True,
    with_info=True,
)
```


```python
fig = tfds.show_examples(ds_info, ds_train)
```

    WARNING: For consistency with `tfds.load`, the `tfds.show_examples` signature has been modified from (info, ds) to (ds, info).
    The old signature is deprecated and will be removed. Please change your call to `tfds.show_examples(ds, info)`



    
![png](output_5_1.png)
    


## 2-2. 데이터 전처리


```python
def normalize_and_resize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.resize(image, [224, 224])
    return tf.cast(image, tf.float32) / 255., label
```


```python
# 데이터셋(ds)을 가공하는 메인함수
def apply_normalize_on_dataset(ds, is_test=False, batch_size=16, with_aug=False):
    ds = ds.map(
        normalize_and_resize_img,  # 기본적인 전처리 함수 적용
        num_parallel_calls=2
    )
    if not is_test and with_aug:
        ds = ds.map(
            augment,       # augment 함수 적용
            num_parallel_calls=2
        )
    ds = ds.batch(batch_size)
    if not is_test:
        ds = ds.repeat()
        ds = ds.shuffle(200)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
```


```python
ds_train_no_aug = apply_normalize_on_dataset(ds_train, with_aug=False)
```

## 2-3. 이미지 섞기
-> 사진데이터를 잘라서 섞고 라벨에 대한 값 또한 전체사진에 차지하는 비율만큼 라벨을 새로 만들어준다. 


```python
# 데이터셋에서 이미지 2개를 가져옵니다. 
for i, (image, label) in enumerate(ds_train_no_aug.take(1)):
    if i == 0:
        image_a = image[0]
        image_b = image[1]
        label_a = label[0]
        label_b = label[1]
        break

plt.subplot(1,2,1)
plt.imshow(image_a)

plt.subplot(1,2,2)
plt.imshow(image_b)
```




    <matplotlib.image.AxesImage at 0x7f168076a8b0>




    
![png](output_11_1.png)
    



```python
def get_clip_box(image_a, image_b, img_size=224):
    # get center of box
    x = tf.cast( tf.random.uniform([],0, img_size),tf.int32)
    y = tf.cast( tf.random.uniform([],0, img_size),tf.int32)

    # get width of box
    _prob = tf.random.uniform([],0,1)
    width = tf.cast(img_size * tf.math.sqrt(1-_prob),tf.int32)
    
    # clip box in image and get minmax bbox
    xa = tf.math.maximum(0, x-width//2)
    ya = tf.math.maximum(0, y-width//2)
    yb = tf.math.minimum(img_size, y+width//2)
    xb = tf.math.minimum(img_size, x+width//2)
    
    return xa, ya, xb, yb
```


```python
xa, ya, xb, yb = get_clip_box(image_a, image_b)
print(xa, ya, xb, yb)
```

    tf.Tensor(77, shape=(), dtype=int32) tf.Tensor(154, shape=(), dtype=int32) tf.Tensor(155, shape=(), dtype=int32) tf.Tensor(224, shape=(), dtype=int32)



```python
# mix two images
def mix_2_images(image_a, image_b, xa, ya, xb, yb, img_size=224):
    one = image_a[ya:yb,0:xa,:]
    two = image_b[ya:yb,xa:xb,:]
    three = image_a[ya:yb,xb:img_size,:]
    middle = tf.concat([one,two,three],axis=1)
    top = image_a[0:ya,:,:]
    bottom = image_a[yb:img_size,:,:]
    mixed_img = tf.concat([top, middle, bottom],axis=0)
    
    return mixed_img
```


```python
mixed_img = mix_2_images(image_a, image_b, xa, ya, xb, yb)
plt.imshow(mixed_img.numpy())
```




    <matplotlib.image.AxesImage at 0x7f1680668700>




    
![png](output_15_1.png)
    


## 2-4. 라벨 섞기


```python
# mix two labels
def mix_2_label(label_a, label_b, xa, ya, xb, yb, img_size=224, num_classes=120):
    mixed_area = (xb-xa)*(yb-ya)
    total_area = img_size*img_size
    a = tf.cast(mixed_area/total_area, tf.float32)

    if len(label_a.shape)==0:
        label_a = tf.one_hot(label_a, num_classes)
    if len(label_b.shape)==0:
        label_b = tf.one_hot(label_b, num_classes)
    mixed_label = (1-a)*label_a + a*label_b
    return mixed_label
```


```python
mixed_label = mix_2_label(label_a, label_b, xa, ya, xb, yb)
mixed_label
```




    <tf.Tensor: shape=(120,), dtype=float32, numpy=
    array([0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.10881697, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.891183  , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ],
          dtype=float32)>



## 2-5. cutmix( ) 함수 구현


```python
def cutmix(image, label, prob = 1.0, batch_size=16, img_size=224, num_classes=120):
    mixed_imgs = []
    mixed_labels = []

    for i in range(batch_size):
        image_a = image[i]
        label_a = label[i]
        j = tf.cast(tf.random.uniform([],0, batch_size),tf.int32)
        image_b = image[j]
        label_b = label[j]
        xa, ya, xb, yb = get_clip_box(image_a, image_b)
        mixed_imgs.append(mix_2_images(image_a, image_b, xa, ya, xb, yb))
        mixed_labels.append(mix_2_label(label_a, label_b, xa, ya, xb, yb))

    mixed_imgs = tf.reshape(tf.stack(mixed_imgs),(batch_size, img_size, img_size, 3))
    mixed_labels = tf.reshape(tf.stack(mixed_labels), (batch_size, num_classes))
    
    return mixed_imgs, mixed_labels
```

## 2-6. 수정된 apply_normalize_on_datasset( )


```python
def normalize_and_resize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.resize(image, [224, 224])
    return tf.cast(image, tf.float32) / 255., label

def augment(image,label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_crop(image, [224, 224, 3])
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.2, upper=0.5)
    return image,label

def onehot(image, label) :
    label = tf.one_hot(label, num_classes)
#     label = tf.reshape(tf.stack(label),(batch_size, num_classes))
    return image, label

def apply_normalize_on_dataset(ds, is_test=False, batch_size=16, with_aug=False, with_cutmix=False):
    ds = ds.map(
        normalize_and_resize_img, 
        num_parallel_calls=2
    )
    if not is_test and with_aug:
        ds = ds.map(
            augment
        )
    
    ds = ds.batch(batch_size)
    
    if not is_test and with_cutmix:
        ds = ds.map(
            cutmix,
            num_parallel_calls=2
        )
        
    else:
        ds = ds.map(
            onehot,
            num_parallel_calls=2
        )
    
    if not is_test:
        ds = ds.repeat()
        ds = ds.shuffle(200)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds
```


```python
num_classes = ds_info.features["label"].num_classes
```


```python
apply_normalize_on_dataset(ds_train, batch_size=16, with_aug=False, with_cutmix=True)
```




    <PrefetchDataset shapes: ((16, 224, 224, 3), (16, 120)), types: (tf.float32, tf.float32)>



# 3. 모델 생성 및 훈련
## 3-1. 데이터 준비
### 🔺 하이퍼파라미터


```python
from tensorflow.keras.callbacks import EarlyStopping

lr = 0.005
batch_size = 16

es_3 = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
es_5 = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
es = es_5

steps_per_epoch=int(ds_info.splits['train'].num_examples/batch_size)
validation_steps=int(ds_info.splits['test'].num_examples/batch_size)
epochs=10
```


```python
num_classes = ds_info.features["label"].num_classes
num_classes
```




    120




```python
(ds_train, ds_test), ds_info = tfds.load(
    'stanford_dogs',
    split=['train', 'test'],
    as_supervised=True,
    shuffle_files=True,
    with_info=True,
)

ds_train_normal = apply_normalize_on_dataset(
    ds_train,
    batch_size=batch_size,
    with_aug=False,
    with_cutmix=False,
)
ds_train_aug = apply_normalize_on_dataset(
    ds_train,
    batch_size=batch_size,
    with_aug=True,
    with_cutmix=False,
)
ds_train_cutmix = apply_normalize_on_dataset(
    ds_train,
    batch_size=batch_size,
    with_aug=False,
    with_cutmix=True,
)
ds_train_all = apply_normalize_on_dataset(
    ds_train,
    batch_size=batch_size,
    with_aug=True,
    with_cutmix=True
)

ds_test = apply_normalize_on_dataset(ds_test, is_test = True)
```

## 3-2. 모델
### ResNet50, Normal Data


```python
tf.random.set_seed(1994)
```


```python
resnet50 = keras.models.Sequential([
    keras.applications.resnet.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224,3),
        pooling='avg',
    ),
    keras.layers.Dense(num_classes, activation = 'softmax')
])
```


```python
resnet50.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(lr=lr),
    metrics=['accuracy'],
)

history_resnet50_normal = resnet50.fit(
    ds_train_normal, # augmentation 적용하지 않은 데이터셋 사용
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
    callbacks=[es]
)
```

    /opt/conda/lib/python3.9/site-packages/keras/optimizer_v2/optimizer_v2.py:355: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      warnings.warn(


    Epoch 1/10
    750/750 [==============================] - 166s 208ms/step - loss: 2.7461 - accuracy: 0.3682 - val_loss: 3.3104 - val_accuracy: 0.2162
    Epoch 2/10
    750/750 [==============================] - 155s 206ms/step - loss: 1.0444 - accuracy: 0.7220 - val_loss: 1.1312 - val_accuracy: 0.6726
    Epoch 3/10
    750/750 [==============================] - 156s 208ms/step - loss: 0.4662 - accuracy: 0.8963 - val_loss: 0.9877 - val_accuracy: 0.7084
    Epoch 4/10
    750/750 [==============================] - 155s 206ms/step - loss: 0.1994 - accuracy: 0.9745 - val_loss: 0.9445 - val_accuracy: 0.7217
    Epoch 5/10
    750/750 [==============================] - 155s 207ms/step - loss: 0.0940 - accuracy: 0.9948 - val_loss: 0.9408 - val_accuracy: 0.7255
    Epoch 6/10
    750/750 [==============================] - 155s 206ms/step - loss: 0.0562 - accuracy: 0.9975 - val_loss: 0.9648 - val_accuracy: 0.7204
    Epoch 7/10
    750/750 [==============================] - 169s 226ms/step - loss: 0.0361 - accuracy: 0.9993 - val_loss: 0.9659 - val_accuracy: 0.7259
    Epoch 8/10
    750/750 [==============================] - 155s 207ms/step - loss: 0.0281 - accuracy: 0.9992 - val_loss: 0.9985 - val_accuracy: 0.7229
    Epoch 9/10
    750/750 [==============================] - 155s 207ms/step - loss: 0.0210 - accuracy: 0.9999 - val_loss: 0.9981 - val_accuracy: 0.7259
    Epoch 10/10
    750/750 [==============================] - 155s 207ms/step - loss: 0.0180 - accuracy: 0.9997 - val_loss: 0.9988 - val_accuracy: 0.7267
    Epoch 00010: early stopping



```python
del resnet50
```

### ResNet50, Augmented data


```python
aug_resnet50 = keras.models.Sequential([
    keras.applications.resnet.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224,3),
        pooling='avg',
    ),
    keras.layers.Dense(num_classes, activation = 'softmax')
])
```


```python
aug_resnet50.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(lr=lr),
    metrics=['accuracy'],
)

history_resnet50_aug = aug_resnet50.fit(
   ds_train_aug, # augmentation 적용하지 않은 데이터셋 사용
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
    callbacks=[es]
)
```

    Epoch 1/10
    750/750 [==============================] - 163s 207ms/step - loss: 2.9537 - accuracy: 0.3157 - val_loss: 1.9753 - val_accuracy: 0.4683
    Epoch 2/10
    750/750 [==============================] - 155s 206ms/step - loss: 1.3266 - accuracy: 0.6395 - val_loss: 1.5007 - val_accuracy: 0.5689
    Epoch 3/10
    750/750 [==============================] - 154s 206ms/step - loss: 0.8752 - accuracy: 0.7566 - val_loss: 1.4804 - val_accuracy: 0.5780
    Epoch 4/10
    750/750 [==============================] - 154s 206ms/step - loss: 0.5601 - accuracy: 0.8549 - val_loss: 1.2650 - val_accuracy: 0.6406
    Epoch 5/10
    750/750 [==============================] - 155s 206ms/step - loss: 0.3690 - accuracy: 0.9104 - val_loss: 1.2363 - val_accuracy: 0.6543
    Epoch 6/10
    750/750 [==============================] - 155s 206ms/step - loss: 0.2452 - accuracy: 0.9507 - val_loss: 1.1976 - val_accuracy: 0.6658
    Epoch 7/10
    750/750 [==============================] - 155s 206ms/step - loss: 0.1613 - accuracy: 0.9734 - val_loss: 1.1191 - val_accuracy: 0.6895
    Epoch 8/10
    750/750 [==============================] - 155s 206ms/step - loss: 0.1086 - accuracy: 0.9875 - val_loss: 1.1801 - val_accuracy: 0.6793
    Epoch 9/10
    750/750 [==============================] - 154s 206ms/step - loss: 0.0736 - accuracy: 0.9931 - val_loss: 1.1493 - val_accuracy: 0.6884
    Epoch 10/10
    750/750 [==============================] - 154s 206ms/step - loss: 0.0546 - accuracy: 0.9958 - val_loss: 1.1380 - val_accuracy: 0.6997



```python
del aug_resnet50
```

### ResNet50, CutMix data


```python
cutmix_resnet50 = keras.models.Sequential([
    keras.applications.resnet.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224,3),
        pooling='avg',
    ),
    keras.layers.Dense(num_classes, activation = 'softmax')
])
```


```python
cutmix_resnet50.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(lr=lr),
    metrics=['accuracy'],
)

history_resnet50_cutmix = cutmix_resnet50.fit(
    ds_train_cutmix, # augmentation 적용하지 않은 데이터셋 사용
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
    callbacks=[es]
)
```

    Epoch 1/10
    750/750 [==============================] - 164s 209ms/step - loss: 4.1208 - accuracy: 0.1553 - val_loss: 3.6573 - val_accuracy: 0.1700
    Epoch 2/10
    750/750 [==============================] - 156s 208ms/step - loss: 3.1458 - accuracy: 0.3869 - val_loss: 1.4131 - val_accuracy: 0.6143
    Epoch 3/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.8140 - accuracy: 0.4868 - val_loss: 1.2281 - val_accuracy: 0.6556
    Epoch 4/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.5961 - accuracy: 0.5497 - val_loss: 1.1241 - val_accuracy: 0.6852
    Epoch 5/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.4474 - accuracy: 0.6033 - val_loss: 1.0853 - val_accuracy: 0.6964
    Epoch 6/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.3189 - accuracy: 0.6449 - val_loss: 1.1218 - val_accuracy: 0.6840
    Epoch 7/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.2459 - accuracy: 0.6686 - val_loss: 1.0177 - val_accuracy: 0.7106
    Epoch 8/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.1313 - accuracy: 0.7022 - val_loss: 1.0507 - val_accuracy: 0.7112
    Epoch 9/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.0910 - accuracy: 0.7101 - val_loss: 1.0375 - val_accuracy: 0.7135
    Epoch 10/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.0079 - accuracy: 0.7374 - val_loss: 1.0965 - val_accuracy: 0.7077



```python
del cutmix_resnet50
```

### ResNet50, Augmented + CutMix data


```python
all_resnet50 = keras.models.Sequential([
    keras.applications.resnet.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224,3),
        pooling='avg',
    ),
    keras.layers.Dense(num_classes, activation = 'softmax')
])
```


```python
all_resnet50.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(lr=lr),
    metrics=['accuracy'],
)

history_resnet50_all = all_resnet50.fit(
    ds_train_all, # augmentation 적용하지 않은 데이터셋 사용
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
    callbacks=[es]
)
```

    Epoch 1/10
    750/750 [==============================] - 162s 206ms/step - loss: 4.3420 - accuracy: 0.1058 - val_loss: 2.5408 - val_accuracy: 0.3322
    Epoch 2/10
    750/750 [==============================] - 155s 207ms/step - loss: 3.4114 - accuracy: 0.3091 - val_loss: 1.8795 - val_accuracy: 0.4860
    Epoch 3/10
    750/750 [==============================] - 156s 208ms/step - loss: 3.0447 - accuracy: 0.4095 - val_loss: 1.4650 - val_accuracy: 0.5946
    Epoch 4/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.8744 - accuracy: 0.4577 - val_loss: 1.4262 - val_accuracy: 0.6051
    Epoch 5/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.7121 - accuracy: 0.5132 - val_loss: 1.3631 - val_accuracy: 0.6163
    Epoch 6/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.6124 - accuracy: 0.5368 - val_loss: 1.6807 - val_accuracy: 0.5465
    Epoch 7/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.5396 - accuracy: 0.5592 - val_loss: 1.6819 - val_accuracy: 0.5475
    Epoch 8/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.4319 - accuracy: 0.5944 - val_loss: 1.1993 - val_accuracy: 0.6649
    Epoch 9/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.3939 - accuracy: 0.6136 - val_loss: 1.2280 - val_accuracy: 0.6586
    Epoch 10/10
    750/750 [==============================] - 156s 208ms/step - loss: 2.3104 - accuracy: 0.6326 - val_loss: 1.2574 - val_accuracy: 0.6582



```python
del all_resnet50
```

## 4. 결과 비교


```python
plt.figure(figsize=(20, 10))
plt.plot(history_resnet50_normal.history['val_accuracy'], 'r')
plt.plot(history_resnet50_aug.history['val_accuracy'], 'b')
plt.plot(history_resnet50_cutmix.history['val_accuracy'], 'g')
plt.plot(history_resnet50_all.history['val_accuracy'], 'k')
plt.title('Model validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['No Augmentation', 'With Augmentation', 'With CutMix', 'Augmented+CutMix'], loc='best')
plt.grid(True)
plt.savefig('fig1_Accuracy.png')
plt.show()
```


    
![png](output_47_0.png)
    



```python
plt.figure(figsize=(20,10))
plt.plot(history_resnet50_normal.history['val_loss'], 'r')
plt.plot(history_resnet50_aug.history['val_loss'], 'b')
plt.plot(history_resnet50_cutmix.history['val_loss'], 'g')
plt.plot(history_resnet50_all.history['val_loss'], 'k')
plt.title('Model validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['No Augmentation', 'With Augmentation', 'With CutMix', 'Augmented+CutMix'], loc='best')
plt.grid(True)
plt.savefig('fig1_loss.png')
plt.show()
```


    
![png](output_48_0.png)
    


# 회고
- loss와 accuracy 그래프를 비교해보았을때 'Augmented+CutMix'가 중간에 약간 튀는 현상이 있지만 이후에는 다른 모델과 큰 차이가 없다는 것을 알 수 있다. 
- 이번 노드를 하면서 '모델을 구현하는게 이렇게 어려운거구나...'라는 걸 알았다.
- 그리고 학습시간이 너무 오래 걸려서 다양한 시도를 하지 못해서 아쉽다.
- 모델 구현 진짜 열심히 공부해야겠다^^;;
