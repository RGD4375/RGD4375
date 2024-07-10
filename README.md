- 👋 Hi, I’m @RGD4375
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
RGD4375/RGD4375 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
import tensorflow as tf

# تحميل مجموعة بيانات MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# تسوية البيانات إلى نطاق [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# بناء النموذج البسيط
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# تعريف دالة فقدان وأسلوب الخسارة والتحسين
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# تدريب النموذج
model.fit(x_train, y_train, epochs=5)

# تقييم النموذج
model.evaluate(x_test, y_test, verbose=2)

