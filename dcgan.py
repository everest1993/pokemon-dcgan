import tensorflow as tf
from tensorflow import keras


latent = 256

# generator (upsample 2x)
generator = keras.models.Sequential([
    # espande un vettore (256, ) in un vettore lungo 15*15*256 = 57600
    keras.layers.Dense(15*15*256, use_bias=False, input_shape=(latent,)), # 15 x 15
    keras.layers.BatchNormalization(), # normalizza le attivazioni del layer precedente
    keras.layers.ReLU(), # max(0, x) -> spinge attivazioni positive e facilita la propagazione del gradiente
    
    # trasforma il vettore 57600 in una matrice di feature (15, 15, 256) per iniziare a usare convoluzioni 
    # e costruire immagini in modo spaziale con Conv2DTranspose
    keras.layers.Reshape((15, 15, 256)),

    # upsampling -> raddoppio delle dimensioni spaziali da (15, 15, 256) a (30, 30, 128)
    keras.layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False), # 30 x 30
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),

    keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False), # (60, 60, 64)
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),

    keras.layers.Conv2DTranspose(32, 4, strides=2, padding="same", use_bias=False), # (120, 120, 32)
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),

    # converte le feature in un’immagine RGB (da canali a 3)
    keras.layers.Conv2D(3, 3, padding="same", activation="tanh")
])

# discriminator
discriminator = keras.models.Sequential([
    # downsampling da (120, 120, 3) a (60, 60, 64)
    keras.layers.Conv2D(64, 4, strides=2, padding="same", input_shape=(120, 120, 3)),
    # come ReLU ma lascia passare anche i negativi: x se x > 0, alpha * x se x < 0
    keras.layers.LeakyReLU(),

    keras.layers.Conv2D(128, 4, strides=2, padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(0.2),

    keras.layers.Conv2D(256, 4, strides=2, padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(0.2),

    keras.layers.Conv2D(512, 4, strides=2, padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(0.2),

    # per ogni canale fa la media spaziale: (8, 8, 512) -> (512, )
    keras.layers.GlobalAveragePooling2D(),

    # spegne casualmente il 30% dei neuroni durante training per impedire al discriminatore di diventare 
    # troppo bravo troppo in fretta, cosa che può bloccare il generatore.
    keras.layers.Dropout(0.5),

    keras.layers.Dense(1)
])

# DCGAN
class DCGAN(tf.keras.Model):
    def compile(self, g_opt, d_opt, loss_fn):
        super(DCGAN, self).compile()
        self.g_opt, self.d_opt, self.loss_fn = g_opt, d_opt, loss_fn


    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        # crea un batch di rumore gaussiano con forma (batch_size, latent), input del generatore
        noise = tf.random.normal([batch_size, latent])

        # crea due registratori di gradiente
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = generator(noise, training=True) # generazione immagini false
            real_pred = discriminator(real_images, training=True) # classificazione immagini reali
            fake_pred = discriminator(fake_images, training=True) # classificazione immagini false

            # 1 per immagini reali, 0 per immagini false nel discriminatore
            d_loss = self.loss_fn(tf.ones_like(real_pred), real_pred) + \
                     self.loss_fn(tf.zeros_like(fake_pred), fake_pred)
            # il generatore viene premiato se riesce a far diventare fake_pred positivo
            g_loss = self.loss_fn(tf.ones_like(fake_pred), fake_pred)

        # calcola i gradienti di loss rispetto ai parametri allenabili e accoppia ogni gradiente con 
        # la variabile corrispondente (zip)
        self.g_opt.apply_gradients(zip(gen_tape.gradient(g_loss, generator.trainable_variables), 
                                       generator.trainable_variables))
        self.d_opt.apply_gradients(zip(disc_tape.gradient(d_loss, discriminator.trainable_variables), 
                                       discriminator.trainable_variables))
        return {"d_loss": d_loss, "g_loss": g_loss}