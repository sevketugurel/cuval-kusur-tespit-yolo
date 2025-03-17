"""
Cycle-GAN implementation for synthetic data generation in sack defect detection.
Reference: https://arxiv.org/abs/1703.10593
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from glob import glob
import time
from tqdm import tqdm

class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer."""
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

def downsample(filters, size, apply_norm=True):
    """Downsampling block for the discriminator and generator."""
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    
    if apply_norm:
        result.add(tfa.layers.InstanceNormalization())
    
    result.add(layers.LeakyReLU(0.2))
    
    return result

def upsample(filters, size, apply_dropout=False):
    """Upsampling block for the generator."""
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    
    result.add(tfa.layers.InstanceNormalization())
    
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    
    result.add(layers.ReLU())
    
    return result

def residual_block(input_tensor, filters, kernel_size=3):
    """Residual block for the generator."""
    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(filters, kernel_size, padding='valid', use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)
    
    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(filters, kernel_size, padding='valid', use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    
    return layers.add([input_tensor, x])

def build_generator(input_shape=(256, 256, 3), filters=64, num_residual_blocks=9):
    """Build the generator model."""
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution block
    x = ReflectionPadding2D(padding=(3, 3))(inputs)
    x = layers.Conv2D(filters, 7, padding='valid', use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)
    
    # Downsampling
    n_downsampling = 2
    for i in range(n_downsampling):
        filters *= 2
        x = downsample(filters, 3, apply_norm=True)(x)
    
    # Residual blocks
    for i in range(num_residual_blocks):
        x = residual_block(x, filters)
    
    # Upsampling
    for i in range(n_downsampling):
        filters //= 2
        x = upsample(filters, 3, apply_dropout=False)(x)
    
    # Final output block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, 7, padding='valid', activation='tanh')(x)
    
    return keras.Model(inputs=inputs, outputs=x)

def build_discriminator(input_shape=(256, 256, 3), filters=64):
    """Build the discriminator model."""
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inputs = layers.Input(shape=input_shape)
    
    # First layer doesn't use instance normalization
    x = layers.Conv2D(filters, 4, strides=2, padding='same',
                      kernel_initializer=initializer)(inputs)
    x = layers.LeakyReLU(0.2)(x)
    
    # Downsampling
    filter_sizes = [filters * 2, filters * 4, filters * 8]
    for filter_size in filter_sizes:
        x = layers.Conv2D(filter_size, 4, strides=2, padding='same',
                          kernel_initializer=initializer, use_bias=False)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
    
    # Final layer
    x = layers.Conv2D(1, 4, strides=1, padding='same',
                      kernel_initializer=initializer)(x)
    
    return keras.Model(inputs=inputs, outputs=x)

class CycleGAN:
    """CycleGAN model for image-to-image translation."""
    def __init__(self, input_shape=(256, 256, 3), lambda_cycle=10.0, lambda_identity=0.5):
        self.input_shape = input_shape
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        # Build generators and discriminators
        self.g_AB = build_generator(input_shape)
        self.g_BA = build_generator(input_shape)
        self.d_A = build_discriminator(input_shape)
        self.d_B = build_discriminator(input_shape)
        
        # Define optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        # Initialize checkpoint manager
        self.checkpoint_dir = './checkpoints/cyclegan'
        self.checkpoint = tf.train.Checkpoint(
            generator_AB=self.g_AB,
            generator_BA=self.g_BA,
            discriminator_A=self.d_A,
            discriminator_B=self.d_B,
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_dir, max_to_keep=5)
        
    def load_checkpoint(self):
        """Load the latest checkpoint if it exists."""
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"Restored from {self.checkpoint_manager.latest_checkpoint}")
            return True
        return False
    
    @tf.function
    def train_step(self, real_A, real_B):
        """Execute a single training step."""
        with tf.GradientTape(persistent=True) as tape:
            # Generate fake images
            fake_B = self.g_AB(real_A, training=True)
            fake_A = self.g_BA(real_B, training=True)
            
            # Cycle consistency
            cycled_A = self.g_BA(fake_B, training=True)
            cycled_B = self.g_AB(fake_A, training=True)
            
            # Identity mapping
            same_A = self.g_BA(real_A, training=True)
            same_B = self.g_AB(real_B, training=True)
            
            # Discriminator outputs
            disc_real_A = self.d_A(real_A, training=True)
            disc_fake_A = self.d_A(fake_A, training=True)
            disc_real_B = self.d_B(real_B, training=True)
            disc_fake_B = self.d_B(fake_B, training=True)
            
            # Generator adversarial loss
            gen_A_loss = self.generator_loss(disc_fake_A)
            gen_B_loss = self.generator_loss(disc_fake_B)
            
            # Cycle consistency loss
            cycle_A_loss = self.cycle_loss(real_A, cycled_A) * self.lambda_cycle
            cycle_B_loss = self.cycle_loss(real_B, cycled_B) * self.lambda_cycle
            
            # Identity loss
            id_A_loss = self.identity_loss(real_A, same_A) * self.lambda_cycle * self.lambda_identity
            id_B_loss = self.identity_loss(real_B, same_B) * self.lambda_cycle * self.lambda_identity
            
            # Total generator loss
            total_gen_loss = gen_A_loss + gen_B_loss + cycle_A_loss + cycle_B_loss + id_A_loss + id_B_loss
            
            # Discriminator loss
            disc_A_loss = self.discriminator_loss(disc_real_A, disc_fake_A)
            disc_B_loss = self.discriminator_loss(disc_real_B, disc_fake_B)
            total_disc_loss = disc_A_loss + disc_B_loss
        
        # Calculate gradients and apply to optimizers
        generator_gradients = tape.gradient(total_gen_loss, 
                                           self.g_AB.trainable_variables + self.g_BA.trainable_variables)
        discriminator_gradients = tape.gradient(total_disc_loss,
                                               self.d_A.trainable_variables + self.d_B.trainable_variables)
        
        self.generator_optimizer.apply_gradients(zip(generator_gradients, 
                                                   self.g_AB.trainable_variables + self.g_BA.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                       self.d_A.trainable_variables + self.d_B.trainable_variables))
        
        return {
            'gen_loss': total_gen_loss,
            'disc_loss': total_disc_loss,
            'gen_A_loss': gen_A_loss,
            'gen_B_loss': gen_B_loss,
            'cycle_A_loss': cycle_A_loss,
            'cycle_B_loss': cycle_B_loss
        }
    
    def generator_loss(self, generated_output):
        """Generator loss based on whether it fooled the discriminator."""
        return tf.reduce_mean(tf.losses.binary_crossentropy(
            tf.ones_like(generated_output), generated_output, from_logits=True))
    
    def discriminator_loss(self, real_output, generated_output):
        """Discriminator loss based on real and generated images."""
        real_loss = tf.reduce_mean(tf.losses.binary_crossentropy(
            tf.ones_like(real_output), real_output, from_logits=True))
        generated_loss = tf.reduce_mean(tf.losses.binary_crossentropy(
            tf.zeros_like(generated_output), generated_output, from_logits=True))
        return real_loss + generated_loss
    
    def cycle_loss(self, real_image, cycled_image):
        """Cycle consistency loss."""
        return tf.reduce_mean(tf.abs(real_image - cycled_image))
    
    def identity_loss(self, real_image, same_image):
        """Identity loss for preserving color and content."""
        return tf.reduce_mean(tf.abs(real_image - same_image))
    
    def train(self, train_A, train_B, epochs=40, batch_size=1, save_interval=5):
        """Train the CycleGAN model."""
        # Create datasets
        train_A = tf.data.Dataset.from_tensor_slices(train_A).batch(batch_size)
        train_B = tf.data.Dataset.from_tensor_slices(train_B).batch(batch_size)
        
        # Create a zip dataset
        train_dataset = tf.data.Dataset.zip((train_A, train_B))
        
        # Training loop
        for epoch in range(epochs):
            start = time.time()
            
            # Initialize metrics
            gen_loss_sum = 0
            disc_loss_sum = 0
            n_batches = 0
            
            for real_A, real_B in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{epochs}"):
                losses = self.train_step(real_A, real_B)
                gen_loss_sum += losses['gen_loss']
                disc_loss_sum += losses['disc_loss']
                n_batches += 1
            
            # Calculate average losses
            avg_gen_loss = gen_loss_sum / n_batches
            avg_disc_loss = disc_loss_sum / n_batches
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, "
                  f"Time: {time.time()-start:.2f}s")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.checkpoint_manager.save()
                print(f"Checkpoint saved at epoch {epoch+1}")
    
    def generate_images(self, test_input, direction='AB'):
        """Generate images using the trained generators."""
        if direction == 'AB':
            prediction = self.g_AB(test_input)
        else:
            prediction = self.g_BA(test_input)
        
        # Convert from [-1, 1] to [0, 1]
        prediction = (prediction * 0.5 + 0.5).numpy()
        test_input = (test_input * 0.5 + 0.5).numpy()
        
        return test_input, prediction

def preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess an image for the CycleGAN."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = (img / 127.5) - 1  # Normalize to [-1, 1]
    return img

def load_images(image_dir, target_size=(256, 256)):
    """Load all images from a directory."""
    image_paths = glob(os.path.join(image_dir, '*.jpg')) + glob(os.path.join(image_dir, '*.png'))
    images = []
    
    for path in tqdm(image_paths, desc="Loading images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    
    return tf.stack(images) if images else None

def main():
    """Main function to demonstrate CycleGAN usage."""
    # Example usage
    normal_dir = 'data/normal'
    defect_dir = 'data/defect'
    
    # Load images
    normal_images = load_images(normal_dir)
    defect_images = load_images(defect_dir)
    
    if normal_images is None or defect_images is None:
        print("Error: Could not load images. Please check the directories.")
        return
    
    # Create and train CycleGAN
    cyclegan = CycleGAN()
    cyclegan.train(normal_images, defect_images, epochs=50, batch_size=1)
    
    # Generate some sample images
    sample_normal = normal_images[:5]
    sample_defect = defect_images[:5]
    
    # Generate defects from normal images
    _, generated_defects = cyclegan.generate_images(sample_normal, direction='AB')
    
    # Generate normal images from defects
    _, generated_normal = cyclegan.generate_images(sample_defect, direction='BA')
    
    # Save the generated images
    output_dir = 'output/cyclegan'
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img in enumerate(generated_defects):
        plt.imsave(os.path.join(output_dir, f'normal_to_defect_{i}.png'), img)
    
    for i, img in enumerate(generated_normal):
        plt.imsave(os.path.join(output_dir, f'defect_to_normal_{i}.png'), img)
    
    print(f"Generated images saved to {output_dir}")

if __name__ == '__main__':
    main() 