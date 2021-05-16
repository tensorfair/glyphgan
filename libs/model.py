from .util import *
import imageio
from tqdm import tqdm


class WPGAN:
    def __init__(self, specs, location):
        """
        :param specs: dictionary of info on input and output data.
        :param location: directory path to save to.
        """
        self.specs = specs
        self.location = location
        self.kernel = 4

    def generator(self, embedding):
        """
        :param embedding: embedded data to reconstruct.
        :return: This function returns the reconstructed embedding image.
        """

        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            embedding = tf.layers.conv2d_transpose(embedding, filters=512, kernel_size=self.kernel, strides=1, padding="valid", activation=tf.nn.relu, name='e_1')
            embedding = tf.layers.conv2d_transpose(embedding, filters=256, kernel_size=self.kernel, strides=2, padding="same", activation=tf.nn.relu, name='e_2')
            embedding = tf.layers.conv2d_transpose(embedding, filters=128, kernel_size=self.kernel, strides=2, padding="same", activation=tf.nn.relu, name='e_3')
            embedding = tf.layers.conv2d_transpose(embedding, filters=64, kernel_size=self.kernel, strides=2, padding="same", activation=tf.nn.relu, name='e_4')
            embedding = tf.layers.conv2d_transpose(embedding, filters=3, kernel_size=self.kernel, strides=2, padding="same", name="e_5")
            generated = tf.nn.sigmoid(embedding, name="generated")
            return generated

    def discriminator(self, target):
        """
        :param target: data to discriminate.
        :return: This function returns a sigmoid value of real or fake.
        """

        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            target = tf.layers.conv2d(target, filters=64, kernel_size=self.kernel, strides=2, padding="same", activation=tf.nn.leaky_relu)
            target = tf.layers.conv2d(target, filters=128, kernel_size=self.kernel, strides=2, padding="same", activation=tf.nn.leaky_relu)
            target = tf.layers.conv2d(target, filters=256, kernel_size=self.kernel, strides=2, padding="same", activation=tf.nn.leaky_relu)
            target = tf.layers.conv2d(target, filters=512, kernel_size=self.kernel, strides=2, padding="same", activation=tf.nn.leaky_relu)
            decision = tf.layers.conv2d(target, filters=1, kernel_size=self.kernel, strides=1, padding="valid", activation=tf.nn.leaky_relu)
            return decision

    def optimize(self, loss, scope):
        """
        :param loss: the classifier loss.
        :param scope: the model's variable scope.
        :return: This function returns a model's optimizer.
        """
        ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
        with tf.control_dependencies(ops):
            var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            return tf.train.AdamOptimizer(self.specs['lr']).minimize(loss, var_list=var)

    def generator_graph(self, target_embedded):
        """
        :param target_embedded: ground truth image embedding.
        :return: This function returns the generator graph of the model and the reconstructed and encoded outputs.
        """
        generated = self.generator(target_embedded)
        fake = self.discriminator(generated)
        generator_loss = -tf.reduce_mean(fake)
        generator_ops = self.optimize(generator_loss, 'generator')
        return generator_ops, generated

    def discriminator_graph(self, target, generated):
        """
        :param target: ground truth image.
        :param generated: generated image.
        :return: This function returns the discriminator graph of the model.
        """
        real = self.discriminator(target)
        fake = self.discriminator(generated)
        sum_hat = tf.random_uniform([], 0.0, 1.0) * target + (1 - tf.random_uniform([], 0.0, 1.0)) * generated
        summed = self.discriminator(sum_hat)
        gradient_penalty = tf.gradients(summed, sum_hat)[0]
        gradient_penalty = tf.sqrt(tf.reduce_sum(tf.square(gradient_penalty), axis=1))
        gradient_penalty = tf.reduce_mean(tf.square(gradient_penalty - 1.0) * 10)
        gan = tf.reduce_mean(fake) - tf.reduce_mean(real)
        discriminator_loss = gan + gradient_penalty
        discriminator_ops = self.optimize(discriminator_loss, 'discriminator')
        return discriminator_ops

    def freeze_generator(self):
        """
        :return: This function returns the frozen generator model.
        """
        tf.reset_default_graph()
        target_embedded = tf.placeholder(tf.float32, [None, 1, 1, self.specs['embedding']], name='target_embedded_ph')
        reconstructed = self.generator(target_embedded)
        with tf.Session(config=tf_config_setup()) as sess:
            coord, threads = initialization(sess)
            generator_variables = [var for var in tf.global_variables() if var.name.startswith('generator/e_') or var.name.startswith('generator/generated')]
            generator_saver = tf.train.Saver(generator_variables)
            generator_saver.restore(sess, os.path.join(self.location, 'generator.ckpt'))
            prepared_embedding = prepare_input(self.specs['batch_size'], 0)
            epoch_feed = {target_embedded: prepared_embedding}
            generated_output = sess.run(reconstructed, epoch_feed)
            imageio.imwrite('final_generated.png', (generated_output[0]*255).astype(np.uint8))
            reconstructed_node_names = ['generator/generated']
            reconstructed_frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, reconstructed_node_names)
            with open(os.path.join(self.location, 'generator_graph.pb'), 'wb') as f:
                f.write(reconstructed_frozen_graph_def.SerializeToString())
            coord.request_stop()
            coord.join(threads)
        return

    def train(self):
        """
        :return: This function runs the model training.
        """
        get_batch_dir = build_inputs(self.specs, self.specs['dir'])
        target = tf.placeholder(tf.float32, [None, None, None, 3], name='target_ph')
        target_embedded = tf.placeholder(tf.float32, [None, 1, 1, self.specs['embedding']], name='target_embedded_ph')
        generator_ops, generated = self.generator_graph(target_embedded)
        discriminator_ops = self.discriminator_graph(target, generated)
        with tf.Session(config=tf_config_setup()) as sess:
            coord, threads = initialization(sess)
            generator_variables = [var for var in tf.global_variables() if var.name.startswith('generator/e_') or var.name.startswith('generator/generated')]
            generator_saver = tf.train.Saver(generator_variables)
            epoch = 0
            while True:
                if epoch < self.specs['epochs']:
                    for character in tqdm(range(0, 26)):
                        prepared_embedding = prepare_input(self.specs['batch_size'], character)
                        target_batch = sess.run(get_batch_dir['%d' % character])
                        target_batch = target_batch / 255
                        discriminator_feed = {target_embedded: prepared_embedding, target: target_batch}
                        for i in range(self.specs['d_loops']):
                            sess.run(discriminator_ops, discriminator_feed)
                        generator_feed = {target_embedded: prepared_embedding}
                        _, generated_output = sess.run([generator_ops, generated], generator_feed)
                        if epoch % 100 == 0:
                            progress = build_log_dir(self.location, 'progress')
                            imageio.imwrite(os.path.join(progress, '%d_%d_model_output.png' % (character, epoch)), (generated_output[0]*255).astype(np.uint8))
                    if epoch % 100 == 0:
                        generator_saver.save(sess, os.path.join(self.location, 'generator.ckpt'))
                    epoch += 1
                    print("Epoch %d/%d complete." % (epoch, self.specs['epochs']))
                else:
                    break
            coord.request_stop()
            coord.join(threads)
        return
