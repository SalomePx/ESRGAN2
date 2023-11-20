from absl import logging
from tqdm import tqdm
import tensorflow as tf

from models import RRDB_Model, DiscriminatorVGG128
from lr_scheduler import MultiStepLR
from loss import PixelLoss, ContentLoss, DiscriminatorLoss, GeneratorLoss
from utils import ProgressBar, set_memory_growth
from config import LOAD_MODEL, PATCH_SIZE, NUM_EPOCHS


def train(train_dataset, val_dataset, model_savedir):
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    # define network
    generator = RRDB_Model(PATCH_SIZE, 1, {'nf': 64, 'nb': 23 })
    generator.summary(line_length=80)
    discriminator = DiscriminatorVGG128(PATCH_SIZE, 1)
    discriminator.summary(line_length=80)

    # define optimizer
    lr_steps = [50000, 100000, 200000, 300000]
    lr_rate = 0.5
    learning_rate_G = MultiStepLR(1e-4, lr_steps, lr_rate)
    learning_rate_D = MultiStepLR(1e-4, lr_steps, lr_rate)
    optimizer_G = tf.keras.optimizers.Adam(learning_rate_G, beta_1=0.9, beta_2=0.99)
    optimizer_D = tf.keras.optimizers.Adam(learning_rate_D, beta_1=0.9, beta_2=0.99)

    # define losses function
    pixel_loss_fn = PixelLoss(criterion='l1')
    fea_loss_fn = ContentLoss(criterion='l1')
    gen_loss_fn = GeneratorLoss(gan_type='gan')
    dis_loss_fn = DiscriminatorLoss(gan_type='gan')

    # load checkpoint
    checkpoint_dir = f'{model_savedir}/checkpoints'
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'), optimizer_G=optimizer_G, optimizer_D=optimizer_D, model=generator, discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_dir, max_to_keep=None)

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {} at step {}.'.format(
            manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        if LOAD_MODEL is not None:
            pretrain_dir = f'{model_savedir}/checkpoints'
            if tf.train.latest_checkpoint(pretrain_dir):
                checkpoint.restore(tf.train.latest_checkpoint(pretrain_dir))
                checkpoint.step.assign(0)
                print("[*] training from pretrain model {}.".format(pretrain_dir))
            else:
                print("[*] cannot find pretrain model {}.".format(pretrain_dir))
        else:
            print("[*] training from scratch.")

    # define training step function
    @tf.function
    def train_step(lr, hr):
        with tf.GradientTape(persistent=True) as tape:
            sr = generator(lr, training=True)
            hr_output = discriminator(hr, training=True)
            sr_output = discriminator(sr, training=True)

            losses_G = {}
            losses_D = {}
            losses_G['reg'] = tf.reduce_sum(generator.losses)
            losses_D['reg'] = tf.reduce_sum(discriminator.losses)
            losses_G['pixel'] = 1e-2 * pixel_loss_fn(hr, sr)
            losses_G['feature'] = 1.0 * fea_loss_fn(hr, sr)

            losses_G['gan'] = 5e-3 * gen_loss_fn(hr_output, sr_output)
            losses_D['gan'] = dis_loss_fn(hr_output, sr_output)

            total_loss_G = tf.add_n([l for l in losses_G.values()])
            total_loss_D = tf.add_n([l for l in losses_D.values()])

        grads_G = tape.gradient(total_loss_G, generator.trainable_variables)
        grads_D = tape.gradient(total_loss_D, discriminator.trainable_variables)
        optimizer_G.apply_gradients(zip(grads_G, generator.trainable_variables))
        optimizer_D.apply_gradients(zip(grads_D, discriminator.trainable_variables))

        return total_loss_G, total_loss_D, losses_G, losses_D

    # training loop
    summary_writer = tf.summary.create_file_writer('./logs/')
    prog_bar = ProgressBar(NUM_EPOCHS, checkpoint.step.numpy())

    for epoch in range(NUM_EPOCHS):
        print(f"STARTING EPOCH {epoch}")
        for step, (x_batch, y_batch) in tqdm(enumerate(train_dataset), total=len(train_dataset)):
            checkpoint.step.assign_add(1)
            steps = checkpoint.step.numpy()

            total_loss_G, total_loss_D, losses_G, losses_D = train_step(x_batch, y_batch)

            #prog_bar.update(
            #    "loss_G={:.4f}, loss_D={:.4f}, lr_G={:.1e}, lr_D={:.1e}".format(
            #        total_loss_G.numpy(), total_loss_D.numpy(),
            #        optimizer_G.lr(steps).numpy(), optimizer_D.lr(steps).numpy()))

            if steps % 10 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar('loss_G/total_loss', total_loss_G, step=steps)
                    tf.summary.scalar('loss_D/total_loss', total_loss_D, step=steps)
                    for k, l in losses_G.items():
                        tf.summary.scalar('loss_G/{}'.format(k), l, step=steps)
                    for k, l in losses_D.items():
                        tf.summary.scalar('loss_D/{}'.format(k), l, step=steps)

                    tf.summary.scalar('learning_rate_G', optimizer_G.lr(steps), step=steps)
                    tf.summary.scalar('learning_rate_D', optimizer_D.lr(steps), step=steps)

        if epoch % 5 == 0:
            manager.save()
            print("\n[*] save ckpt file at {}".format(manager.latest_checkpoint))

        print("\n [*] training done!")
